"""ACT-JEPA: Action Chunking Transformer with Joint-Embedding Predictive Architecture.

Extends ACT with an auxiliary JEPA loss: a target encoder (EMA copy of the online
encoder) plus a future-observation predictor head trained jointly with the action head.
At inference only the action branch runs, so cost is identical to vanilla ACT.

Reference: arXiv 2501.14622
"""

import copy
from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
)
from lerobot.policies.act_jepa.configuration_act_jepa import ACTJEPAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTJEPAPolicy(PreTrainedPolicy):
    """ACT-JEPA Policy: ACT + auxiliary JEPA observation-prediction loss.

    During training, the forward pass runs two parallel branches from the shared encoder:
      1. Action decoder → action predictions → L1 action loss
      2. JEPA predictor → predicted future obs latents → L1 vs target encoder output

    During inference, only the action branch runs (identical to vanilla ACT).
    """

    config_class = ACTJEPAConfig
    name = "act_jepa"

    def __init__(self, config: ACTJEPAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTJEPA(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # Exclude target encoder from optimization — it's updated via EMA only.
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.act.backbone") and not n.startswith("model.target_") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.act.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch, compute_jepa=False)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat), jepa_loss = self.model(batch, compute_jepa=True)

        # Action L1 loss (same as vanilla ACT)
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item(), "jepa_loss": jepa_loss.item()}

        # VAE KL-divergence loss
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight + jepa_loss * self.config.jepa_loss_weight
        else:
            loss = l1_loss + jepa_loss * self.config.jepa_loss_weight

        loss_dict["loss"] = loss.item()

        # EMA update of target encoder after each forward pass
        self.model.update_target_encoder()

        return loss, loss_dict


class ACTJEPA(nn.Module):
    """ACT model extended with JEPA target encoder and predictor.

    Architecture:
        Online path (receives gradients):
            observation → backbone → encoder → encoder_out
            encoder_out → decoder + action_head → action predictions
            encoder_out → jepa_predictor + jepa_proj → predicted obs latents

        Target path (EMA, no gradients):
            future observation states → target_state_proj → target_encoder → target latents
    """

    def __init__(self, config: ACTJEPAConfig):
        super().__init__()
        self.config = config

        # ── Online encoder + action decoder (same as vanilla ACT) ──
        self.act = ACT(config)

        # ── JEPA predictor (parallel to action decoder) ──
        prediction_horizon = config.jepa_prediction_horizon

        # Learnable positional embeddings for predictor query tokens
        self.jepa_predictor_pos_embed = nn.Embedding(prediction_horizon, config.dim_model)

        # Predictor: cross-attention transformer (same architecture as action decoder)
        self.jepa_predictor = ACTDecoder(config)

        # Project predictor output to target latent space
        self.jepa_projection_head = nn.Linear(config.dim_model, config.dim_model)

        # ── Target encoder (EMA copy, no gradients) ──
        # Project observation state to the model's hidden dimension
        if config.robot_state_feature:
            self.target_state_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        # Target encoder processes projected states
        self.target_encoder = ACTEncoder(config)
        # Initialize target encoder from online encoder weights
        self._init_target_encoder()

        self._ema_decay = config.jepa_ema_decay

        # Xavier init for new parameters
        for p in chain(self.jepa_predictor.parameters(), self.jepa_projection_head.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_target_encoder(self):
        """Initialize target encoder weights from online encoder."""
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.act.encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: θ_target = decay * θ_target + (1 - decay) * θ_online"""
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.act.encoder.parameters()
        ):
            target_param.data.mul_(self._ema_decay).add_(online_param.data, alpha=1.0 - self._ema_decay)

    def forward(
        self,
        batch: dict[str, Tensor],
        compute_jepa: bool = True,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None], Tensor]:
        """Forward pass with optional JEPA branch.

        Args:
            batch: Training batch. Must contain future observation states when compute_jepa=True.
                   batch[OBS_STATE] shape: (B, horizon, state_dim) during training with delta_timestamps,
                   or (B, state_dim) during inference.
            compute_jepa: Whether to compute the JEPA loss (True during training, False at inference).

        Returns:
            actions: (B, chunk_size, action_dim) predicted actions
            (mu, log_sigma_x2): VAE latent parameters (or (None, None))
            jepa_loss: scalar JEPA loss (0.0 if compute_jepa=False)
        """
        # During training with delta_timestamps, observation.state has shape (B, T, state_dim)
        # where T = prediction_horizon. We use t=0 as the current observation for the encoder,
        # and t=0:T as the target sequence for JEPA. The dataloader applies the same delta
        # indices to ALL observation.* keys, so image streams and *_is_pad flags also arrive
        # with an extra time dim and must be sliced down for the vanilla ACT branch.
        obs_state = batch.get(OBS_STATE)
        has_future_obs = obs_state is not None and obs_state.dim() == 3

        if has_future_obs:
            future_obs_states = obs_state  # (B, T, state_dim) — full sequence including t=0
            future_obs_pad = batch.get(f"{OBS_STATE}_is_pad")  # (B, T) or None
        else:
            future_obs_states = None
            future_obs_pad = None

        # Build a sliced batch with only t=0 observations for the ACT encoder
        act_batch = dict(batch)
        if has_future_obs:
            act_batch[OBS_STATE] = obs_state[:, 0]  # (B, state_dim)
            # Slice image streams: (B, T, C, H, W) -> (B, C, H, W)
            for key in self.config.image_features:
                tensor = act_batch.get(key)
                if tensor is not None and tensor.dim() == 5:
                    act_batch[key] = tensor[:, 0]
            # Slice pad flags for state and image keys: (B, T) -> (B,)
            pad_keys = [f"{OBS_STATE}_is_pad"] + [f"{k}_is_pad" for k in self.config.image_features]
            for pk in pad_keys:
                tensor = act_batch.get(pk)
                if tensor is not None and tensor.dim() == 2:
                    act_batch[pk] = tensor[:, 0]
            # Rebuild OBS_IMAGES list from the sliced per-key tensors
            if self.config.image_features and OBS_IMAGES in act_batch:
                act_batch[OBS_IMAGES] = [act_batch[key] for key in self.config.image_features]

        # Run vanilla ACT forward (encoder + VAE + decoder)
        actions, (mu, log_sigma_x2) = self.act(act_batch)

        # Compute JEPA loss if requested (training only)
        if compute_jepa and future_obs_states is not None:
            jepa_loss = self._compute_jepa_loss(future_obs_states, act_batch, future_obs_pad)
        else:
            jepa_loss = torch.tensor(0.0, device=actions.device)

        return actions, (mu, log_sigma_x2), jepa_loss

    def _compute_jepa_loss(
        self,
        future_obs_states: Tensor,
        batch: dict[str, Tensor],
        future_obs_pad: Tensor | None = None,
    ) -> Tensor:
        """Compute the JEPA observation-prediction loss.

        Args:
            future_obs_states: (B, T, state_dim) future observation states
            batch: sliced training batch (current-frame inputs only) for re-running the
                online encoder. Image and state tensors must already be (B, ...) not (B, T, ...).
            future_obs_pad: optional (B, T) padding mask for future observations.

        Returns:
            Scalar L1 loss between predicted and target latents.
        """
        B, T, state_dim = future_obs_states.shape

        # ── Target branch (no gradients) ──
        with torch.no_grad():
            # Project future states to model dimension
            target_input = self.target_state_proj(future_obs_states)  # (B, T, D)
            # Run through target encoder: expects (T, B, D)
            target_input = target_input.permute(1, 0, 2)  # (T, B, D)
            target_latents = self.target_encoder(target_input)  # (T, B, D)
            target_latents = target_latents.permute(1, 0, 2)  # (B, T, D)

        # ── Predictor branch (receives gradients via encoder_out) ──
        # We need the encoder output from the online encoder. Re-derive it from the ACT's
        # last encoder pass. To avoid redundant computation, we cache it during ACT.forward.
        # For now, we re-run the encoder portion. This is the same computation ACT.forward does.
        encoder_out, encoder_pos_embed = self._get_encoder_output(batch)

        # Predictor input: zero tokens + learnable positional embeddings (DETR-style queries)
        predictor_in = torch.zeros(
            (T, B, self.config.dim_model),
            dtype=encoder_out.dtype,
            device=encoder_out.device,
        )
        predictor_out = self.jepa_predictor(
            predictor_in,
            encoder_out,
            decoder_pos_embed=self.jepa_predictor_pos_embed.weight[:T].unsqueeze(1),
            encoder_pos_embed=encoder_pos_embed,
        )  # (T, B, D)
        predictor_out = predictor_out.permute(1, 0, 2)  # (B, T, D)
        pred_latents = self.jepa_projection_head(predictor_out)  # (B, T, D)

        # ── L1 loss in latent space ──
        # Mask padding if observation padding info is available
        if future_obs_pad is not None:
            mask = ~future_obs_pad  # (B, T)
            loss = (F.l1_loss(pred_latents, target_latents, reduction="none") * mask.unsqueeze(-1)).mean()
        else:
            loss = F.l1_loss(pred_latents, target_latents)

        return loss

    def _get_encoder_output(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Re-derive the online encoder output and positional embeddings.

        This mirrors the encoder portion of ACT.forward() to get encoder_out
        for the JEPA predictor's cross-attention.
        """
        act = self.act
        obs_state = batch.get(OBS_STATE)
        if obs_state is not None and obs_state.dim() == 3:
            obs_state = obs_state[:, 0]  # use current timestep

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Latent sample (zeros during this re-computation — we don't need the VAE path here)
        latent_sample = torch.zeros(
            [batch_size, act.config.latent_dim],
            dtype=torch.float32,
            device=obs_state.device if obs_state is not None else batch[OBS_ENV_STATE].device,
        )

        # Build encoder input tokens
        encoder_in_tokens = [act.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(act.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if act.config.robot_state_feature and obs_state is not None:
            encoder_in_tokens.append(act.encoder_robot_state_input_proj(obs_state))
        if act.config.env_state_feature:
            encoder_in_tokens.append(act.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if act.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = act.backbone(img)["feature_map"]
                cam_pos_embed = act.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = act.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = act.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_out, encoder_in_pos_embed
