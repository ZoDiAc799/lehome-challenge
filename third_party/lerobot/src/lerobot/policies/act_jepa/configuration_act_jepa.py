"""Configuration for ACT-JEPA policy.

ACT-JEPA extends ACT with an auxiliary JEPA (Joint-Embedding Predictive Architecture) loss.
A target encoder (EMA of the online encoder) and a predictor head are trained jointly with
the action head. At inference, only the action branch runs.

Reference: arXiv 2501.14622
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_jepa")
@dataclass
class ACTJEPAConfig(ACTConfig):
    """Configuration for ACT-JEPA policy.

    Inherits all ACT parameters and adds JEPA-specific ones.

    Additional args:
        jepa_ema_decay: Exponential moving average decay for the target encoder.
        jepa_loss_weight: Weight of the JEPA observation-prediction loss relative to the action loss.
        jepa_prediction_horizon: Number of future timesteps to predict. Defaults to chunk_size if None.
        jepa_predictor_n_layers: Number of transformer layers in the JEPA predictor.
    """

    # JEPA-specific parameters
    jepa_ema_decay: float = 0.999
    jepa_loss_weight: float = 1.0
    jepa_prediction_horizon: int | None = None  # defaults to chunk_size
    jepa_predictor_n_layers: int = 1  # match action decoder depth by default

    def __post_init__(self):
        super().__post_init__()
        if self.jepa_prediction_horizon is None:
            self.jepa_prediction_horizon = self.chunk_size

    @property
    def observation_delta_indices(self) -> list | None:
        """Request future observation states for JEPA targets."""
        horizon = self.jepa_prediction_horizon if self.jepa_prediction_horizon else self.chunk_size
        return list(range(horizon))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
