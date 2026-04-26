"""Processor for ACT-JEPA policy. Reuses ACT's pre/post processors."""

from typing import Any

import torch

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.act_jepa.configuration_act_jepa import ACTJEPAConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_act_jepa_pre_post_processors(
    config: ACTJEPAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates pre/post processors for ACT-JEPA. Same as ACT."""
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
