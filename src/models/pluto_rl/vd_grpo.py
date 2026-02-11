from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class VdGrpoConfig:
    c_scale: float = 0.1
    beta_kl: float = 0.02
    lambda_il: float = 0.05
    ratio_clip: float = 0.2
    use_ratio_clip: bool = True


def compute_scene_advantages(reward: torch.Tensor, c_scale: float) -> torch.Tensor:
    """reward: [B, M] where M is scene-level group size."""
    return c_scale * (reward - reward.mean(dim=-1, keepdim=True))


def compute_vd_grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
    config: VdGrpoConfig,
) -> Dict[str, torch.Tensor]:
    ratio = torch.exp(logp_new - logp_old)
    if config.use_ratio_clip:
        ratio_clip = ratio.clamp(1.0 - config.ratio_clip, 1.0 + config.ratio_clip)
        pg = torch.minimum(ratio * advantage.detach(), ratio_clip * advantage.detach())
    else:
        pg = ratio * advantage.detach()
    loss_pg = -pg.mean()
    return {"loss_pg": loss_pg, "ratio": ratio}


def sampled_kl(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    weights = torch.exp(logp_new).detach()
    return (weights * (logp_new - logp_ref)).mean()
