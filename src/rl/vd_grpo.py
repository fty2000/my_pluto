from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class VDGRPOConfig:
    advantage_scale: float = 0.1
    beta_kl: float = 0.02
    lambda_il: float = 0.05


def scene_wise_advantage(reward: torch.Tensor, scale: float) -> torch.Tensor:
    """reward: [B, M] where M=Q*G for each scene."""
    mean_reward = reward.mean(dim=-1, keepdim=True)
    return scale * (reward - mean_reward)


def compute_vd_grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    reward: torch.Tensor,
    logp_ref: torch.Tensor,
    il_loss: torch.Tensor,
    cfg: VDGRPOConfig,
) -> Dict[str, torch.Tensor]:
    adv = scene_wise_advantage(reward, cfg.advantage_scale)
    ratio = torch.exp(logp_new - logp_old)
    loss_pg = -(ratio * adv.detach()).mean()

    kl = (torch.exp(logp_new) * (logp_new - logp_ref)).mean()
    total = loss_pg + cfg.beta_kl * kl + cfg.lambda_il * il_loss

    return {
        "loss": total,
        "loss_pg": loss_pg.detach(),
        "kl": kl.detach(),
        "adv_mean": adv.mean().detach(),
    }
