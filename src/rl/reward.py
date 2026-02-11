from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PlanR1RewardConfig:
    """Plan-R1 style reward config.

    Reward per step:
        R_t = G_t * S_t + (1 - G_t) * unsafe_penalty

    where
        G_t = Π_k I_safe,k(t)
        S_t = Σ_m w_m * c_m(t)

    By default unsafe_penalty=0, matching the strict gate style used in Plan-R1 descriptions.
    """

    dt: float = 0.1
    unsafe_penalty: float = 0.0

    # safety-margin shaping
    beta_collision: float = 1.0
    alpha_road: float = 0.5

    # speed/comfort normalization
    min_speed: float = 0.0
    lambda_slow: float = 0.25
    a_max: float = 4.0
    j_max: float = 8.0

    # weighted soft objectives
    w_progress: float = 1.0
    w_speed: float = 0.2
    w_comfort: float = 0.05
    w_margin: float = 0.3
    w_collision: float = 1.0


@dataclass
class PlanR1RewardTerms:
    """Inputs to compute Plan-R1 style reward.

    All tensors are [B, T].
    """

    d_collision: torch.Tensor
    d_road: torch.Tensor
    speed_limit: torch.Tensor
    rule_safe: Optional[torch.Tensor] = None


def _speed_from_traj(ego_traj: torch.Tensor, dt: float) -> torch.Tensor:
    pos = ego_traj[..., :2]
    vel_vec = (pos[:, 1:] - pos[:, :-1]) / dt
    return torch.cat([vel_vec.norm(dim=-1), vel_vec[:, -1:].norm(dim=-1)], dim=1)


def compute_planr1_reward(
    ego_traj: torch.Tensor,
    terms: PlanR1RewardTerms,
    cfg: PlanR1RewardConfig,
) -> torch.Tensor:
    """Compute Plan-R1 style gated multi-objective reward.

    Args:
        ego_traj: [B, T, >=2], ego trajectory in local frame.
        terms: distances / limits / safety-indicator inputs.
        cfg: reward coefficients.
    Returns:
        reward: [B, T]
    """
    speed = _speed_from_traj(ego_traj, cfg.dt)

    i_coll = (terms.d_collision > 0).float()
    i_road = (terms.d_road > 0).float()
    i_rule = (
        terms.rule_safe.float()
        if terms.rule_safe is not None
        else torch.ones_like(i_coll)
    )

    gate = i_coll * i_road * i_rule

    # dense safety shaping
    c_collision = -F.softplus(cfg.beta_collision - terms.d_collision)
    c_margin = -F.softplus(cfg.alpha_road - terms.d_road)

    # speed objective
    speed_over = (speed - terms.speed_limit).clamp_min(0.0)
    speed_under = (cfg.min_speed - speed).clamp_min(0.0)
    c_speed = -(speed_over.square() + cfg.lambda_slow * speed_under.square())

    # comfort objective
    accel = torch.diff(speed, dim=1, prepend=speed[:, :1]) / cfg.dt
    jerk = torch.diff(accel, dim=1, prepend=accel[:, :1]) / cfg.dt
    c_comfort = -(accel.abs() / cfg.a_max + jerk.abs() / cfg.j_max)

    # progress objective
    c_progress = speed

    soft = (
        cfg.w_progress * c_progress
        + cfg.w_speed * c_speed
        + cfg.w_comfort * c_comfort
        + cfg.w_margin * c_margin
        + cfg.w_collision * c_collision
    )

    return gate * soft + (1.0 - gate) * cfg.unsafe_penalty


# backward-compatible name used by existing code

def compute_planr1_style_reward(
    ego_traj: torch.Tensor,
    d_collision: torch.Tensor,
    d_road: torch.Tensor,
    speed_limit: torch.Tensor,
    weights,
    dt: float = 0.1,
) -> torch.Tensor:
    cfg = PlanR1RewardConfig(dt=dt)
    for k, v in dict(weights).items():
        if hasattr(cfg, k):
            setattr(cfg, k, float(v))
    terms = PlanR1RewardTerms(
        d_collision=d_collision,
        d_road=d_road,
        speed_limit=speed_limit,
    )
    return compute_planr1_reward(ego_traj=ego_traj, terms=terms, cfg=cfg)
