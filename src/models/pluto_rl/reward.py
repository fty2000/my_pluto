from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PlanR1RewardWeights:
    progress: float = 1.0
    speed: float = 0.5
    comfort: float = 0.2
    margin: float = 0.3
    collision: float = 1.0


def _pairwise_min_distance(ego_xy: torch.Tensor, others_xy: torch.Tensor) -> torch.Tensor:
    """ego_xy: [B, T, 2], others_xy: [B, T, N, 2]"""
    diff = ego_xy.unsqueeze(2) - others_xy
    dist = torch.norm(diff, dim=-1)
    return dist.min(dim=-1).values


def compute_plan_r1_reward(
    ego_traj: torch.Tensor,
    others_traj: torch.Tensor,
    road_signed_distance: torch.Tensor,
    speed_limit: torch.Tensor,
    weights: PlanR1RewardWeights,
    dt: float = 0.1,
    collision_margin: float = 1.0,
    road_margin: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        ego_traj: [B, T, 4] (x, y, cos_h, sin_h)
        others_traj: [B, N, T, 2]
        road_signed_distance: [B, T]
        speed_limit: [B, T]
    """
    ego_xy = ego_traj[..., :2]
    ego_speed = torch.norm(torch.diff(ego_xy, dim=1, prepend=ego_xy[:, :1]) / dt, dim=-1)

    min_dist = _pairwise_min_distance(ego_xy, others_traj[..., :2].transpose(1, 2))
    collision_indicator = (min_dist > 0.0).float()
    road_indicator = (road_signed_distance > 0.0).float()
    safe_gate = collision_indicator * road_indicator

    progress = torch.norm(torch.diff(ego_xy, dim=1, prepend=ego_xy[:, :1]), dim=-1) / dt
    speed_cost = -(
        F.relu(ego_speed - speed_limit).square() + 0.5 * F.relu(0.3 * speed_limit - ego_speed).square()
    )

    acc = torch.diff(ego_speed, dim=1, prepend=ego_speed[:, :1]) / dt
    jerk = torch.diff(acc, dim=1, prepend=acc[:, :1]) / dt
    comfort = -(acc.abs() / 4.0 + jerk.abs() / 8.0)

    margin = -F.softplus(torch.tensor(road_margin, device=ego_traj.device) - road_signed_distance)
    collision = -F.softplus(torch.tensor(collision_margin, device=ego_traj.device) - min_dist)

    dense = (
        weights.progress * progress
        + weights.speed * speed_cost
        + weights.comfort * comfort
        + weights.margin * margin
        + weights.collision * collision
    )
    reward = safe_gate * dense

    return {
        "step_reward": reward,
        "episode_return": reward.sum(-1),
        "safe_gate": safe_gate,
        "collision_indicator": collision_indicator,
        "road_indicator": road_indicator,
    }
