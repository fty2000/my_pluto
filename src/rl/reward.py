from typing import Dict

import torch
import torch.nn.functional as F


def compute_planr1_style_reward(
    ego_traj: torch.Tensor,
    d_collision: torch.Tensor,
    d_road: torch.Tensor,
    speed_limit: torch.Tensor,
    weights: Dict[str, float],
    dt: float = 0.1,
) -> torch.Tensor:
    """Compute dense Plan-R1 style reward with safety gating and ESDF shaping.

    Args:
        ego_traj: [B, T, 4] (x, y, cos_h, sin_h) or [B, T, >=4], speed inferred from xy.
        d_collision: [B, T] minimum distance to obstacles / agents.
        d_road: [B, T] ESDF signed distance to drivable boundary (positive if safe).
        speed_limit: [B, T].
    """
    pos = ego_traj[..., :2]
    vel_vec = (pos[:, 1:] - pos[:, :-1]) / dt
    speed = torch.cat([vel_vec.norm(dim=-1), vel_vec[:, -1:].norm(dim=-1)], dim=1)

    i_coll = (d_collision > 0).float()
    i_road = (d_road > 0).float()
    safe_gate = i_coll * i_road

    c_collision = -F.softplus(weights.get("beta_collision", 1.0) - d_collision)
    c_margin = -F.softplus(weights.get("alpha_road", 0.5) - d_road)

    speed_over = (speed - speed_limit).clamp_min(0.0)
    speed_under = (weights.get("min_speed", 0.0) - speed).clamp_min(0.0)
    c_speed = -(speed_over**2 + weights.get("lambda_slow", 0.25) * speed_under**2)

    accel = torch.diff(speed, dim=1, prepend=speed[:, :1]) / dt
    jerk = torch.diff(accel, dim=1, prepend=accel[:, :1]) / dt
    c_comfort = -(accel.abs() / weights.get("a_max", 4.0) + jerk.abs() / weights.get("j_max", 8.0))

    progress = speed

    soft = (
        weights.get("w_progress", 1.0) * progress
        + weights.get("w_speed", 0.2) * c_speed
        + weights.get("w_comfort", 0.05) * c_comfort
        + weights.get("w_margin", 0.3) * c_margin
        + weights.get("w_collision", 1.0) * c_collision
    )
    return safe_gate * soft
