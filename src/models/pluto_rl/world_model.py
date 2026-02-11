from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class IDMConfig:
    a_max: float = 1.0
    v0: float = 15.0
    delta: float = 4.0
    min_gap: float = 2.0
    time_headway: float = 1.5


class PlutoWorldModel(torch.nn.Module):
    """Frozen world model wrapper based on PLUTO agent prediction head + IDM patch."""

    def __init__(self, policy_model: torch.nn.Module, idm_cfg: Optional[IDMConfig] = None):
        super().__init__()
        self.policy_model = policy_model
        self.idm_cfg = idm_cfg or IDMConfig()
        for p in self.policy_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, features: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        out = self.policy_model(features)
        return out["prediction"]

    def apply_idm_patch(
        self,
        others_prediction: torch.Tensor,
        ego_speed: torch.Tensor,
        front_mask: torch.Tensor,
        headway_distance: torch.Tensor,
        rel_speed: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 1D IDM speed correction to front-vehicle candidates in ego lane."""
        cfg = self.idm_cfg
        exec_prediction = others_prediction.clone()
        speed = torch.norm(exec_prediction[..., 4:6], dim=-1)

        desired_gap = cfg.min_gap + ego_speed.unsqueeze(-1) * cfg.time_headway
        desired_gap = desired_gap + (ego_speed.unsqueeze(-1) * rel_speed) / (2 * (cfg.a_max + 1e-3) ** 0.5)
        idm_acc = cfg.a_max * (
            1 - (speed / cfg.v0).pow(cfg.delta) - (desired_gap / headway_distance.clamp_min(1e-2)).pow(2)
        )
        corrected_speed = torch.clamp(speed + idm_acc, min=0.0)
        speed = torch.where(front_mask, torch.minimum(speed, corrected_speed), speed)

        heading = torch.atan2(exec_prediction[..., 3], exec_prediction[..., 2])
        exec_prediction[..., 4] = speed * torch.cos(heading)
        exec_prediction[..., 5] = speed * torch.sin(heading)
        return exec_prediction
