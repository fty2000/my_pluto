from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class IDMConfig:
    enabled: bool = True
    a_max: float = 1.5
    v0: float = 15.0
    delta: float = 4.0
    min_gap: float = 2.0
    headway: float = 1.5


class PlutoWorldModel:
    """Frozen world model wrapper using PLUTO agent prediction head + optional IDM speed correction."""

    def __init__(self, model, idm: Optional[IDMConfig] = None):
        self.model = model
        self.idm = idm or IDMConfig()

    @torch.no_grad()
    def predict_others(self, data: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        out = self.model(data)
        # prediction excludes ego by model design: [B, A-1, T, 6]
        return out["prediction"]

    def apply_idm_speed_patch(
        self,
        predicted_others: torch.Tensor,
        ego_speed: torch.Tensor,
        distance_to_lead: torch.Tensor,
        relative_speed: torch.Tensor,
    ) -> torch.Tensor:
        if not self.idm.enabled:
            return predicted_others

        dt = 0.1
        base_speed = predicted_others[..., 4].clamp_min(0.0)
        s_star = self.idm.min_gap + base_speed * self.idm.headway + (
            base_speed * relative_speed
        ) / (2 * torch.sqrt(torch.tensor(self.idm.a_max, device=base_speed.device)).clamp_min(1e-3))
        accel = self.idm.a_max * (
            1 - (base_speed / self.idm.v0).pow(self.idm.delta) - (s_star / distance_to_lead.clamp_min(1.0)).pow(2)
        )
        idm_speed = (base_speed + accel * dt).clamp_min(0.0)
        corrected = predicted_others.clone()
        corrected[..., 4] = torch.minimum(base_speed, idm_speed)
        return corrected
