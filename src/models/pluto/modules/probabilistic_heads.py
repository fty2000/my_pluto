import math

import torch
import torch.nn as nn


class ProbTrajHead(nn.Module):
    """Trajectory head with latent-noise injection."""

    def __init__(self, d_q: int, d_z: int, hidden_dim: int, future_steps: int, traj_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_q + d_z, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, future_steps * traj_dim),
        )
        self.future_steps = future_steps
        self.traj_dim = traj_dim

    def forward(self, query_feature: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([query_feature, latent_z], dim=-1)
        out = self.fc(feat)
        return out.view(-1, self.future_steps, self.traj_dim)


class TrajUncertaintyHead(nn.Module):
    """Uncertainty head for Stage-1 warm-up Gaussian NLL over x/y channels."""

    def __init__(self, d_q: int, future_steps: int, min_log_sigma: float = -5.0, max_log_sigma: float = 2.0):
        super().__init__()
        self.fc = nn.Linear(d_q, future_steps * 2)
        self.future_steps = future_steps
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

    def forward(self, query_feature: torch.Tensor) -> torch.Tensor:
        log_sigma = self.fc(query_feature).view(-1, self.future_steps, 2)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return torch.exp(log_sigma)


def gaussian_log_prob_from_eps(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Log probability of a diagonal Gaussian sample represented as z = eps * exp(log_std)."""
    dim = eps.shape[-1]
    quadratic = (eps**2).sum(-1)
    return -0.5 * (quadratic + 2 * log_std.sum() + dim * math.log(2 * math.pi))
