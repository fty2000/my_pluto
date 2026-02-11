import torch
import torch.nn as nn


class ProbTrajectoryHead(nn.Module):
    """Trajectory decoder conditioned on query embedding and low-dimensional latent noise."""

    def __init__(self, d_q: int, d_z: int, future_steps: int, traj_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.future_steps = future_steps
        self.traj_dim = traj_dim
        self.net = nn.Sequential(
            nn.Linear(d_q + d_z, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, future_steps * traj_dim),
        )

    def forward(self, q_i: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([q_i, z], dim=-1)
        traj = self.net(feat)
        return traj.view(-1, self.future_steps, self.traj_dim)


class TrajUncertaintyHead(nn.Module):
    """Predict per-step XY aleatoric uncertainty for Stage-1 warm-up NLL."""

    def __init__(self, d_q: int, future_steps: int, min_log_sigma: float = -5.0, max_log_sigma: float = 2.0):
        super().__init__()
        self.future_steps = future_steps
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.fc = nn.Linear(d_q, future_steps * 2)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        log_sigma = self.fc(q).view(-1, self.future_steps, 2)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return torch.exp(log_sigma)
