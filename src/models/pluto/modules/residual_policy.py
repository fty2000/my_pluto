import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def decode_action_xy(a: torch.Tensor, k: int, dx_max: float = 0.5, dy_max: float = 0.5):
    a = a.view(k, 2)
    delta_px_k = dx_max * torch.tanh(a[:, 0])
    delta_py_k = dy_max * torch.tanh(a[:, 1])
    return delta_px_k, delta_py_k


def interp_to_all_steps(delta_k: torch.Tensor, key_idx: List[int], t_steps: int):
    delta_all = torch.zeros(t_steps, device=delta_k.device, dtype=delta_k.dtype)
    delta_all[: key_idx[0] + 1] = delta_k[0]

    for i in range(len(key_idx) - 1):
        s, e = key_idx[i], key_idx[i + 1]
        v_s, v_e = delta_k[i], delta_k[i + 1]
        length = e - s
        for t in range(1, length + 1):
            alpha = t / (length + 1e-6)
            delta_all[s + t] = (1 - alpha) * v_s + alpha * v_e

    delta_all[key_idx[-1] :] = delta_k[-1]
    return delta_all


def decode_residual_xy(
    a: torch.Tensor,
    tau_base: torch.Tensor,
    key_idx: List[int],
    dt: float,
    dx_max: float = 0.5,
    dy_max: float = 0.5,
):
    t_steps = tau_base.shape[0]
    k = len(key_idx)

    delta_px_k, delta_py_k = decode_action_xy(a, k, dx_max, dy_max)
    delta_px_all = interp_to_all_steps(delta_px_k, key_idx, t_steps)
    delta_py_all = interp_to_all_steps(delta_py_k, key_idx, t_steps)

    pos_base = tau_base[:, 0:2]
    pos_new = pos_base + torch.stack([delta_px_all, delta_py_all], dim=-1)

    vel_new = torch.zeros_like(pos_new)
    vel_new[0] = tau_base[0, 4:6]
    vel_new[1:] = (pos_new[1:] - pos_new[:-1]) / dt

    theta = torch.atan2(vel_new[:, 1], vel_new[:, 0] + 1e-6)
    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)

    delta = torch.zeros_like(tau_base)
    delta[:, 0] = pos_new[:, 0] - tau_base[:, 0]
    delta[:, 1] = pos_new[:, 1] - tau_base[:, 1]
    delta[:, 2] = cos_th - tau_base[:, 2]
    delta[:, 3] = sin_th - tau_base[:, 3]
    delta[:, 4] = vel_new[:, 0] - tau_base[:, 4]
    delta[:, 5] = vel_new[:, 1] - tau_base[:, 5]

    return delta


def project_to_feasible(tau: torch.Tensor, v_max: float = 30.0):
    tau = tau.clone()
    v = tau[:, 4:6]
    speed = v.norm(dim=-1, keepdim=True)
    speed_clamped = torch.clamp(speed, max=v_max)
    tau[:, 4:6] = v * (speed_clamped / (speed + 1e-6))
    return tau


class ResidualHead(nn.Module):
    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, 2 * k),
        )
        self.log_std = nn.Parameter(torch.full((2 * k,), -1.0))

        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, q_sel: torch.Tensor, eenc: torch.Tensor, e_av: torch.Tensor):
        e_scene = eenc.mean(dim=0)
        h = torch.cat([q_sel, e_av.squeeze(0), e_scene], dim=-1)
        mu = self.fc(h)
        log_std = self.log_std
        return mu, log_std


def sample_action(mu: torch.Tensor, log_std: torch.Tensor):
    std = log_std.exp()
    eps = torch.randn_like(mu)
    a = mu + std * eps
    log_prob = -0.5 * (
        (((a - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi)).sum()
    )
    return a, log_prob
