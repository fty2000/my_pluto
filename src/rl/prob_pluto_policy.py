import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class PolicySample:
    mode_index: torch.Tensor
    latent: torch.Tensor
    trajectory: torch.Tensor
    logp_disc: torch.Tensor
    logp_cont: torch.Tensor

    @property
    def logp(self) -> torch.Tensor:
        return self.logp_disc + self.logp_cont


class ProbPlutoPolicy:
    """Sampling/log-prob interface for probabilistic PLUTO planner."""

    def __init__(self, model, deterministic_latent: bool = False):
        self.model = model
        self.deterministic_latent = deterministic_latent

    def _gather_query(self, query: torch.Tensor, mode_index: torch.Tensor) -> torch.Tensor:
        batch_index = torch.arange(query.shape[0], device=query.device)
        return query[batch_index, mode_index]

    def _latent_sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.deterministic_latent:
            return torch.zeros(batch_size, self.model.latent_dim, device=device)
        eps = torch.randn(batch_size, self.model.latent_dim, device=device)
        return eps * torch.exp(self.model.log_std_z)

    def sample(self, data: Dict[str, Dict[str, torch.Tensor]]) -> PolicySample:
        out = self.model(data)
        logits = out["policy_logits"]
        probs = logits.softmax(dim=-1)
        mode_index = torch.distributions.Categorical(probs=probs).sample()

        z = self._latent_sample(logits.shape[0], logits.device)
        q_i = self._gather_query(out["policy_query"], mode_index)
        traj = self.model.prob_traj_head(q_i, z)

        logp_disc = probs.gather(1, mode_index.unsqueeze(-1)).clamp_min(1e-12).log().squeeze(-1)
        sigma = torch.exp(self.model.log_std_z)
        eps = z / sigma.clamp_min(1e-8)
        logp_cont = -0.5 * (
            (eps**2).sum(dim=-1)
            + 2 * self.model.log_std_z.sum()
            + self.model.latent_dim * math.log(2 * math.pi)
        )

        return PolicySample(
            mode_index=mode_index,
            latent=z,
            trajectory=traj,
            logp_disc=logp_disc,
            logp_cont=logp_cont,
        )

    def logprob(
        self,
        data: Dict[str, Dict[str, torch.Tensor]],
        mode_index: torch.Tensor,
        latent: torch.Tensor,
        cached_out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        out = cached_out if cached_out is not None else self.model(data)
        logits = out["policy_logits"]
        logp_disc_all = logits.log_softmax(dim=-1)
        logp_disc = logp_disc_all.gather(1, mode_index.unsqueeze(-1)).squeeze(-1)

        sigma = torch.exp(self.model.log_std_z)
        eps = latent / sigma.clamp_min(1e-8)
        logp_cont = -0.5 * (
            (eps**2).sum(dim=-1)
            + 2 * self.model.log_std_z.sum()
            + self.model.latent_dim * math.log(2 * math.pi)
        )

        return logp_disc + logp_cont
