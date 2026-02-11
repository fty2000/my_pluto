from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from src.models.pluto_rl.reward import PlanR1RewardWeights, compute_plan_r1_reward
from src.models.pluto_rl.vd_grpo import (
    VdGrpoConfig,
    compute_scene_advantages,
    compute_vd_grpo_loss,
    sampled_kl,
)
from src.models.pluto_rl.world_model import PlutoWorldModel


@dataclass
class Stage2TrainConfig:
    group_samples: int = 4
    gamma: float = 0.99


class ContinuousVDGRPOTrainer:
    """PyTorch trainer helper for Stage-2 continuous VD-GRPO fine-tuning."""

    def __init__(
        self,
        policy: nn.Module,
        policy_ref: nn.Module,
        world_model: PlutoWorldModel,
        optimizer: torch.optim.Optimizer,
        reward_weights: PlanR1RewardWeights,
        grpo_cfg: VdGrpoConfig,
        train_cfg: Stage2TrainConfig,
    ):
        self.policy = policy
        self.policy_old = policy.make_reference_policy()
        self.policy_ref = policy_ref.eval()
        self.world_model = world_model.eval()
        self.optimizer = optimizer
        self.reward_weights = reward_weights
        self.grpo_cfg = grpo_cfg
        self.train_cfg = train_cfg

    @torch.no_grad()
    def collect_group_rollout(self, features: Dict[str, Dict[str, torch.Tensor]]):
        """Collects scene-level grouped samples for VD-GRPO updates."""
        others = self.world_model(features)
        all_samples = []
        for _ in range(self.train_cfg.group_samples):
            sample = self.policy_old.sample(features, deterministic=False)
            road_distance = features["cost_maps"][..., 0].mean(dim=(-2, -1)).unsqueeze(-1).repeat(1, sample["trajectory"].shape[1])
            speed_limit = torch.full_like(road_distance, 13.9)
            reward_info = compute_plan_r1_reward(
                sample["trajectory"],
                others[..., :2],
                road_distance,
                speed_limit,
                self.reward_weights,
            )
            all_samples.append(
                {
                    "mode_index": sample["mode_index"],
                    "eps": sample["eps"],
                    "logp_old": sample["logp"],
                    "return": reward_info["episode_return"],
                }
            )
        return all_samples

    def update(self, features: Dict[str, Dict[str, torch.Tensor]], expert_traj: torch.Tensor) -> Dict[str, float]:
        grouped = self.collect_group_rollout(features)

        mode = torch.stack([x["mode_index"] for x in grouped], dim=1)
        eps = torch.stack([x["eps"] for x in grouped], dim=1)
        logp_old = torch.stack([x["logp_old"] for x in grouped], dim=1)
        ret = torch.stack([x["return"] for x in grouped], dim=1)

        adv = compute_scene_advantages(ret, self.grpo_cfg.c_scale)

        B, M = mode.shape
        mode_flat = mode.reshape(B * M)
        eps_flat = eps.reshape(B * M, -1)

        repeated_features = {
            k: {kk: vv.repeat_interleave(M, dim=0) for kk, vv in v.items()} if isinstance(v, dict) else v.repeat_interleave(M, dim=0)
            for k, v in features.items()
        }

        logp_new = self.policy.logprob(repeated_features, mode_flat, eps_flat).view(B, M)
        logp_ref = self.policy_ref.logprob(repeated_features, mode_flat, eps_flat).view(B, M)

        pg = compute_vd_grpo_loss(logp_new, logp_old, adv, self.grpo_cfg)
        kl = sampled_kl(logp_new, logp_ref)

        warm = self.policy.sample(features, deterministic=True)["trajectory"]
        il_loss = (warm[..., :2] - expert_traj[..., :2]).abs().mean()

        loss = pg["loss_pg"] + self.grpo_cfg.beta_kl * kl + self.grpo_cfg.lambda_il * il_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "loss_pg": float(pg["loss_pg"].detach()),
            "kl": float(kl.detach()),
            "il_loss": float(il_loss.detach()),
            "mean_return": float(ret.mean().detach()),
        }

    def sync_old_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
