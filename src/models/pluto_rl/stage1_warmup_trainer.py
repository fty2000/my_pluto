from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from src.models.pluto.pluto_trainer import LightningTrainer


class ProbabilisticWarmupTrainer(LightningTrainer):
    """Stage-1 trainer: IL + uncertainty warm-up on top of ProbabilisticPlanningModel."""

    def __init__(
        self,
        lambda_nll: float = 0.1,
        lambda_nll_end: float = 1.0,
        lambda_nll_ramp_ratio: float = 0.3,
        freeze_log_std_z_epochs: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_nll = lambda_nll
        self.lambda_nll_start = lambda_nll
        self.lambda_nll_end = lambda_nll_end
        self.lambda_nll_ramp_ratio = lambda_nll_ramp_ratio
        self.freeze_log_std_z_epochs = freeze_log_std_z_epochs


    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        ramp_epochs = max(1, int(self.epochs * self.lambda_nll_ramp_ratio))
        progress = min(1.0, float(self.current_epoch + 1) / float(ramp_epochs))
        self.lambda_nll = self.lambda_nll_start + progress * (
            self.lambda_nll_end - self.lambda_nll_start
        )

        has_log_std = hasattr(self.model, "log_std_z") and self.model.log_std_z is not None
        if has_log_std:
            self.model.log_std_z.requires_grad = self.current_epoch >= self.freeze_log_std_z_epochs

        self.log(
            "stage1/lambda_nll",
            float(self.lambda_nll),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _gaussian_nll(self, mu_xy: torch.Tensor, sigma_xy: torch.Tensor, gt_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        diff = gt_xy - mu_xy
        var = sigma_xy.square().clamp_min(1e-6)
        nll = 0.5 * (diff.square() / var + 2 * sigma_xy.log() + torch.log(torch.tensor(2 * torch.pi, device=mu_xy.device)))
        nll = nll.sum(-1)
        return (nll * valid_mask).sum() / valid_mask.sum().clamp_min(1)

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
        features, targets, scenarios = batch
        data = features["feature"].data
        out = self.forward(data)

        losses = self._compute_objectives(out, data)
        self._log_step(losses["loss"], losses, None, prefix)
        return losses["loss"] if self.training else torch.zeros(1, device=self.device)

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        base_losses = super()._compute_objectives(res, data)
        if "trajectory_uncertainty" not in res:
            base_losses["nll_loss"] = 0.0
            return base_losses

        trajectory = res["trajectory"]
        sigma = res["trajectory_uncertainty"]
        bs, R, M, T, _ = trajectory.shape

        valid_mask = data["agent"]["valid_mask"][:, 0, -T:]
        targets_pos = data["agent"]["target"][:, 0, :, :2]

        modes = trajectory[..., :2].reshape(bs, R * M, T, 2)
        dist = ((modes - targets_pos.unsqueeze(1)) ** 2).sum(dim=(-1, -2))
        best_mode = dist.argmin(-1)
        batch_idx = torch.arange(bs, device=trajectory.device)

        mu_best = modes[batch_idx, best_mode]
        sigma_best = sigma[batch_idx, best_mode]
        loss_nll = self._gaussian_nll(mu_best, sigma_best, targets_pos, valid_mask)

        total = base_losses["loss"] + self.lambda_nll * loss_nll
        base_losses["loss"] = total
        base_losses["nll_loss"] = loss_nll.item()
        return base_losses
