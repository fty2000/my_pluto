import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from omegaconf import DictConfig, OmegaConf

from src.custom_training.custom_training_builder import (
    build_lightning_datamodule,
    update_config_for_training,
)
from src.rl import IDMConfig, PlutoWorldModel, ProbPlutoPolicy, VDGRPOConfig, compute_planr1_style_reward, compute_vd_grpo_loss

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

set_default_path()
CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"


def _to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_device(v, device) for v in data)
    return data


def _inject_rl_defaults(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    if "rl" not in cfg:
        cfg.rl = {}

    defaults = {
        "enabled": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 5,
        "steps_per_epoch": 500,
        "val_steps": 100,
        "group_size": 4,
        "speed_limit": 13.9,
        "optimizer_lr": 1e-4,
        "optimizer_weight_decay": 1e-4,
        "grad_clip_norm": 5.0,
        "advantage_scale": 0.1,
        "beta_kl": 0.02,
        "lambda_il": 0.05,
        "old_policy_sync_interval": 50,
        "save_dir": "rl_checkpoints",
        "resume_ckpt": None,
        "ref_ckpt": None,
        "idm_enabled": True,
        "idm_a_max": 1.5,
        "idm_v0": 15.0,
        "idm_delta": 4.0,
        "idm_min_gap": 2.0,
        "idm_headway": 1.5,
    }
    for k, v in defaults.items():
        if k not in cfg.rl:
            cfg.rl[k] = v

    cfg.model.probabilistic_policy = True
    if "latent_dim" not in cfg.model:
        cfg.model.latent_dim = 8

    OmegaConf.set_struct(cfg, True)


def _load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)

    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded checkpoint %s (missing=%d, unexpected=%d)", ckpt_path, len(missing), len(unexpected))


def _compute_collision_distance(ego_traj: torch.Tensor, others_traj: torch.Tensor) -> torch.Tensor:
    # ego_traj: [B, T, 4], others_traj: [B, A, T, 6]
    ego_xy = ego_traj[..., :2].unsqueeze(1)
    others_xy = others_traj[..., :2]
    d = torch.norm(ego_xy - others_xy, dim=-1) - 2.0  # rough vehicle-circle margin
    return d.min(dim=1).values


def _compute_road_distance(ego_traj: torch.Tensor, radius: float) -> torch.Tensor:
    # fallback ESDF surrogate: keep trajectory inside feature radius
    return radius - torch.norm(ego_traj[..., :2], dim=-1)


def _small_il_loss(model_out: Dict[str, torch.Tensor], data: Dict[str, Any]) -> torch.Tensor:
    target_xy = data["agent"]["target"][:, 0, :, :2]
    logits = model_out["policy_logits"]
    best_idx = logits.argmax(dim=-1)
    batch = torch.arange(logits.shape[0], device=logits.device)
    pred_xy = model_out["prob_trajectory_mu"][batch, best_idx, :, :2]
    return F.smooth_l1_loss(pred_xy, target_xy)


def _validate(
    model: torch.nn.Module,
    world_model: PlutoWorldModel,
    val_loader,
    device: torch.device,
    speed_limit: float,
    max_steps: int,
) -> float:
    model.eval()
    policy = ProbPlutoPolicy(model, deterministic_latent=False)
    total_reward = 0.0
    count = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= max_steps:
                break
            features, _, _ = batch
            data = _to_device(features["feature"].data, device)
            sample = policy.sample(data)
            others = world_model.predict_others(data)

            d_collision = _compute_collision_distance(sample.trajectory, others)
            d_road = _compute_road_distance(sample.trajectory, radius=model.radius)
            speed_limit_t = torch.full_like(d_road, speed_limit)
            reward = compute_planr1_style_reward(
                sample.trajectory,
                d_collision,
                d_road,
                speed_limit_t,
                weights={},
            ).sum(dim=-1)
            total_reward += reward.mean().item()
            count += 1
    return total_reward / max(count, 1)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    _inject_rl_defaults(cfg)

    build_logger(cfg)
    update_config_for_training(cfg)
    build_training_experiment_folder(cfg=cfg)

    worker = build_worker(cfg)

    device = torch.device(cfg.rl.device)

    with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_rl_components"):
        model = build_torch_module_wrapper(cfg.model).to(device)
        if cfg.rl.resume_ckpt:
            _load_model_weights(model, cfg.rl.resume_ckpt)
        elif cfg.checkpoint:
            _load_model_weights(model, cfg.checkpoint)

        ref_model = copy.deepcopy(model).to(device)
        if cfg.rl.ref_ckpt:
            _load_model_weights(ref_model, cfg.rl.ref_ckpt)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        wm_model = copy.deepcopy(model).to(device)
        wm_model.eval()
        for p in wm_model.parameters():
            p.requires_grad_(False)

        datamodule = build_lightning_datamodule(cfg, worker, model)
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.rl.optimizer_lr,
        weight_decay=cfg.rl.optimizer_weight_decay,
    )

    vd_cfg = VDGRPOConfig(
        advantage_scale=cfg.rl.advantage_scale,
        beta_kl=cfg.rl.beta_kl,
        lambda_il=cfg.rl.lambda_il,
    )

    idm_cfg = IDMConfig(
        enabled=cfg.rl.idm_enabled,
        a_max=cfg.rl.idm_a_max,
        v0=cfg.rl.idm_v0,
        delta=cfg.rl.idm_delta,
        min_gap=cfg.rl.idm_min_gap,
        headway=cfg.rl.idm_headway,
    )
    world_model = PlutoWorldModel(wm_model, idm=idm_cfg)

    old_model = copy.deepcopy(model).to(device)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    save_dir = Path(cfg.output_dir) / cfg.rl.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = -1e9

    for epoch in range(cfg.rl.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0

        for step, batch in enumerate(train_loader):
            if step >= cfg.rl.steps_per_epoch:
                break

            features, _, _ = batch
            data = _to_device(features["feature"].data, device)

            with torch.no_grad():
                others = world_model.predict_others(data)

            old_policy = ProbPlutoPolicy(old_model)
            mode_list, latent_list, logp_old_list, reward_list = [], [], [], []

            for _ in range(cfg.rl.group_size):
                with torch.no_grad():
                    sample_old = old_policy.sample(data)

                d_collision = _compute_collision_distance(sample_old.trajectory, others)
                d_road = _compute_road_distance(sample_old.trajectory, radius=model.radius)
                speed_limit_t = torch.full_like(d_road, cfg.rl.speed_limit)
                reward_t = compute_planr1_style_reward(
                    sample_old.trajectory,
                    d_collision,
                    d_road,
                    speed_limit_t,
                    weights={},
                )
                reward = reward_t.sum(dim=-1)

                mode_list.append(sample_old.mode_index)
                latent_list.append(sample_old.latent)
                logp_old_list.append(sample_old.logp)
                reward_list.append(reward)

            mode = torch.stack(mode_list, dim=1)
            latent = torch.stack(latent_list, dim=1)
            logp_old = torch.stack(logp_old_list, dim=1)
            reward = torch.stack(reward_list, dim=1)

            out_new = model(data)
            new_policy = ProbPlutoPolicy(model)
            ref_policy = ProbPlutoPolicy(ref_model)

            logp_new_parts, logp_ref_parts = [], []
            for g in range(cfg.rl.group_size):
                logp_new_parts.append(
                    new_policy.logprob(data, mode[:, g], latent[:, g], cached_out=out_new)
                )
                with torch.no_grad():
                    logp_ref_parts.append(ref_policy.logprob(data, mode[:, g], latent[:, g]))

            logp_new = torch.stack(logp_new_parts, dim=1)
            logp_ref = torch.stack(logp_ref_parts, dim=1)

            il_loss = _small_il_loss(out_new, data)

            loss_dict = compute_vd_grpo_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                reward=reward,
                logp_ref=logp_ref,
                il_loss=il_loss,
                cfg=vd_cfg,
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.rl.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reward += reward.mean().item()
            global_step += 1

            if global_step % cfg.rl.old_policy_sync_interval == 0:
                old_model.load_state_dict(model.state_dict())

            if global_step % 20 == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.4f reward=%.4f pg=%.4f kl=%.4f",
                    epoch,
                    global_step,
                    loss.item(),
                    reward.mean().item(),
                    loss_dict["loss_pg"].item(),
                    loss_dict["kl"].item(),
                )

        avg_loss = epoch_loss / max(1, min(cfg.rl.steps_per_epoch, step + 1))
        avg_reward = epoch_reward / max(1, min(cfg.rl.steps_per_epoch, step + 1))

        val_reward = _validate(
            model=model,
            world_model=world_model,
            val_loader=val_loader,
            device=device,
            speed_limit=cfg.rl.speed_limit,
            max_steps=cfg.rl.val_steps,
        )

        logger.info(
            "[epoch %d] train_loss=%.4f train_reward=%.4f val_reward=%.4f",
            epoch,
            avg_loss,
            avg_reward,
            val_reward,
        )

        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "metrics": {
                "train_loss": avg_loss,
                "train_reward": avg_reward,
                "val_reward": val_reward,
            },
        }

        epoch_ckpt = save_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt_payload, epoch_ckpt)

        if val_reward > best_val:
            best_val = val_reward
            torch.save(ckpt_payload, save_dir / "best.pt")

    logger.info("RL training finished. best_val_reward=%.4f", best_val)


if __name__ == "__main__":
    main()
