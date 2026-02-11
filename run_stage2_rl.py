import argparse
import os
from typing import Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.models.pluto.probabilistic_pluto_policy import ProbabilisticPlanningModel
from src.models.pluto_rl import (
    ContinuousVDGRPOTrainer,
    PlanR1RewardWeights,
    PlutoWorldModel,
    Stage2TrainConfig,
    VdGrpoConfig,
)


def _to_lightning_style_state_dict(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"model.{k}": v for k, v in model_state.items()}


def export_stage2_checkpoints(model: torch.nn.Module, pt_path: str, ckpt_path: str, epoch: int, global_step: int) -> None:
    state_dict = model.state_dict()
    torch.save(state_dict, pt_path)

    lightning_ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": _to_lightning_style_state_dict(state_dict),
    }
    torch.save(lightning_ckpt, ckpt_path)

def maybe_init_distributed():
    if "RANK" not in os.environ:
        return False, 0, 1, "cpu"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed(enabled: bool):
    if enabled:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Stage-2 Continuous VD-GRPO fine-tuning")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--group_samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--export_pt_path", type=str, default="stage2_vd_grpo_policy.pt")
    parser.add_argument("--export_ckpt_path", type=str, default="stage2_vd_grpo_policy.ckpt")
    args = parser.parse_args()

    ddp, rank, world_size, device = maybe_init_distributed()

    policy = ProbabilisticPlanningModel.from_deterministic_checkpoint(args.stage1_ckpt)
    policy = policy.to(device)

    policy_ref = policy.make_reference_policy().to(device)
    world_model = PlutoWorldModel(policy_ref).to(device)

    if ddp:
        policy = DDP(policy, device_ids=[int(device.split(":")[-1])], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = ContinuousVDGRPOTrainer(
        policy=policy.module if ddp else policy,
        policy_ref=policy_ref,
        world_model=world_model,
        optimizer=optimizer,
        reward_weights=PlanR1RewardWeights(),
        grpo_cfg=VdGrpoConfig(),
        train_cfg=Stage2TrainConfig(group_samples=args.group_samples),
    )

    # Placeholder dataloader: integrate with nuPlan feature cache in production.
    train_loader = DataLoader([], batch_size=1)

    global_step = 0
    for epoch in range(args.epochs):
        trainer.sync_old_policy()
        for batch in train_loader:
            features = batch[0]
            expert_traj = batch[1]
            for k, v in features.items():
                if isinstance(v, dict):
                    features[k] = {kk: vv.to(device) for kk, vv in v.items()}
                else:
                    features[k] = v.to(device)
            expert_traj = expert_traj.to(device)
            logs = trainer.update(features, expert_traj)
            global_step += 1
            if rank == 0:
                print({"epoch": epoch, "global_step": global_step, **logs})

    if rank == 0:
        export_stage2_checkpoints(
            model=(policy.module if ddp else policy),
            pt_path=args.export_pt_path,
            ckpt_path=args.export_ckpt_path,
            epoch=args.epochs,
            global_step=global_step,
        )

    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
