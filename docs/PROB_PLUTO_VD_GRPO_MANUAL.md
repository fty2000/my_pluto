# Probabilistic-PLUTO + Continuous VD-GRPO 操作手册

本文档给出从 `PLUTO` 迁移到 `Probabilistic-PLUTO + Continuous VD-GRPO` 的完整落地步骤，包括：

1. Stage-0 原始 PLUTO 预训练（已有能力）
2. Stage-1 IL + 不确定性 warm-up
3. Stage-2 World Model + Plan-R1 风格奖励 + 场景级连续 VD-GRPO
4. 单卡/多卡训练建议

---

## 1. 新增模块总览

### 1.1 策略模型

- `src/models/pluto/probabilistic_pluto_policy.py`
  - 新类 `ProbabilisticPlanningModel`
  - 在 PLUTO decoder query 上新增：
    - `ProbTrajHead`: 连续 latent `z` 注入轨迹
    - `log_std_z`: 可学习 latent 方差
    - `TrajUncertaintyHead`: Stage-1 的 `(x, y)` 不确定性
  - 提供以下关键接口：
    - `sample(data, deterministic=False)`
    - `logprob(data, mode_index, eps)`
    - `make_reference_policy()`

- `src/models/pluto/modules/probabilistic_heads.py`
  - `ProbTrajHead`
  - `TrajUncertaintyHead`
  - `gaussian_log_prob_from_eps`

### 1.2 Stage-1 训练器

- `src/models/pluto_rl/stage1_warmup_trainer.py`
  - `ProbabilisticWarmupTrainer`
  - 在原 PLUTO loss 基础上加入 NLL：
    - 最邻近 mode 匹配 GT
    - 计算 `(x, y)` Gaussian NLL
    - `loss_total = loss_pluto + lambda_nll * loss_nll`

### 1.3 World Model + Reward + VD-GRPO

- `src/models/pluto_rl/world_model.py`
  - `PlutoWorldModel`: 冻结 policy 的 agent prediction 分支
  - `apply_idm_patch`: 前车 IDM 速度修正

- `src/models/pluto_rl/reward.py`
  - `compute_plan_r1_reward`: 安全门控 + 稠密 shaping

- `src/models/pluto_rl/vd_grpo.py`
  - `compute_scene_advantages`: `A = c * (R - mean_scene)`
  - `compute_vd_grpo_loss`: ratio (可选 clip)
  - `sampled_kl`

- `src/models/pluto_rl/stage2_rl_trainer.py`
  - `ContinuousVDGRPOTrainer`
  - 组合 reward、GRPO loss、KL regularization 与弱 IL regularization

### 1.4 配置与脚本

- `config/model/probabilistic_pluto_model.yaml`
- `config/custom_trainer/prob_stage1_warmup_trainer.yaml`
- `config/training/train_prob_pluto_stage1.yaml`
- `run_stage2_rl.py`（支持 `torchrun` 多卡启动，导出 `.pt` + Lightning兼容 `.ckpt`）

---

## 2. Stage-0：原始 PLUTO 预训练（基线）

如果你已有官方 PLUTO checkpoint，可直接跳到 Stage-1。

示例（单机多卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001
```

输出：`checkpoints/*.ckpt`

---

## 3. Stage-1：IL + Uncertainty Warm-up

### 3.1 训练目标

- 保留 imitation 主能力（继承 PLUTO）
- 学习 trajectory 不确定性 `sigma_tau(x, y)`
- latent policy 头完成参数初始化（推理时先可 deterministic）

### 3.2 启动命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_prob_pluto_stage1 \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=24 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  custom_trainer.lambda_nll=0.1 \
  model.init_from_pluto_ckpt=/path/to/pluto_stage0.ckpt
```

> `model.init_from_pluto_ckpt` 是 Stage-1 的**预训练权重入口参数**：
> - 传入 Stage-0/原版 PLUTO 的 checkpoint 时，会自动按 `strict=False` 加载可匹配 backbone 权重；
> - 新增 probabilistic head/uncertainty head 保持随机初始化；
> - 若留空（`null`）则完全从头训练 probabilistic 版本。

建议：

- `lambda_nll` 前 30% epoch 从 0.1 线性升到 1.0
- 首轮 warm-up 可固定 `log_std_z`（冻结）

以上两项**已融入代码**（`ProbabilisticWarmupTrainer`）：

- `lambda_nll` 会按 `lambda_nll_ramp_ratio` 在前段 epoch 线性从 `lambda_nll` 增长到 `lambda_nll_end`；
- `log_std_z` 会在前 `freeze_log_std_z_epochs` 轮自动冻结，之后自动解冻。

对应配置项（`config/custom_trainer/prob_stage1_warmup_trainer.yaml`）：

- `lambda_nll: 0.1`
- `lambda_nll_end: 1.0`
- `lambda_nll_ramp_ratio: 0.3`
- `freeze_log_std_z_epochs: 1`

### 3.3 从旧权重初始化（推荐方式）

优先使用 Hydra 参数入口：`model.init_from_pluto_ckpt=/path/to/pluto_stage0.ckpt`。

这样可以直接复用 `run_training.py` 的训练管线，不需要手改脚本。`from_deterministic_checkpoint(...)` 仍可在离线脚本里使用，但训练入口推荐统一走配置参数。

---

## 4. Stage-2：Continuous VD-GRPO 后训练

### 4.1 核心流程

1. 构造 `policy_ref`（Stage-1 冻结）
2. 构造 `policy_old`（每轮迭代同步）
3. 使用 `PlutoWorldModel` 预测其他车轨迹
4. 对每个场景采样 group（同场景多个 `(mode, z)`）
5. `compute_plan_r1_reward`
6. `A = c * (R - mean_scene)`（不做方差归一化）
7. 更新：`loss_pg + beta_kl * kl + lambda_il * loss_il`

### 4.2 单卡示例

```bash
python run_stage2_rl.py \
  --stage1_ckpt /path/to/stage1.ckpt \
  --epochs 5 \
  --group_samples 4 \
  --lr 1e-5 \
  --export_pt_path /path/to/stage2_vd_grpo_policy.pt \
  --export_ckpt_path /path/to/stage2_vd_grpo_policy.ckpt
```

### 4.3 多卡示例（推荐）

```bash
torchrun --nproc_per_node=4 run_stage2_rl.py \
  --stage1_ckpt /path/to/stage1.ckpt \
  --epochs 8 \
  --group_samples 4 \
  --lr 1e-5 \
  --export_pt_path /path/to/stage2_vd_grpo_policy.pt \
  --export_ckpt_path /path/to/stage2_vd_grpo_policy.ckpt
```

说明：

- `run_stage2_rl.py` 自动读取 `RANK/WORLD_SIZE/LOCAL_RANK` 并初始化 DDP
- 训练完成后主卡同时保存：
  - `stage2_vd_grpo_policy.pt`（纯 `state_dict`）
  - `stage2_vd_grpo_policy.ckpt`（Lightning 兼容格式，可直接走 `run_training.py ... checkpoint=...`）

---

## 5. 多卡训练要点（Stage-1 + Stage-2）

1. **全局 batch 对齐**
   - 经验上先固定 `global_batch = per_gpu_batch * gpu_num`
   - 改 GPU 数时用线性缩放 LR（仅作初始点）

2. **梯度稳定**
   - Stage-2 默认建议：`clip_grad_norm=5.0`
   - 若 reward 波动大，降低 `c_scale` 到 `0.05`

3. **KL 守门**
   - `beta_kl` 初始建议 `0.02`
   - 若策略偏离 IL 过快，提升到 `0.05 ~ 0.1`

4. **world model 偏差**
   - 先开启 IDM patch
   - 若仍过保守，提高 progress/speed reward 权重

5. **日志建议**
   - 监控：`mean_return`, `collision_indicator`, `road_indicator`, `kl`, `ratio`

---

## 6. 推荐超参（首版）

- `latent_dim`: 8~16（默认 12）
- `group_samples`: 4（稳定后试 8）
- `c_scale`: 0.1
- `beta_kl`: 0.02
- `lambda_il`: 0.05
- `log_std_z` 初值：`log(0.1)`

---

## 7. 推理与效果测试（Inference / Eval）

下面给出一个从 checkpoint 做推理验证的最小操作流程。

### 7.1 仿真回放（可执行的 Stage-2 闭环方案）

现在仓库里已经提供一条**可直接执行**的 Stage-2 仿真路径：

- 新增 planner 配置：`config/planner/prob_pluto_planner.yaml`
- 新增运行脚本：`script/run_prob_pluto_planner.sh`

该方案沿用现有 `PlutoPlanner` 后处理链路，但模型替换为 `ProbabilisticPlanningModel`，并在模型 `eval` 时输出仿真兼容字段（`candidate_trajectories` / `probability` / `output_prediction`）。

#### 与原 PLUTO 命令的区别

- 原命令：`script/run_pluto_planner.sh` + `planner=pluto_planner`
- Stage-2 命令：`script/run_prob_pluto_planner.sh` + `planner=prob_pluto_planner`

#### Stage-2 闭环仿真步骤

1. 先完成 Stage-2 训练，得到 `stage2_vd_grpo_policy.ckpt`（推荐）
2. 将文件放入 `checkpoints/` 目录（或自行改脚本路径）
3. 执行：

```bash
sh ./script/run_prob_pluto_planner.sh \
  nuplan_mini \
  mini_demo_scenario \
  stage2_vd_grpo_policy.ckpt \
  /tmp/prob_pluto_sim
```

如果你需要反应式挑战，可把脚本中的 `CHALLENGE` 改为 `closed_loop_reactive_agents`。

#### 原 PLUTO checkpoint 的兼容性

- 原 PLUTO checkpoint 仍建议继续用 `script/run_pluto_planner.sh`。
- Stage-1/Stage-2 probabilistic checkpoint 建议统一用 `script/run_prob_pluto_planner.sh`。

检查点（与原流程相同）：

- 是否能稳定输出轨迹（无 NaN / 无明显抖动）
- 是否出现明显越界、追尾
- 是否在路口/并线场景保持可行速度

### 7.2 批量离线验证（建议）

可用 `run_training.py` 的 validate 流程跑同分布评估：

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=validate +training=train_prob_pluto_stage1 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  checkpoint=/path/to/stage1.ckpt
```

Stage-2 训练脚本现在会直接导出 `stage2_vd_grpo_policy.ckpt`，可直接复用 validate 管线（这是当前推荐的统一测试入口）：

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=validate +training=train_prob_pluto_stage1 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  checkpoint=/path/to/stage2_vd_grpo_policy.ckpt
```

如果你只保留了 `.pt` 文件，则需要先封装成 Lightning `.ckpt`（或写独立评测脚本直接加载 state_dict 做 rollout）。

### 7.3 采样一致性自检（策略层）

对同一场景多次调用 `policy.sample(...)`，检查：

- `mode_index` 分布是否合理（非单一 mode 坍塌）
- `logp` 是否有限值（`isfinite`）
- `trajectory` 速度/加速度统计是否在可接受区间

这一步可以尽早发现 `log_std_z` 过大、策略发散或奖励黑客化问题。

## 8. 集成注意事项

当前 `run_stage2_rl.py` 中 dataloader 使用 placeholder（空数据），你需要接入现有 nuPlan cache datamodule（复用 `src/custom_training` 管线）后即可完整执行。

建议改造点：

- 封装 `RLDataModule` 输出：
  - `features`（与 PLUTO 一致）
  - `expert_traj`（用于弱 IL 正则）
  - 可选：`speed_limit`, `road_signed_distance`
- 在 `ContinuousVDGRPOTrainer.collect_group_rollout` 中替换示例速度/道路占位量

完成上述接入后，即可实现端到端的「Stage-1 预训练 + Stage-2 后训练」。
