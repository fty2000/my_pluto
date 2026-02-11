# Probabilistic-PLUTO + Continuous VD-GRPO 操作手册

本文档说明如何在当前仓库中完成：

1. Stage-0/Stage-1 监督预训练（IL + uncertainty warm-up）
2. Stage-2 后训练（Continuous VD-GRPO）
3. 训练后测试与评估

> 说明：本仓库新增 `run_rl_training.py`，可直接进行 Stage-2 RL 训练。

---

## 1. 环境准备

```bash
conda create -n pluto python=3.9
conda activate pluto

# 安装 nuPlan devkit
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
pip install -r requirements.txt

# 回到当前仓库
cd /workspace/my_pluto
sh ./script/setup_env.sh

# 建议
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## 2. 数据缓存（Cache）

### 2.1 Sanity-check 缓存

```bash
python run_training.py \
  py_func=cache +training=train_pluto \
  scenario_builder=nuplan_mini \
  cache.cache_path=/nuplan/exp/sanity_check \
  cache.cleanup_cache=true \
  scenario_filter=training_scenarios_tiny \
  worker=sequential
```

### 2.2 全量缓存

```bash
python run_training.py \
  py_func=cache +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.cleanup_cache=true \
  scenario_filter=training_scenarios_1M \
  worker.threads_per_node=40
```

---

## 3. Stage-0：PLUTO 基线预训练

如果你已有官方 checkpoint，可跳过此阶段。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=1e-4 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_stage0
```

可选启用 CIL：

```bash
... model.use_hidden_proj=true +custom_trainer.use_contrast_loss=true
```

---

## 4. Stage-1：Probabilistic-PLUTO warm-up（IL + NLL）

新增可用参数：

- `model.probabilistic_policy=true`
- `model.use_uncertainty_head=true`
- `model.latent_dim=8`
- `+custom_trainer.use_prob_nll_loss=true`
- `+custom_trainer.nll_weight=0.1`

推荐用 Stage-0 ckpt 初始化：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  checkpoint=/path/to/stage0.ckpt \
  model.probabilistic_policy=true \
  model.use_uncertainty_head=true \
  model.latent_dim=8 \
  +custom_trainer.use_prob_nll_loss=true \
  +custom_trainer.nll_weight=0.1 \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=5e-4 epochs=10 warmup_epochs=1 weight_decay=1e-4 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto_stage1_prob
```

NLL 权重建议：

- 前期 `nll_weight=0.1`
- 中期提升至 `0.5`
- 后期可尝试 `1.0`

---

## 5. Stage-2：Continuous VD-GRPO 后训练（run_rl_training.py）

### 5.1 训练入口

```bash
python run_rl_training.py \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  checkpoint=/path/to/stage1.ckpt
```

这会自动：

- 强制 `model.probabilistic_policy=true`
- 构建 datamodule
- 复制参考策略 `ref_model`（可由 `rl.ref_ckpt` 指定）
- 冻结 world model（使用 PLUTO prediction head）
- 进行 scene-group 采样并优化 VD-GRPO loss

### 5.2 常用 RL 参数

可直接在命令行覆盖：

```bash
python run_rl_training.py ... \
  rl.epochs=5 \
  rl.steps_per_epoch=500 \
  rl.val_steps=100 \
  rl.group_size=4 \
  rl.optimizer_lr=1e-4 \
  rl.optimizer_weight_decay=1e-4 \
  rl.advantage_scale=0.1 \
  rl.beta_kl=0.02 \
  rl.lambda_il=0.05 \
  rl.old_policy_sync_interval=50 \
  rl.speed_limit=13.9 \
  rl.grad_clip_norm=5.0 \
  rl.save_dir=rl_checkpoints
```

IDM 相关：

```bash
rl.idm_enabled=true \
rl.idm_a_max=1.5 \
rl.idm_v0=15.0 \
rl.idm_delta=4.0 \
rl.idm_min_gap=2.0 \
rl.idm_headway=1.5
```

### 5.3 多卡/设备

当前 `run_rl_training.py` 已支持 DDP 多卡。

#### 单卡

```bash
python run_rl_training.py ... rl.device=cuda
```

#### 多卡（推荐 torchrun）

```bash
torchrun --nproc_per_node=4 run_rl_training.py \
  +training=train_pluto \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/cache_pluto_1M \
  cache.use_cache_without_dataset=true \
  checkpoint=/path/to/stage1.ckpt \
  rl.device=cuda rl.epochs=5 rl.group_size=4
```

可选 DDP 参数：

```bash
rl.ddp_find_unused_parameters=false
```

说明：
- 脚本会自动根据 `WORLD_SIZE/LOCAL_RANK` 初始化分布式并使用 `DistributedSampler`。
- checkpoint 仅在 rank0 保存。
- 训练/验证指标会做多卡均值聚合。

---

## 6. 断点恢复与参考策略

### 6.1 从 RL checkpoint 恢复

```bash
python run_rl_training.py ... rl.resume_ckpt=/path/to/epoch_005.pt
```

### 6.2 单独指定 reference policy

```bash
python run_rl_training.py ... rl.ref_ckpt=/path/to/stage1_or_ref.ckpt
```

---

## 7. 训练输出

`run_rl_training.py` 在 `${output_dir}/${rl.save_dir}` 下保存：

- `epoch_XXX.pt`：每轮模型
- `best.pt`：按 `val_reward` 最优

pt 内容包括：

- `state_dict`
- `optimizer`
- `cfg`
- `metrics`（train_loss / train_reward / val_reward）

---

## 8. 测试与评估

### 8.1 训练期在线指标（RL 脚本）

重点看：

- `loss`
- `reward`
- `loss_pg`
- `kl`
- `val_reward`

### 8.2 闭环仿真评测

```bash
sh ./script/run_pluto_planner.sh \
  pluto_planner \
  nuplan_mini \
  mini_demo_scenario \
  /path/to/ckpt \
  /path/to/save_video
```

建议对比：

1. Stage-0 baseline
2. Stage-1 probabilistic
3. Stage-2 RL best

重点观察：

- collision
- offroad
- progress
- comfort

---

## 9. 推荐完整流程

1. 先做 `cache`（mini + full）
2. 训练/下载 Stage-0
3. 训练 Stage-1（打开 probabilistic + uncertainty + NLL）
4. 运行 Stage-2（`run_rl_training.py`）
5. 取 `best.pt` 做闭环仿真对比

---

## 10. 常见问题排查

### Q1: RL loss 震荡很大

- 降低 `rl.advantage_scale`（0.1 -> 0.05）
- 提高 `rl.old_policy_sync_interval` 稳定 ratio
- 增大 `rl.beta_kl`

### Q2: 策略太保守

- 增加进度权重（需在 `src/rl/reward.py` 的 weights 调参）
- 适度提高速度相关 reward 权重
- 检查 IDM 参数是否过强

### Q3: 训练太慢

- 降低 `rl.group_size`
- 降低 `rl.steps_per_epoch`
- 使用更小场景过滤器先调通



## 11. Plan-R1 Reward 对齐说明（当前实现）

当前 `src/rl/reward.py` 已按 Plan-R1 的门控形式实现：

- `G_t = I_coll(t) * I_road(t) * I_rule(t)`
- `S_t = w_prog*c_progress + w_speed*c_speed + w_comfort*c_comfort + w_margin*c_margin + w_coll*c_collision`
- `R_t = G_t * S_t + (1 - G_t) * unsafe_penalty`

其中默认 `unsafe_penalty=0`，即安全门控失效时本步不再优化软目标。

可通过 `run_rl_training.py` 的 Hydra 参数直接配置：

```bash
rl.reward_w_progress=1.0 rl.reward_w_speed=0.2 rl.reward_w_comfort=0.05 rl.reward_w_margin=0.3 rl.reward_w_collision=1.0 rl.reward_alpha_road=0.5 rl.reward_beta_collision=1.0 rl.reward_unsafe_penalty=0.0
```
