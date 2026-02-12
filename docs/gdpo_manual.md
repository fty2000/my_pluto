# PLUTO Residual-GDPO 训练与评估操作手册

本文档给出从环境检查、数据缓存、监督预训练、GDPO 微调到仿真评估的完整流程。

## 0. 功能开关与代码入口

- residual policy 集成入口：`src/models/pluto/pluto_model.py`
- GDPO 训练入口：`src/models/pluto/pluto_trainer.py`
- 训练配置：
  - `config/model/pluto_model.yaml`
  - `config/custom_trainer/pluto_trainer.yaml`
  - `config/training/train_pluto_gdpo.yaml`

核心开关：
- `model.enable_residual_policy=true`
- `custom_trainer.use_gdpo=true`

> 注意：GDPO 当前为“离线近似”版本（在训练 batch 上进行 group sampling 与 reward 聚合），并非完整 nuPlan 闭环 rollout 环境。

---

## 1. 前置准备

1. 完成 nuPlan 数据集与 devkit 安装。
2. 完成本仓库依赖安装。
3. 建议先执行缓存与小规模 sanity check。

---

## 2. 数据缓存（推荐先跑 tiny）

```bash
python run_training.py \
  py_func=cache +training=train_pluto \
  scenario_builder=nuplan_mini \
  cache.cache_path=/nuplan/exp/sanity_check \
  cache.cleanup_cache=true \
  scenario_filter=training_scenarios_tiny \
  worker=sequential
```

全量缓存示例：

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

## 3. 监督预训练（获取基础 checkpoint）

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=4 \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1
```

训练完成后，checkpoint 默认保存在当前实验目录下的 `checkpoints/`。

---

## 4. GDPO 微调（Residual head）

### 4.1 推荐 sanity 配置

```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto_gdpo \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/sanity_check cache.use_cache_without_dataset=true \
  worker=single_machine_thread_pool worker.max_workers=4 \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1 \
  checkpoint=/path/to/pluto_supervised.ckpt \
  model.enable_residual_policy=true \
  custom_trainer.use_gdpo=true \
  custom_trainer.gdpo_group_size=4 \
  custom_trainer.gdpo_reward_weights=[2.0,2.0,1.0,1.0,1.0] \
  custom_trainer.gdpo_residual_l2=0.001 \
  custom_trainer.gdpo_freeze_backbone=true
```

### 4.2 常用超参数说明

- `custom_trainer.gdpo_group_size`：每个样本做多少次采样（G）
- `custom_trainer.gdpo_reward_weights`：多奖励加权（顺序：collision/offroad/speed/comfort/progress）
- `custom_trainer.gdpo_residual_l2`：动作 L2 正则
- `custom_trainer.gdpo_freeze_backbone`：是否冻结除 residual head 外参数（建议 true）
- `custom_trainer.gdpo_speed_limit`：速度惩罚阈值（m/s）
- `custom_trainer.gdpo_collision_distance`：碰撞近似阈值（m）

---

## 5. 训练日志解读

重点观察如下指标：

- `objectives/train_gdpo_loss`
- `objectives/train_reward_collision`
- `objectives/train_reward_offroad`
- `objectives/train_reward_speed`
- `objectives/train_reward_comfort`
- `objectives/train_reward_progress`
- `objectives/train_residual_l2`

建议：
1. 若 loss 抖动大：先减小 `lr`。
2. 若 reward 方差过大：减小 `gdpo_group_size` 或提高 batch size。
3. 若轨迹偏移过大：增大 `gdpo_residual_l2`，并收紧 `model.residual_dx_max/dy_max`。

---

## 6. 评估与仿真测试

### 6.1 离线验证（validation）

```bash
python run_training.py \
  py_func=validate +training=train_pluto_gdpo \
  scenario_builder=nuplan \
  cache.cache_path=/nuplan/exp/sanity_check cache.use_cache_without_dataset=true \
  checkpoint=/path/to/your_gdpo_checkpoint.ckpt \
  model.enable_residual_policy=true \
  custom_trainer.use_gdpo=true
```

### 6.2 nuPlan planner 闭环仿真

```bash
sh ./script/run_pluto_planner.sh \
  pluto_planner nuplan_mini mini_demo_scenario \
  /path/to/your_gdpo_checkpoint.ckpt \
  /dir_to_save_the_simulation_result_video
```

查看输出视频，重点检查：
- 是否出现明显越界/碰撞
- 速度是否过激
- 是否比监督基线更平顺/更高效

---

## 7. 常见问题排查

1. **报错：GDPO requires model.enable_residual_policy=true**  
   确保命令行覆盖了 `model.enable_residual_policy=true`。

2. **报错：Residual policy did not return residual_log_prob**  
   确保 `custom_trainer.use_gdpo=true` 时，模型 forward 的采样路径被调用（当前 trainer 已自动注入 `sample_residual=True`）。

3. **训练无改进**  
   优先检查 reward 权重是否过度偏向某一项，建议先提高安全项（collision/offroad）。

