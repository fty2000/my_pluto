#!/usr/bin/env bash
set -euo pipefail

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

PLANNER=${1:-pluto_planner}
BUILDER=${2:-nuplan_mini}
FILTER=${3:-mini_demo_scenario}
CKPT=${4:-}
VIDEO_SAVE_DIR=${5:-$cwd}
ENABLE_RESIDUAL=${6:-false}

if [ -z "$CKPT" ]; then
  echo "Usage: sh ./script/run_pluto_planner.sh <planner> <builder> <filter> <ckpt_name_or_path> <video_save_dir> [enable_residual=true|false]"
  exit 1
fi

if [[ "$CKPT" = /* ]] || [[ "$CKPT" =~ ^[A-Za-z]:[\\/] ]]; then
  CKPT_PATH="$CKPT"
else
  CKPT_PATH="$CKPT_ROOT/$CKPT"
fi

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=sequential \
    verbose=true \
    experiment_uid="pluto_planner/$FILTER" \
    planner.pluto_planner.render=true \
    planner.pluto_planner.planner_ckpt="$CKPT_PATH" \
    planner.pluto_planner.planner.enable_residual_policy=$ENABLE_RESIDUAL \
    +planner.pluto_planner.save_dir=$VIDEO_SAVE_DIR
