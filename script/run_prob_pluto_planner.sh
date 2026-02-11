cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

BUILDER=$1
FILTER=$2
CKPT=$3
VIDEO_SAVE_DIR=$4

CHALLENGE="closed_loop_nonreactive_agents"
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"

python run_simulation.py \
    +simulation=$CHALLENGE \
    planner=prob_pluto_planner \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=sequential \
    verbose=true \
    experiment_uid="prob_pluto_planner/$FILTER" \
    planner.prob_pluto_planner.render=true \
    planner.prob_pluto_planner.planner_ckpt="$CKPT_ROOT/$CKPT" \
    +planner.prob_pluto_planner.save_dir=$VIDEO_SAVE_DIR
