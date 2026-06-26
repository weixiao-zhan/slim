#!/usr/bin/env bash
# Single-GPU PPO+critic e2e smoke for the FSDP actor refactor + per-component
# critic LR. Mirrors sweep_placement.sh's COMMON_ARGS/PPO_ARGS but colocates
# actor+critic+rollout on one GPU (full_colocate) so it runs on a 1-GPU box.
# Exercises the new --lr-*-start-step / --critic-value-head-lr args and grades
# with sanity_check.py (finite actor + critic losses, >=2 weight updates).
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
LOG_DIR="$REPO_DIR/outputs/verify_critic_lr"
mkdir -p "$LOG_DIR"

COMMON_ARGS="
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len 4096
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.6
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64

    --attn-implementation sdpa
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096

    --optimizer adam
    --lr 1e-6
    --hf-checkpoint $MODEL_DIR
"

# PPO + critic, with the new per-component LR start-steps:
#   value head warms from rollout 0, backbone from rollout 1, actor from rollout 1.
PPO_ARGS="--advantage-estimator ppo_gae --gamma 1.0 --lambd 0.95 --value-clip 0.2 \
    --eps-clip 0.2 --eps-clip-high 0.28 \
    --critic-lr 5e-5 --critic-value-head-lr 1e-4 \
    --lr-critic-value-head-start-step 0 --lr-critic-start-step 1 --lr-actor-start-step 1 \
    --actor-num-gpus 1 --critic-num-gpus 1 --critic-colocate --rollout-colocate"

log="$LOG_DIR/full_colocate.log"

start_ray
set +e
run_train "$COMMON_ARGS $PPO_ARGS --save $LOG_DIR/full_colocate.ckpt" "$REPO_DIR/train.py" 2>&1 | tee "$log"
set -e

verdict_line="$(uv run python "$REPO_DIR/tests/sanity_check.py" "$log" 1 1)"
cleanup
echo
echo "================  VERIFY CRITIC-LR (1 GPU)  ================"
echo "  full_colocate  $verdict_line"
echo "==========================================================="
[[ "$verdict_line" == PASS* ]] && { echo "PASSED"; exit 0; } || { echo "FAILED"; exit 1; }
