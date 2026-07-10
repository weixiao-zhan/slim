#!/usr/bin/env bash
# Algorithm sweep: GRPO / GSPO / PPO across the same mixed (math+vision) dataset on a
# single 8-GPU node with Qwen3.5-2B. PPO additionally trains a critic (expect_critic=1).
#
# Usage:  bash tests/sweep_algo.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"

maybe_detach "$0" "$@"

# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. Rollout colocated on all GPUs.
COMMON_ARGS="
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len 8192
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --old-logprob-source rollout
    --optimizer adam
    --lr 1e-6
    --hf-checkpoint $MODEL_DIR
"

# Single-line algorithm specs (split on newlines by read).
GRPO_ARGS="--advantage-estimator grpo --disable-rewards-std-normalization --actor-num-gpus 8"
GSPO_ARGS="--advantage-estimator gspo --disable-rewards-std-normalization --eps-clip 3e-4 --eps-clip-high 4e-4 --actor-num-gpus 8"
PPO_ARGS="--advantage-estimator ppo_gae --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28 --lr-critic 5e-5 --actor-num-gpus 8 --critic-num-gpus 8 --critic-colocate"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[grpo]="1|0|$GRPO_ARGS"
COMBOS[gspo]="1|0|$GSPO_ARGS"
COMBOS[ppo]="1|1|$PPO_ARGS"

ORDER=(grpo gspo ppo)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

# Submit all combos to one cluster; they queue and run sequentially.
# See tests/RUNNING_TESTS.md for watching/grading via ray job logs.
start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args"
done
