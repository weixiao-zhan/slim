#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Placement sweep: exercise every (rollout_colocate, critic_colocate) combo plus
# an HSDP control on a single 8-GPU node, on the small mixed (math+vision) dataset.
#
# Usage:  bash tests/sweep_placement.sh [combo_name ...]
#   With no args, runs every combo. With args, runs only the named combos.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
maybe_detach "$0" "$@"

# Sizes: 8 prompts x 4 samples = 32 episodes; gbs=32 -> one train step per rollout.
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

    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --lr 1e-6
    --hf-checkpoint $MODEL_DIR
"

# Single-line: these are embedded in COMBOS specs split by `read`, which stops at newlines.
PPO_ARGS="--advantage-estimator ppo_gae --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28 --lr-critic 5e-5"
GRPO_ARGS="--advantage-estimator grpo --disable-rewards-std-normalization"

# combo -> "train_cmd|expect_actor|expect_critic|cluster_args"
declare -A COMBOS
COMBOS[default]="slim-train|1|1|$PPO_ARGS --actor-num-gpus 2 --critic-num-gpus 2 --rollout-num-gpus 4"
COMBOS[default_async]="slim-train-async|1|1|$PPO_ARGS --actor-num-gpus 2 --critic-num-gpus 2 --rollout-num-gpus 4"
COMBOS[colocate_rollout]="slim-train|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --rollout-colocate"
COMBOS[colocate_critic]="slim-train|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-num-gpus 4"
COMBOS[colocate_critic_async]="slim-train-async|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-num-gpus 4"
COMBOS[full_colocate]="slim-train|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-colocate"
# HSDP control: 8 actor GPUs, replicated DP 4 and sharded DP 2, with colocated rollout.
COMBOS[grpo_hsdp]="slim-train|1|0|$GRPO_ARGS --actor-num-gpus 8 --dp-replicate-size 4 --rollout-colocate"

ORDER=(default default_async colocate_rollout colocate_critic colocate_critic_async full_colocate grpo_hsdp)

# Allow running a subset by name.
if [[ $# -gt 0 ]]; then
    ORDER=("$@")
fi

# Submit all combos to one cluster; they queue and run sequentially.
# See tests/RUNNING_TESTS.md for watching/grading via ray job logs.
start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then
        echo "Unknown combo: $name"; exit 2
    fi
    IFS='|' read -r train_cmd expect_actor expect_critic cluster_args <<< "$spec"
    echo ">>> Submitting $name  (cmd=$train_cmd, expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $cluster_args" "$train_cmd"
done
