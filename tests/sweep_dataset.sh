#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Dataset sweep: PPO across data modalities — text math (DAPO-17k) and vision (Geo3K) —
# on a single 8-GPU node with Qwen3.5-2B.
#
# Usage:  bash tests/sweep_dataset.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. PPO on the simplest fully
# separate layout: 2 actor + 2 critic + 4 rollout GPUs, no colocation.
COMMON_ARGS="
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len 8192
    --rollout-temperature 1
    --rollout-shuffle
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64

    --actor-num-gpus 2
    --critic-num-gpus 2
    --rollout-num-gpus 4
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator ppo_gae
    --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28
    --lr 1e-6
    --lr-critic 5e-5
    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args"  (one line each; split on newlines by read)
declare -A COMBOS
COMBOS[math]="1|1|--prompt-data $REPO_DIR/datasets/dapo17k/train.parquet"
COMBOS[vision]="1|1|--prompt-data $REPO_DIR/datasets/geo3k/train.parquet"

ORDER=(math vision)
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
