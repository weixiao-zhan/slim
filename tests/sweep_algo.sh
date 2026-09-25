#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare GRPO, GSPO, and PPO.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
DATASET_DIR="$REPO_DIR/datasets/mixed"

COMMON_ARGS="
    --num-rollout 4
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len $(K 16)
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-radix-cache-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)

    --old-logprob-source rollout

    --lr 1e-6

    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[grpo]="1|0|$(group_advantage_filter_args 32) --actor-num-gpus 8 --advantage-estimator grpo --disable-group-advantage-std-normalization"
COMBOS[gspo]="1|0|$(group_advantage_filter_args 32) --actor-num-gpus 8 --advantage-estimator gspo --disable-group-advantage-std-normalization --eps-clip 3e-4 --eps-clip-high 4e-4"
COMBOS[ppo]="1|1|--actor-num-gpus 8 --critic-num-gpus 8 --critic-colocate --advantage-estimator ppo_gae --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28 --lr-critic 5e-5"

ORDER=(grpo gspo ppo)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args"
done
