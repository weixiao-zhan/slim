#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare actor, critic, and rollout placements.
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

    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)

    --advantage-estimator ppo_gae
    --value-clip 0.2
    --eps-clip 0.2
    --eps-clip-high 0.28

    --lr 1e-6
    --lr-critic 5e-5

    --hf-checkpoint $MODEL_DIR
"

# combo -> "train_cmd|expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[default_async]="slim-train-async|1|1|--actor-num-gpus 2 --critic-num-gpus 2 --rollout-num-gpus 4"
COMBOS[colocate_rollout]="slim-train|1|1|--actor-num-gpus 4 --critic-num-gpus 4 --rollout-colocate"
COMBOS[colocate_critic_async]="slim-train-async|1|1|--actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-num-gpus 4"
COMBOS[full_colocate]="slim-train|1|1|--actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-colocate"

ORDER=(default_async colocate_rollout colocate_critic_async full_colocate)

if [[ $# -gt 0 ]]; then
    ORDER=("$@")
fi

start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then
        echo "Unknown combo: $name"; exit 2
    fi
    IFS='|' read -r train_cmd expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (cmd=$train_cmd, expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args" "$train_cmd"
done
