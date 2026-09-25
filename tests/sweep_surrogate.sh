#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare GRPO policy surrogate objectives.
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
    $(group_advantage_filter_args 32)

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-radix-cache-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus 8
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)

    --advantage-estimator grpo
    --disable-group-advantage-std-normalization
    --old-logprob-source rollout

    --lr 1e-6

    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[ppo_clip]="1|0|--policy-surrogate ppo_clip --eps-clip 0.2 --eps-clip-high 0.28"
COMBOS[is]="1|0|--policy-surrogate is"
COMBOS[tis]="1|0|--policy-surrogate tis --eps-clip 1 --eps-clip-high 1"
COMBOS[cis]="1|0|--policy-surrogate cis --eps-clip 1 --eps-clip-high 1"

ORDER=(ppo_clip is tis cis)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args"
done
