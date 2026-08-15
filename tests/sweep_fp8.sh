#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare BF16 and block-FP8 rollout weights.
source "$(dirname "$0")/common.sh"

BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-FP8"
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
    --sglang-mem-fraction-static 0.8
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus $NUM_GPUS
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)

    --advantage-estimator grpo
    --disable-group-advantage-std-normalization
    --policy-surrogate cis
    --old-logprob-source rollout
    --eps-clip 1
    --eps-clip-high 1

    --lr 1e-6
    --lr-decay-style constant
"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[bf16]="1|0|--hf-checkpoint $BF16_MODEL_DIR"
COMBOS[fp8_fp32]="1|0|--hf-checkpoint $FP8_MODEL_DIR --load $BF16_MODEL_DIR"

ORDER=(bf16 fp8_fp32)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args"
done
