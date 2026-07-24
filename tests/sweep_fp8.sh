#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# FP8 sweep: GRPO + CIS on the mixed (math+vision) dataset, comparing the rollout-weight
# precision against a bf16 baseline on a single 8-GPU node with Qwen3.5-2B:
#   bf16       -> bf16 rollout weights (base case)
#   fp8_fp32   -> block-FP8 rollout weights, fp32 block scales
#   fp8_ue8m0  -> block-FP8 rollout weights, ue8m0 (power-of-two) block scales
# Rollout colocates on all 8 GPUs. The fp8_* combos require pre-forged FP8 checkpoints
# (see tests/RUNNING_TESTS.md).
#
# Usage:  bash tests/sweep_fp8.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B-FP8"
FP8_UE8M0_MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B-FP8-ue8m0"
DATASET_DIR="$REPO_DIR/datasets/mixed"
maybe_detach "$0" "$@"

# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. GRPO + CIS, rollout colocated.
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
    --sglang-mem-fraction-static 0.8
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus $NUM_GPUS
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator grpo
    --disable-rewards-std-normalization
    --policy-surrogate cis
    --old-logprob-source rollout
    --eps-clip 1
    --eps-clip-high 1

    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
"

# combo -> "expect_actor|expect_critic|combo_args".  fp8 runs use the forged FP8 weights as
# the rollout/hf checkpoint and load the bf16 master weights for training.
declare -A COMBOS
COMBOS[bf16]="1|0|--hf-checkpoint $BF16_MODEL_DIR"
COMBOS[fp8_fp32]="1|0|--hf-checkpoint $FP8_MODEL_DIR --load $BF16_MODEL_DIR"
COMBOS[fp8_ue8m0]="1|0|--hf-checkpoint $FP8_UE8M0_MODEL_DIR --load $BF16_MODEL_DIR"

ORDER=(bf16 fp8_fp32 fp8_ue8m0)
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
