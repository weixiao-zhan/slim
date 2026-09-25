#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3.6 MoE GRPO on mixed DAPO math and Geometry3K data.
# SGLang uses four TP2 replicas. NeMo uses CP2 and EP8 on eight GPUs.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.6-35B-A3B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
RESULT_DIR="${NEMO_MIXED_CP_EP_RESULT_DIR:-/tmp/slim-nemo-mixed-cp2-ep8}"

mkdir -p "$RESULT_DIR"

stop_ray() {
    uv run ray stop --force >/dev/null 2>&1 || true
    cleanup
}
trap stop_ray EXIT

start_ray
run_train_wait "
    --num-rollout 4
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len $(K 16)
    --rollout-temperature 1
    --rollout-shuffle
    $(group_advantage_filter_args 16)

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --skip-eval-before-train

    --rollout-num-gpus-per-replica 2
    --sglang-mem-fraction-static 0.7
    --sglang-attention-backend fa3
    --sglang-mamba-radix-cache-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus 8
    --context-parallel-size 2
    --expert-model-parallel-size 8
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 8)

    --advantage-estimator grpo
    --policy-surrogate ppo_clip
    --disable-group-advantage-std-normalization
    --old-logprob-source rollout
    --use-rollout-routing-replay
    --eps-clip 0.2
    --eps-clip-high 0.28

    --lr 3e-6
    --lr-warmup-iters 0
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0

    --hf-checkpoint $MODEL_DIR
" 2>&1 | tee "$RESULT_DIR/run.log"
