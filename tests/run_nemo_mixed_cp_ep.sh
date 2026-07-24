#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3.5 MoE GRPO on mixed DAPO math and Geometry3K data.
# SGLang uses four TP2 replicas. NeMo uses CP2 and EP8 on eight GPUs.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-35B-A3B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
RESULT_DIR="${NEMO_MIXED_CP_EP_RESULT_DIR:-/tmp/slim-nemo-mixed-cp2-ep8-five-step}"

mkdir -p "$RESULT_DIR"

stop_ray() {
    uv run ray stop --force >/dev/null 2>&1 || true
    cleanup
}
trap stop_ray EXIT

start_ray
run_train_wait "
    --num-rollout 5
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --num-steps-per-rollout 1
    --max-context-len 16384
    --rollout-temperature 1
    --rollout-group-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 64

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --eval-log-passrate
    --eval-interval 5
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet
    --eval-n-samples-per-prompt 1

    --rollout-num-gpus-per-replica 2
    --sglang-mem-fraction-static 0.7
    --sglang-attention-backend fa3
    --mamba-radix-cache-strategy extra_buffer
    --sglang-page-size 64
    --sglang-enforce-disable-flashinfer-allreduce-fusion
    --rollout-colocate

    --actor-num-gpus 8
    --context-parallel-size 2
    --expert-model-parallel-size 8
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator grpo
    --policy-surrogate ppo_clip
    --disable-rewards-std-normalization
    --old-logprob-source rollout
    --use-rollout-routing-replay
    --eps-clip 0.2
    --eps-clip-high 0.28

    --optimizer adam
    --lr 3e-6
    --lr-warmup-iters 0
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0

    --hf-checkpoint $MODEL_DIR
" 2>&1 | tee "$RESULT_DIR/run.log"
