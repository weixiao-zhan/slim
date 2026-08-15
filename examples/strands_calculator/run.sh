#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# GRPO qualification for interleaved calculator use on an 8-GPU node: training
# takes all 8 and the rollout engines time-share them, one replica per GPU. Every
# SGLang chat completion is retained as a separate trajectory within the episode.

# Strands is an optional extra. Sync before common.sh so that its
# patch_dependencies.py run re-applies the SGLang patches afterwards.
uv sync --extra dev --extra strands
source "$(dirname "$0")/../../tests/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
DATASET_DIR="$REPO_DIR/datasets/strands_calculator"

ARGS="
    --num-rollout 10
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --num-steps-per-rollout 1
    --max-context-len $(K 64)
    --rollout-temperature 1
    --rollout-shuffle
    $(group_advantage_filter_args 32)

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --custom-generate-function-path examples.strands_calculator.generate.generate
    --eval-interval 10
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet
    --eval-n-samples-per-prompt 1
    --eval-save-rollout $REPO_DIR/outputs/strands_calculator/eval/{rollout_id}/{dataset_key}.pt

    --rollout-colocate
    --rollout-num-gpus-per-replica 1
    --rollout-concurrency-per-replica 32
    --sglang-context-length $(K 64)
    --sglang-tool-call-parser qwen3_coder
    --sglang-mem-fraction-static 0.7
    --mamba-radix-cache-strategy extra_buffer
    --sglang-page-size 64
    --sglang-attention-backend triton

    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 20)

    --advantage-estimator grpo
    --disable-group-advantage-std-normalization
    --loss-normalization-unit episode
    --old-logprob-source rollout
    --lr 1e-6
    --actor-num-gpus 8
    --context-parallel-size 4
    --hf-checkpoint $MODEL_DIR
"

start_ray
run_train_wait "$ARGS"
