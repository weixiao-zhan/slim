#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Block-FP8 rollout with CIS, score centering and fp32 master weights, on the
# Strands agent, with colocated rollout.
#
# Ray workers inherit the environment of the shell that starts Ray, so export these first;
# without them the FlashInfer JIT cannot find cuda_runtime.h and every SGLang engine dies:
#   export CUDA_HOME=$HOME/.cache/cuda13_home
#   export FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK

# Strands is an optional extra. Sync before common.sh so that its
# patch_dependencies.py run re-applies the SGLang patches afterwards.
uv sync --extra dev --extra strands
source "$(dirname "$0")/common.sh"

BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
# Weightless block-FP8 rollout checkpoint; the first weight sync fills it:
#   python tools/convert_hf_to_fp8.py --model-dir models/Qwen3.5-4B --save-dir models/Qwen3.5-4B-FP8 \
#       --ref-config tools/quant_recipes/qwen35_128blockfp8.json --config-only
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-FP8"
# Built by examples/strands_calculator/prepare_dataset.py.
DATASET_DIR="$REPO_DIR/datasets/strands_calculator"
SAVE_DIR="$REPO_DIR/outputs/strands-4b-fp8-cis-sc-fp32"
CONCURRENCY=64

TRAIN_ARGS="
    --num-rollout 40
    --rollout-batch-size 8
    --n-samples-per-prompt 16
    --num-steps-per-rollout 1
    --max-context-len $(K 32)
    --rollout-temperature 1
    --rollout-shuffle
    $(group_advantage_filter_args 16)

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --custom-generate-function-path examples.strands_calculator.generate.generate
    --apply-chat-template-kwargs {\"preserve_thinking\":true}
    --eval-interval 20
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet
    --eval-n-samples-per-prompt 1

    --rollout-num-gpus-per-replica 1
    --rollout-concurrency-per-replica $CONCURRENCY
    --rollout-num-workers 2
    --sglang-tool-call-parser qwen3_coder
    --sglang-reasoning-parser qwen3
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-ssm-dtype float32
    --sglang-max-mamba-cache-size $((5 * CONCURRENCY))
    --sglang-load-format dummy
    --rollout-colocate

    --actor-num-gpus $NUM_GPUS
    --context-parallel-size 2
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)
    --master-weight-dtype fp32

    --advantage-estimator grpo
    --disable-group-advantage-std-normalization
    --policy-surrogate cis
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-score-centering 127
    --loss-normalization-unit episode
    --old-logprob-source rollout

    --lr 1e-6
    --lr-warmup-steps 5

    --hf-checkpoint $BF16_MODEL_DIR
    --rollout-load $FP8_MODEL_DIR
    --save $SAVE_DIR
    --save-interval 20
"

start_ray
run_train "$TRAIN_ARGS $(wandb_args "strands-4b-fp8-cis-sc-fp32")"
