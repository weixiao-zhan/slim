#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen4 GRPO on DAPO math across two 8-GPU nodes.
# SGLang uses one TP8 and EP8 replica per node. NeMo uses CP2 and EP16 on sixteen GPUs, so Muon updates the expert weights.
# The engines serve the language model that NeMo trains and keep the frozen PLE n-gram table in pinned host memory,
# which the colocated weights release keeps, so the weight sync skips the table.
# Run it on the head node with MASTER_ADDR set to the head IP that the worker reaches
# and WORKER_HOST set to the ssh host of the worker. Both nodes hold the repository at
# the same path and the checkpoint at MODEL_DIR.
: "${MASTER_ADDR:?set MASTER_ADDR to the head node IP that the worker node reaches}"
: "${WORKER_HOST:?set WORKER_HOST to the ssh host of the worker node}"
source "$(dirname "$0")/common.sh"

MODEL_DIR="${QWEN4_MODEL_DIR:-$REPO_DIR/models/Qwen3.8-Flash-Next}"
DATASET_DIR="$REPO_DIR/datasets/mixed"
RESULT_DIR="${QWEN4_RESULT_DIR:-/tmp/slim-nemo-qwen4-cp2-ep16}"

mkdir -p "$RESULT_DIR"

on_worker() {
    ssh "$WORKER_HOST" "export PATH=\$HOME/.local/bin:\$PATH RAY_AUTH_MODE=$RAY_AUTH_MODE && cd '$REPO_DIR' && $1"
}

stop_ray() {
    on_worker "uv run ray stop --force; pkill -9 sglang; pkill -9 slim" >/dev/null 2>&1 || true
    uv run ray stop --force >/dev/null 2>&1 || true
    cleanup
}
trap stop_ray EXIT

test -f "$MODEL_DIR/config.json"
on_worker "test -f '$MODEL_DIR/config.json' && uv run python patch_dependencies.py"
start_ray
on_worker "pkill -9 sglang; pkill -9 slim; uv run ray stop --force" || true
on_worker "uv run ray start --address '$MASTER_ADDR:6379' --num-gpus 8 --disable-usage-stats"

run_train_wait "
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len $(K 16)
    --rollout-temperature 1
    --rollout-shuffle
    $(group_advantage_filter_args 16)

    --prompt-data $DATASET_DIR/train_math.parquet
    --rm-type math
    --skip-eval-before-train

    --rollout-num-gpus-per-replica 8
    --sglang-ep-size 8
    --sglang-mem-fraction-static 0.7
    --sglang-attention-backend fa3
    --sglang-mamba-ssm-dtype float32
    --sglang-mamba-radix-cache-strategy extra_buffer_lazy
    --sglang-max-mamba-cache-size 512
    --sglang-page-size 64
    --sglang-json-model-override-args {\"language_model_only\":true}
    --sglang-ple-offload-embedding
    --rollout-colocate

    --actor-num-gpus 16
    --context-parallel-size 2
    --expert-model-parallel-size 16
    --activation-checkpointing
    --freeze-vision-tower
    --freeze-ngram
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 8)

    --advantage-estimator grpo
    --policy-surrogate ppo_clip
    --disable-group-advantage-std-normalization
    --use-rollout-routing-replay
    --eps-clip 0.2
    --eps-clip-high 0.28

    --lr 1e-6
    --lr-warmup-steps 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0

    --hf-checkpoint $MODEL_DIR
" 2>&1 | tee "$RESULT_DIR/run.log"
