#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# MoE rollout-parallelism sweep: GSPO on Qwen3.5-MoE (Qwen3.6-35B-A3B) with rollout routing
# replay (R3) always on, sweeping the sglang rollout parallelism layout on one 8-GPU node:
#   tp1      -> tensor-parallel 1   (8 single-GPU replicas)
#   tp4      -> tensor-parallel 4   (2 replicas, pure TP)
#   tp4_ep4  -> tensor-parallel 4 + expert-parallel 4   (2 replicas, EP across experts)
# The MoE checkpoint is referenced via the models/<name> symlink to NVMe.
#
# Usage:  bash tests/sweep_moe_rollout.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.6-35B-A3B"
DATASET_DIR="$REPO_DIR/datasets/geo3k"
maybe_detach "$0" "$@"

# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. 8 actor GPUs, rollout colocated.
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
    --skip-eval-before-train

    --sglang-mem-fraction-static 0.8
    --sglang-attention-backend fa3
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --sglang-enforce-disable-flashinfer-allreduce-fusion
    --rollout-colocate

    --actor-num-gpus 8
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator gspo
    --disable-group-advantage-std-normalization
    --old-logprob-source rollout
    --eps-clip 3e-4
    --eps-clip-high 4e-4
    --use-rollout-routing-replay

    --lr 1e-5
    --lr-warmup-iters 0
    --lr-decay-style constant
    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args".
# TP degree is the per-replica GPU count; EP is set explicitly via --sglang-ep.
declare -A COMBOS
COMBOS[tp1]="1|0|--rollout-num-gpus-per-replica 1"
COMBOS[tp4]="1|0|--rollout-num-gpus-per-replica 4"
COMBOS[tp4_ep4]="1|0|--rollout-num-gpus-per-replica 4 --sglang-ep 4"

ORDER=(tp1 tp4 tp4_ep4)
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
