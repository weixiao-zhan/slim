#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare episode and trajectory packing configurations.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
DATASET_DIR="$REPO_DIR/datasets/mixed"

# Disaggregated placement: 2 GPUs train while 2 serve the next rollout through the
# one-step-off-policy async trainer. Weight sync uses NCCL instead of CUDA IPC.
COMMON_ARGS="
    --num-rollout 4
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --max-context-len $(K 16)
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus 2
    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --sglang-attention-backend triton

    --actor-num-gpus 2
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu $(K 16)

    --old-logprob-source rollout

    --lr 1e-6

    --hf-checkpoint $MODEL_DIR
"

GRPO="$(group_advantage_filter_args 32) --advantage-estimator grpo --disable-group-advantage-std-normalization"

# One optimizer step per rollout: the whole batch lands in a single step.
UNIT_EPISODE_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit episode"
UNIT_TRAJECTORY_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit trajectory"
UNIT_TOKEN_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit token"

# Two optimizer steps: the flattener partitions across steps, then across ranks.
MULTI_STEP_ARGS="$GRPO --num-steps-per-rollout 2 --loss-normalization-unit episode"

# Context parallel halves the logical DP size, so the partition stride changes.
CP2_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit episode --context-parallel-size 2"

# Group std normalization exercises the Bessel-corrected per-group scatter.
GRPO_STD_ARGS="$(group_advantage_filter_args 32) --advantage-estimator grpo --num-steps-per-rollout 1 --loss-normalization-unit episode"

GSPO_ARGS="$(group_advantage_filter_args 32) --advantage-estimator gspo --disable-group-advantage-std-normalization --num-steps-per-rollout 1 --loss-normalization-unit episode --eps-clip 3e-4 --eps-clip-high 4e-4"

# PPO adds the critic value round trip over trajectory-keyed values.
PPO_ARGS="--advantage-estimator ppo_gae --num-steps-per-rollout 1 --loss-normalization-unit episode --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28 --lr-critic 5e-5 --critic-num-gpus 2 --critic-colocate"

# Fixed micro-batching requires exactly equal pack counts per rank, with no
# dynamic re-split available to reconcile a mismatch.
FIXED_MBS_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit episode --micro-batch-size 2"

# Round-robin assignment instead of Karmarkar-Karp balancing.
NO_BALANCE_ARGS="$GRPO --num-steps-per-rollout 1 --loss-normalization-unit episode --no-balance-data"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
COMBOS[unit_episode]="1|0|$UNIT_EPISODE_ARGS"
COMBOS[unit_trajectory]="1|0|$UNIT_TRAJECTORY_ARGS"
COMBOS[unit_token]="1|0|$UNIT_TOKEN_ARGS"
COMBOS[multi_step]="1|0|$MULTI_STEP_ARGS"
COMBOS[cp2]="1|0|$CP2_ARGS"
COMBOS[grpo_std]="1|0|$GRPO_STD_ARGS"
COMBOS[gspo]="1|0|$GSPO_ARGS"
COMBOS[ppo]="1|1|$PPO_ARGS"
COMBOS[fixed_mbs]="1|0|$FIXED_MBS_ARGS"
COMBOS[no_balance]="1|0|$NO_BALANCE_ARGS"

ORDER=(unit_episode unit_trajectory unit_token multi_step cp2 grpo_std gspo ppo fixed_mbs no_balance)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

start_ray

for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    echo ">>> Submitting $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train "$COMMON_ARGS $combo_args" slim-train-async
done
