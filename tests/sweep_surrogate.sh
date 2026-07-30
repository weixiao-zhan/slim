#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Surrogate sweep: GRPO with each policy-gradient surrogate objective — ppo_clip, is, tis,
# cis — on the mixed (math+vision) dataset, single 8-GPU node, Qwen3.5-2B.
#
# Usage:  bash tests/sweep_surrogate.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
maybe_detach "$0" "$@"

# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. GRPO, rollout colocated.
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
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus 8
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator grpo
    --disable-group-advantage-std-normalization
    --old-logprob-source rollout
    --lr 1e-6
    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args".  ppo_clip uses the standard asymmetric
# clip; is is unclipped (no eps); tis/cis use a wide symmetric clip like test_grpo_mixed_cis.
declare -A COMBOS
COMBOS[ppo_clip]="1|0|--policy-surrogate ppo_clip --eps-clip 0.2 --eps-clip-high 0.28"
COMBOS[is]="1|0|--policy-surrogate is"
COMBOS[tis]="1|0|--policy-surrogate tis --eps-clip 1 --eps-clip-high 1"
COMBOS[cis]="1|0|--policy-surrogate cis --eps-clip 1 --eps-clip-high 1"

ORDER=(ppo_clip is tis cis)
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
