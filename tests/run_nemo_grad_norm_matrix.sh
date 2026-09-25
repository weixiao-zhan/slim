#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.6-35B-A3B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
RESULT_DIR="${NEMO_GRAD_NORM_RESULT_DIR:-/tmp/slim-nemo-grad-norm-matrix}"
ROLLOUT_DATA="${NEMO_GRAD_NORM_ROLLOUT_DATA:-$RESULT_DIR/rollout-{rollout_id}.pt}"
ROLLOUT_CAPTURE="${NEMO_GRAD_NORM_ROLLOUT_CAPTURE:-$RESULT_DIR/rollout-capture-{rollout_id}.pt}"
PROMPT_DATA="$RESULT_DIR/prompts.parquet"
MAX_TOKENS_PER_GPU="${NEMO_GRAD_NORM_MAX_TOKENS_PER_GPU:-$(K 2)}"
MAX_RESPONSE_TOKENS="${NEMO_GRAD_NORM_MAX_RESPONSE_TOKENS:-256}"
ONLY_CASE="${NEMO_GRAD_NORM_CASE:-}"
NUM_ROLLOUTS=4

mkdir -p "$RESULT_DIR"

stop_ray() {
    uv run ray stop --force >/dev/null 2>&1 || true
    cleanup
}
trap stop_ray EXIT

uv run python tests/prepare_nemo_grad_norm_data.py \
    --math-data "$DATASET_DIR/test_math.parquet" \
    --vision-data "$DATASET_DIR/test_vision.parquet" \
    --output "$PROMPT_DATA" \
    2>&1 | tee "$RESULT_DIR/prepare.log"

FIRST_ROLLOUT_CAPTURE="${ROLLOUT_CAPTURE//\{rollout_id\}/0}"
if [[ ! -f "$FIRST_ROLLOUT_CAPTURE" || "${NEMO_GRAD_NORM_REGENERATE_ROLLOUT:-0}" == "1" ]]; then
    start_ray
    run_train_wait "
        --debug-rollout-only
        --num-rollout $NUM_ROLLOUTS
        --rollout-batch-size 8
        --n-samples-per-prompt 4
        --num-steps-per-rollout 1
        --max-context-len $(K 16)
        --rollout-temperature 1

        --prompt-data $PROMPT_DATA
        --rm-type math
        --skip-eval-before-train

        --rollout-num-gpus 8
        --rollout-num-gpus-per-replica 2
        --sglang-mem-fraction-static 0.7
        --sglang-attention-backend fa3
        --sglang-mamba-radix-cache-strategy extra_buffer
        --sglang-page-size 64

        --hf-checkpoint $MODEL_DIR
        --save-debug-rollout-data $ROLLOUT_CAPTURE
    " 2>&1 | tee "$RESULT_DIR/rollout.log"
fi

: > "$RESULT_DIR/trim.log"
for ((rollout_id = 0; rollout_id < NUM_ROLLOUTS; rollout_id++)); do
    capture_path="${ROLLOUT_CAPTURE//\{rollout_id\}/$rollout_id}"
    rollout_path="${ROLLOUT_DATA//\{rollout_id\}/$rollout_id}"
    uv run python tests/trim_nemo_grad_norm_rollout.py \
        --input "$capture_path" \
        --output "$rollout_path" \
        --max-response-tokens "$MAX_RESPONSE_TOKENS" \
        2>&1 | tee -a "$RESULT_DIR/trim.log"
done

for case_config in \
    "cp1_ep1 1 1" \
    "cp1_ep8 1 8" \
    "cp2_ep1 2 1" \
    "cp2_ep8 2 8"
do
    read -r case_name cp_size ep_size <<<"$case_config"
    if [[ -n "$ONLY_CASE" && "$case_name" != "$ONLY_CASE" ]]; then
        continue
    fi
    start_ray
    run_train_wait "
        --num-rollout $NUM_ROLLOUTS
        --rollout-batch-size 8
        --n-samples-per-prompt 4
        --num-steps-per-rollout 1
        --max-context-len $(K 16)
        --rollout-temperature 1

        --prompt-data $PROMPT_DATA
        --rm-type math
        --skip-eval-before-train
        --load-debug-rollout-data $ROLLOUT_DATA

        --actor-num-gpus 8
        --context-parallel-size $cp_size
        --expert-model-parallel-size $ep_size
        --activation-checkpointing
        --use-dynamic-batch-size
        --max-tokens-per-gpu $MAX_TOKENS_PER_GPU

        --advantage-estimator grpo
        --policy-surrogate ppo_clip
        --disable-group-advantage-std-normalization
        --old-logprob-source actor
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
    " 2>&1 | tee "$RESULT_DIR/$case_name.log"
done
