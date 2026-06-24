#!/usr/bin/env bash
# End-to-end RL wall-time comparison: FP8 vs BF16 rollout on Qwen3.5-4B-Base (H100x8).
# GRPO + CIS, mixed math+vision dataset,
# 100 rollout steps, eval every 10.
# Runs fp8 first, then bf16. Uses the forged FP8 checkpoint for rollout, BF16 master weights
# for the training actor (same recipe as sweep_fp8.sh). On H100, sglang uses DeepGEMM by
# default for FP8 GEMMs (fp32 block scales), so no gemm-backend override is needed.
#
# Usage:
#   bash report/fp8_r3/stage3_e2erl.sh [fp8|bf16]   # single run
#   bash report/fp8_r3/stage3_e2erl.sh               # both (fp8 first)
#
# Prerequisites:
#   models/Qwen3.5-4B-Base       (BF16 master weights + HF checkpoint for bf16 run)
#   models/Qwen3.5-4B-Base-FP8   (FP8 rollout checkpoint for fp8 run)
#   datasets/mixed/{train,test_math,test_vision}.parquet  (from tests/prepare_mixed.py)
source "$(dirname "$0")/../../tests/common.sh"

BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-Base"
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-Base-FP8"
DATASET_DIR="$REPO_DIR/datasets/mixed"
LOG_DIR="$REPO_DIR/outputs/fp8_e2erl"
mkdir -p "$LOG_DIR"

WANDB_ARGS="$(wandb_args fp8_e2erl)"

COMMON_ARGS="
    --num-rollout 50
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --num-steps-per-rollout 1
    --max-context-len 8192
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --rollout-concurrency-per-replica 64
    --sglang-mem-fraction-static 0.8
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --eval-interval 10
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet

    --actor-num-gpus $NUM_GPUS
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768

    --advantage-estimator grpo
    --disable-rewards-std-normalization
    --policy-surrogate cis
    --old-logprob-source rollout
    --eps-clip 1
    --eps-clip-high 1

    --optimizer adam
    --lr 3e-6
    --lr-decay-style WSD
    --lr-wsd-decay-style cosine
    --lr-warmup-iters 5
    --lr-decay-iters 50
    --lr-wsd-decay-iters 5
    --min-lr 0

    $WANDB_ARGS
"

declare -A COMBOS
# fp8: FP8 rollout checkpoint + BF16 master weights; DeepGEMM handles fp32 block scales on H100
COMBOS[fp8]="--hf-checkpoint $FP8_MODEL_DIR --load $BF16_MODEL_DIR"
# bf16: standard BF16 rollout + training weights
COMBOS[bf16]="--hf-checkpoint $BF16_MODEL_DIR"

ORDER=(fp8 bf16)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

declare -A RESULTS
for name in "${ORDER[@]}"; do
    combo_args="${COMBOS[$name]:-}"
    if [[ -z "$combo_args" ]]; then echo "Unknown combo: $name (valid: fp8 bf16)"; exit 2; fi
    log="$LOG_DIR/$name.log"

    echo "=============================================================="
    echo ">>> Stage3 E2E RL: $name"
    echo "=============================================================="

    start_ray
    set +e
    run_train "$COMMON_ARGS $combo_args --save $LOG_DIR/$name.ckpt" 2>&1 | tee "$log"
    set -e

    verdict_line="$(uv run python "$REPO_DIR/tests/sanity_check.py" "$log" 1 0)"
    echo "RESULT[$name]: $verdict_line"
    RESULTS[$name]="$verdict_line"
done

cleanup
echo
echo "================  STAGE3 E2E RL SUMMARY  ================"
fail=0
for name in "${ORDER[@]}"; do
    printf "  %-8s %s\n" "$name" "${RESULTS[$name]}"
    [[ "${RESULTS[$name]}" == PASS* ]] || fail=1
done
echo "=========================================================="
[[ $fail -eq 0 ]] && echo "ALL RUNS PASSED" || echo "SOME RUNS FAILED"
exit $fail
