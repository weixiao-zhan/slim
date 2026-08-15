#!/usr/bin/env bash
# End-to-end RL wall-time comparison: FP8 vs BF16 rollout on Qwen3.5-4B-Base (H100x8).
# PPO+GAE, mixed math+vision dataset. Uses the forged FP8 checkpoint for rollout and
# BF16 master weights for the training actor (same recipe as sweep_fp8.sh). On H100,
# sglang uses DeepGEMM by default for FP8 GEMMs (fp32 block scales), so no gemm-backend
# override is needed.
#
# Usage:  bash report/fp8_r3/stage3_e2erl.sh [fp8|bf16 ...]   # default: fp8 then bf16
#
# Prerequisites:
#   models/Qwen3.5-4B-Base       (BF16 master weights + HF checkpoint for bf16 run)
#   models/Qwen3.5-4B-Base-FP8   (FP8 rollout checkpoint for fp8 run)
#   datasets/mixed/{train,test_math,test_vision}.parquet  (from tests/prepare_mixed.py)
source "$(dirname "$0")/../../tests/common.sh"

BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-Base"
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-Base-FP8"
DATASET_DIR="$REPO_DIR/datasets/mixed"

COMMON_ARGS="
    --num-rollout 80
    --rollout-batch-size 32
    --n-samples-per-prompt 16
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rollout-temperature 1
    --rollout-shuffle
    --max-context-len 8192
    --rm-type math
    --eval-interval 10
    --skip-eval-before-train
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet

    --rollout-colocate
    --rollout-num-gpus-per-replica 1
    --rollout-concurrency-per-replica 128
    --sglang-attention-backend fa3
    --sglang-mem-fraction-static 0.8
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --sglang-enable-metrics

    --actor-num-gpus 4
    --critic-num-gpus 4
    --activation-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768

    --advantage-estimator ppo_gae
    --policy-surrogate ppo_clip
    --old-logprob-source rollout
    --eps-clip 0.2
    --eps-clip-high 0.28
    --value-clip 0.2

    --lr 1e-5
    --lr-decay-style WSD
    --lr-wsd-decay-style cosine
    --lr-warmup-iters 5
    --lr-wsd-decay-iters 5
    --lr-critic-value-head 5e-5
"

declare -A COMBOS
# fp8: FP8 rollout checkpoint + BF16 master weights; DeepGEMM handles fp32 block scales on H100
COMBOS[fp8]="--hf-checkpoint $FP8_MODEL_DIR --load $BF16_MODEL_DIR"
# bf16: standard BF16 rollout + training weights
COMBOS[bf16]="--hf-checkpoint $BF16_MODEL_DIR"

ORDER=(fp8 bf16)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi

# Submit both combos to one cluster; they queue and run sequentially.
# See tests/RUNNING_TESTS.md for watching/grading via ray job logs.
start_ray

for name in "${ORDER[@]}"; do
    combo_args="${COMBOS[$name]:-}"
    if [[ -z "$combo_args" ]]; then echo "Unknown combo: $name (valid: fp8 bf16)"; exit 2; fi
    echo ">>> Submitting $name"
    run_train "$COMMON_ARGS $combo_args $(wandb_args fp8_e2erl_$name)"
done
