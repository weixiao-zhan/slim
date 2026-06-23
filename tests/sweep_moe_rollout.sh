#!/usr/bin/env bash
# MoE rollout-parallelism sweep: GSPO on Qwen3.5-MoE (Qwen3.6-35B-A3B) with rollout routing
# replay (R3) always on, sweeping the sglang rollout parallelism layout on one 8-GPU node:
#   tp1      -> tensor-parallel 1   (8 single-GPU replicas)
#   tp4      -> tensor-parallel 4   (2 replicas, pure TP)
#   tp4_ep4  -> tensor-parallel 4 + expert-parallel 4   (2 replicas, EP across experts)
# Each combo runs a few rollout steps and is graded by tests/sanity_check.py.
# The MoE checkpoint is referenced via the models/<name> symlink to NVMe.
#
# Usage:  bash tests/sweep_moe_rollout.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.6-35B-A3B"
DATASET_DIR="$REPO_DIR/datasets/geo3k"
LOG_DIR="$REPO_DIR/outputs/moe_rollout_sweep"
mkdir -p "$LOG_DIR"

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
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator gspo
    --disable-rewards-std-normalization
    --old-logprob-source rollout
    --eps-clip 3e-4
    --eps-clip-high 4e-4
    --use-rollout-routing-replay

    --optimizer adam
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

declare -A RESULTS
for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then echo "Unknown combo: $name"; exit 2; fi
    IFS='|' read -r expect_actor expect_critic combo_args <<< "$spec"
    log="$LOG_DIR/$name.log"

    echo "=============================================================="
    echo ">>> Running $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    echo "=============================================================="

    start_ray
    set +e
    run_train "$COMMON_ARGS $combo_args --save $LOG_DIR/$name.ckpt" 2>&1 | tee "$log"
    set -e

    verdict_line="$(uv run python "$REPO_DIR/tests/sanity_check.py" "$log" "$expect_actor" "$expect_critic")"
    echo "RESULT[$name]: $verdict_line"
    RESULTS[$name]="$verdict_line"
done

cleanup
echo
echo "================  MoE ROLLOUT SWEEP SUMMARY  ================"
fail=0
for name in "${ORDER[@]}"; do
    line="${RESULTS[$name]}"
    printf "  %-20s %s\n" "$name" "$line"
    [[ "$line" == PASS* ]] || fail=1
done
echo "============================================================"
[[ $fail -eq 0 ]] && echo "ALL COMBOS PASSED" || echo "SOME COMBOS FAILED"
exit $fail
