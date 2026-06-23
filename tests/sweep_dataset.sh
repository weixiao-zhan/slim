#!/usr/bin/env bash
# Dataset sweep: PPO across data modalities — text math (DAPO-17k) and vision (Geo3K) —
# on a single 8-GPU node with Qwen3.5-2B. Each combo runs a few rollout steps and is
# graded by tests/sanity_check.py (smoke + finite actor/critic loss + reward).
#
# Usage:  bash tests/sweep_dataset.sh [combo_name ...]
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
LOG_DIR="$REPO_DIR/outputs/dataset_sweep"
mkdir -p "$LOG_DIR"

# Sizes: 8 prompts x 4 samples = 32 episodes; 3 rollout steps. PPO on the simplest fully
# separate layout: 2 actor + 2 critic + 4 rollout GPUs, no colocation.
COMMON_ARGS="
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --max-context-len 8192
    --rollout-temperature 1
    --rollout-shuffle
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64

    --actor-num-gpus 2
    --critic-num-gpus 2
    --rollout-num-gpus 4
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator ppo_gae
    --gamma 1.0 --lambd 0.95 --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam
    --lr 1e-6
    --critic-lr 5e-5
    --hf-checkpoint $MODEL_DIR
"

# combo -> "expect_actor|expect_critic|combo_args"  (one line each; split on newlines by read)
declare -A COMBOS
COMBOS[math]="1|1|--prompt-data $REPO_DIR/datasets/dapo17k/train.parquet"
COMBOS[vision]="1|1|--prompt-data $REPO_DIR/datasets/geo3k/train.parquet"

ORDER=(math vision)
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
echo "================  DATASET SWEEP SUMMARY  ================"
fail=0
for name in "${ORDER[@]}"; do
    line="${RESULTS[$name]}"
    printf "  %-20s %s\n" "$name" "$line"
    [[ "$line" == PASS* ]] || fail=1
done
echo "========================================================"
[[ $fail -eq 0 ]] && echo "ALL COMBOS PASSED" || echo "SOME COMBOS FAILED"
exit $fail
