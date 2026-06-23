#!/usr/bin/env bash
# Placement sweep: exercise every (rollout_colocate, critic_colocate) combo plus
# an HSDP control on a single 8-GPU node. Each combo runs a few rollout steps on the
# small mixed (math+vision) dataset and is graded by tests/sanity_check.py
# (smoke + finite-loss + non-degenerate reward).
#
# Usage:  bash tests/sweep_placement.sh [combo_name ...]
#   With no args, runs every combo. With args, runs only the named combos.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
LOG_DIR="$REPO_DIR/outputs/placement_sweep"
mkdir -p "$LOG_DIR"

# Sizes: 8 prompts x 4 samples = 32 episodes; gbs=32 -> one train step per rollout.
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

    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --optimizer adam
    --lr 1e-6
    --hf-checkpoint $MODEL_DIR
"

# Single-line: these are embedded in COMBOS specs split by `read`, which stops at newlines.
PPO_ARGS="--advantage-estimator ppo_gae --gamma 1.0 --lambd 0.95 --value-clip 0.2 --eps-clip 0.2 --eps-clip-high 0.28 --critic-lr 5e-5"
GRPO_ARGS="--advantage-estimator grpo --disable-rewards-std-normalization"

# combo -> "train_script|expect_actor|expect_critic|cluster_args"
declare -A COMBOS
COMBOS[default]="train.py|1|1|$PPO_ARGS --actor-num-gpus 2 --critic-num-gpus 2 --rollout-num-gpus 4"
COMBOS[default_async]="train_async.py|1|1|$PPO_ARGS --actor-num-gpus 2 --critic-num-gpus 2 --rollout-num-gpus 4 --update-weights-interval 1"
COMBOS[colocate_rollout]="train.py|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --rollout-colocate"
COMBOS[colocate_critic]="train.py|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-num-gpus 4"
COMBOS[colocate_critic_async]="train_async.py|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-num-gpus 4 --update-weights-interval 1"
COMBOS[full_colocate]="train.py|1|1|$PPO_ARGS --actor-num-gpus 4 --critic-num-gpus 4 --critic-colocate --rollout-colocate"
# HSDP control: 8 actor GPUs, 2-GPU replicas (replicate=4, shard=2), colocated rollout.
COMBOS[grpo_hsdp]="train.py|1|0|$GRPO_ARGS --actor-num-gpus 8 --actor-num-gpus-per-replica 2 --rollout-colocate"

ORDER=(default default_async colocate_rollout colocate_critic colocate_critic_async full_colocate grpo_hsdp)

# Allow running a subset by name.
if [[ $# -gt 0 ]]; then
    ORDER=("$@")
fi

declare -A RESULTS
for name in "${ORDER[@]}"; do
    spec="${COMBOS[$name]:-}"
    if [[ -z "$spec" ]]; then
        echo "Unknown combo: $name"; exit 2
    fi
    IFS='|' read -r script expect_actor expect_critic cluster_args <<< "$spec"
    log="$LOG_DIR/$name.log"

    echo "=============================================================="
    echo ">>> Running $name  (script=$script, expect_actor=$expect_actor, expect_critic=$expect_critic)"
    echo "=============================================================="

    start_ray
    set +e
    run_train "$COMMON_ARGS $cluster_args --save $LOG_DIR/$name.ckpt" "$REPO_DIR/$script" 2>&1 | tee "$log"
    set -e

    verdict_line="$(uv run python "$REPO_DIR/tests/sanity_check.py" "$log" "$expect_actor" "$expect_critic")"
    echo "RESULT[$name]: $verdict_line"
    RESULTS[$name]="$verdict_line"
done

cleanup
echo
echo "================  PLACEMENT SWEEP SUMMARY  ================"
fail=0
for name in "${ORDER[@]}"; do
    line="${RESULTS[$name]}"
    printf "  %-28s %s\n" "$name" "$line"
    [[ "$line" == PASS* ]] || fail=1
done
echo "=========================================================="
if [[ $fail -eq 0 ]]; then
    echo "ALL COMBOS PASSED"
else
    echo "SOME COMBOS FAILED"
fi
exit $fail
