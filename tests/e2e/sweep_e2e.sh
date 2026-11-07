#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end sweep on Qwen3.5-4B and a random-weight Qwen3.5 MoE, on the balanced math and vision split.
# Every combo changes several independent axes at once, so that each axis value runs at least once.
#
# Live combos generate with SGLang and cover the axes that need the engines:
#   placement:    the four layouts of docs/placement.md, each with PPO so that actor, critic and rollout are present
#   trainer:      slim-train with colocated rollout, slim-train-async with disaggregated rollout
#   weight sync:  full and merged LoRA weights, bf16 and block-FP8, over CUDA IPC and NCCL
#   rollout:      score-centering top-k and the rollout profiler
# `capture` saves its rollouts, and the script builds a fixture from them in which DP rank 1 holds the vision
# episodes and every other rank holds pure text.
# Replay combos train on that fixture without SGLang and cover the training axes: estimator, surrogate, loss
# normalization, CP2, fixed micro-batch size, LoRA checkpoints and references, and MoE expert LoRA with EP2.
#
# Compare `train/actor/pg_kl_k3` of a live combo across rollouts: with one optimizer step per rollout it measures
# the train/rollout mismatch, and it stays at its first-step level only if the engines receive the trainer weights.
# The fixture persists across runs, so the replay combos run without `capture` once it exists.
# `resume` and `kl_adapter` load the checkpoint that `save` writes, and `ppo_resume` loads the one that
# `ppo_colocate_rollout` writes. Checkpoints go to NVMe scratch, and the script deletes them when it exits.
# GPU counts scale with NUM_GPUS. `ppo_disagg` needs at least 4 GPUs.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
# Weightless block-FP8 rollout checkpoint; the first weight sync fills it.
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-FP8"
# Random-weight Qwen3.5 MoE with 4 layers and 16 experts. It shares the tokenizer and processor of Qwen3.5-4B, so
# it trains on the same fixture.
MOE_MODEL_DIR="$REPO_DIR/models/Qwen3.5-35B-A3B-tiny"
DATASET_DIR="$REPO_DIR/datasets/mixed"
PROFILE_DIR="$REPO_DIR/outputs/profiles"
REPLAY_DIR="${SWEEP_E2E_REPLAY_DIR:-$REPO_DIR/outputs/sweep-e2e-replay}"
REPLAY_DATA="$REPLAY_DIR/replay.pt"
CKPT_DIR="${SWEEP_E2E_CKPT_DIR:-/mnt/sagemaker-nvme/checkpoints/sweep-e2e}"
# A second HF directory with the MoE base weights, so the separate reference starts at KL 0.
MOE_REF_DIR="$CKPT_DIR/ref-Qwen3.5-35B-A3B-tiny"
LIVE_ROLLOUTS=2
HALF=$((NUM_GPUS / 2))
QUARTER=$((NUM_GPUS / 4))

COMMON_ARGS="
    --rollout-batch-size 4
    --n-samples-per-prompt 8
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train_balanced.parquet
    --rm-type math
    --apply-chat-template-kwargs {\"enable_thinking\":false}

    --activation-checkpointing
"

# The combo strings must stay on one line, because `read` splits them at the first newline.
DENSE="--hf-checkpoint $MODEL_DIR --max-context-len $(K 8) --max-tokens-per-gpu $(K 32)"
LIVE="$DENSE --num-rollout $LIVE_ROLLOUTS --rollout-num-gpus-per-replica 1 --sglang-mem-fraction-static 0.6"
LIVE+=" --sglang-mamba-ssm-dtype float32 --sglang-mamba-radix-cache-strategy extra_buffer_lazy"
LIVE+=" --sglang-max-mamba-cache-size 256 --sglang-page-size 64"
FP8="--rollout-load $FP8_MODEL_DIR --sglang-load-format dummy"
REPLAY_DATA_ARGS="--no-balance-data --load-debug-rollout-data $REPLAY_DATA"
REPLAY="--actor-num-gpus $NUM_GPUS $REPLAY_DATA_ARGS"
BATCH="--num-steps-per-rollout 1 --use-dynamic-batch-size"
CLIP="--eps-clip 0.2 --eps-clip-high 0.28"
GRPO="--advantage-estimator grpo --disable-group-advantage-std-normalization"
PPO="--advantage-estimator ppo_gae --value-clip 0.2 $CLIP --lr-critic 5e-5"
FULL_LR="--lr 1e-6"
# NeMo PeftConfig fields. alpha / dim = 2, so a merge with a wrong scale shows up as mismatch.
LORA='{"dim":32,"alpha":64,"target_modules":["*_proj"]}'
# `*.experts` adds the grouped experts of every MoE layer.
MOE_LORA='{"dim":32,"alpha":64,"target_modules":["*_proj","*.experts"]}'
# PEFT trains its fp32 adapters with AdamW; full-parameter training keeps the Muon default.
ADAMW="--optimizer adamw"
# The 1e-4 learning rate moves the policy away from the base, so `kl_loss` grows.
KL="--lr 1e-4 --kl-loss-coef 0.01"
PROFILE="--profile-step-start 0 --profile-step-end 1 --profile-dir $PROFILE_DIR"
SAVED_ADAPTER="$CKPT_DIR/save/iter_0000002/model"

# combo -> "train_cmd|expect_actor|expect_critic|combo_args"
declare -A COMBOS
# Live: LoRA with KL to the shared reference, merged into block-FP8 rollout weights over CUDA IPC.
COMBOS[capture]="slim-train|1|0|$LIVE $FP8 $GRPO --policy-surrogate mis $CLIP --use-score-centering 31 $BATCH --actor-num-gpus $NUM_GPUS --rollout-colocate --context-parallel-size 2 $KL --actor-peft-config $LORA $ADAMW --profile-target rollout $PROFILE --save-debug-rollout-data $REPLAY_DIR/rollout/{rollout_id}.pt"
# Live, layout 1: actor, critic and rollout on disjoint GPUs. Full bf16 weights over NCCL.
COMBOS[ppo_disagg]="slim-train-async|1|1|$LIVE $PPO $FULL_LR $BATCH --actor-num-gpus $QUARTER --critic-num-gpus $QUARTER --rollout-num-gpus $HALF"
# Live, layout 2: rollout shares the actor and critic GPUs. LoRA actor and critic, merged bf16 weights over CUDA
# IPC. Two optimizer steps partition the batch across steps, then ranks. The critic checkpoint holds its adapters
# and value head.
COMBOS[ppo_colocate_rollout]="slim-train|1|1|$LIVE $PPO --lr 1e-5 --num-steps-per-rollout 2 --use-dynamic-batch-size --actor-num-gpus $HALF --critic-num-gpus $HALF --rollout-colocate --actor-peft-config $LORA --critic-peft-config $LORA $ADAMW --save $CKPT_DIR/ppo/actor --critic-save $CKPT_DIR/ppo/critic --save-interval 2"
# Live, layout 3: critic shares the actor GPUs. LoRA actor and critic, merged block-FP8 weights over NCCL.
COMBOS[ppo_colocate_critic]="slim-train-async|1|1|$LIVE $FP8 $PPO --lr 1e-5 $BATCH --actor-num-gpus $HALF --critic-num-gpus $HALF --critic-colocate --rollout-num-gpus $HALF --actor-peft-config $LORA --critic-peft-config $LORA $ADAMW --loss-normalization-unit token"
# Live, layout 4: all roles share the same GPUs. Full bf16 weights over CUDA IPC, and round-robin assignment
# instead of Karmarkar-Karp balancing.
COMBOS[ppo_colocate_all]="slim-train|1|1|$LIVE $PPO $FULL_LR $BATCH --actor-num-gpus $NUM_GPUS --critic-num-gpus $NUM_GPUS --critic-colocate --rollout-colocate --no-balance-data"
# Replay.
COMBOS[grpo]="slim-train|1|0|$DENSE $REPLAY $GRPO --policy-surrogate cis $CLIP $BATCH $FULL_LR --num-rollout 2 --loss-normalization-unit trajectory"
# Replay: group std normalization exercises the Bessel-corrected per-group scatter. CP2 halves the logical DP size.
COMBOS[grpo_std]="slim-train|1|0|$DENSE $REPLAY --advantage-estimator grpo --policy-surrogate is $BATCH $FULL_LR --num-rollout 2 --context-parallel-size 2"
# Replay: fixed micro-batching requires exactly equal pack counts per rank, with no dynamic re-split to reconcile
# a mismatch.
COMBOS[gspo]="slim-train|1|0|$DENSE $REPLAY --advantage-estimator gspo --disable-group-advantage-std-normalization --eps-clip 3e-4 --eps-clip-high 4e-4 --num-steps-per-rollout 1 --micro-batch-size 2 $FULL_LR --num-rollout 2"
# Replay: LoRA with KL to the shared reference, the same model with its adapters disabled. Save at rollout 2, then
# resume with the PEFT config read from the checkpoint. After the resume, `kl_loss` continues from its level at the
# save only if the adapters load.
COMBOS[save]="slim-train|1|0|$DENSE $REPLAY $GRPO $CLIP $BATCH $KL --actor-peft-config $LORA $ADAMW --num-rollout 2 --save $CKPT_DIR/save --save-interval 2"
COMBOS[resume]="slim-train|1|0|$DENSE $REPLAY $GRPO $CLIP $BATCH $KL $ADAMW --num-rollout 3 --load $CKPT_DIR/save"
# Replay: the policy initializes its adapters from the adapter directory that `save` wrote, with its PEFT config and
# a fresh optimizer. The reference is the base with the same adapter, so `kl_loss` starts at 0.
COMBOS[kl_adapter]="slim-train|1|0|$DENSE $REPLAY $GRPO $CLIP $BATCH $KL $ADAMW --num-rollout 2 --load $SAVED_ADAPTER --ref-load $SAVED_ADAPTER"
# Replay: expert LoRA with EP2, KL to a separate reference model, and the torch profiler on the policy update.
COMBOS[moe]="slim-train|1|0|--hf-checkpoint $MOE_MODEL_DIR --max-context-len $(K 8) --max-tokens-per-gpu $(K 16) $REPLAY $GRPO $CLIP $BATCH --num-rollout 2 --expert-model-parallel-size 2 --lr 1e-3 --kl-loss-coef 0.01 --actor-peft-config $MOE_LORA $ADAMW --ref-load $MOE_REF_DIR --profile-target train_pg $PROFILE"
# Replay: resume the LoRA actor and critic of `ppo_colocate_rollout` on the same GPUs.
COMBOS[ppo_resume]="slim-train|1|1|$DENSE $PPO --lr 1e-5 $BATCH $ADAMW --actor-num-gpus $HALF --critic-num-gpus $HALF $REPLAY_DATA_ARGS --num-rollout 3 --load $CKPT_DIR/ppo/actor --critic-load $CKPT_DIR/ppo/critic"

# combo -> paths that an earlier combo writes.
declare -A REQUIRES
for name in grpo grpo_std gspo save moe; do REQUIRES[$name]="$REPLAY_DATA"; done
REQUIRES[resume]="$REPLAY_DATA $CKPT_DIR/save/latest_checkpointed_iteration.txt"
REQUIRES[kl_adapter]="$REPLAY_DATA $SAVED_ADAPTER/adapter_model.safetensors"
REQUIRES[ppo_resume]="$REPLAY_DATA $CKPT_DIR/ppo/actor/latest_checkpointed_iteration.txt $CKPT_DIR/ppo/critic/latest_checkpointed_iteration.txt"

ORDER=(capture grpo grpo_std gspo save resume kl_adapter moe ppo_disagg ppo_colocate_rollout ppo_resume ppo_colocate_critic ppo_colocate_all)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi
for name in "${ORDER[@]}"; do
    if [[ -z "${COMBOS[$name]:-}" ]]; then echo "Unknown combo: $name"; exit 2; fi
done

rm -rf "$CKPT_DIR"
mkdir -p "$MOE_REF_DIR"
ln -s "$MOE_MODEL_DIR"/* "$MOE_REF_DIR"/
trap 'rm -rf "$CKPT_DIR"' EXIT

start_ray

FAILED=()
for name in "${ORDER[@]}"; do
    IFS='|' read -r train_cmd expect_actor expect_critic combo_args <<< "${COMBOS[$name]}"
    missing=""
    for path in ${REQUIRES[$name]:-}; do
        [[ -e "$path" ]] || missing="$path"
    done
    if [[ -n "$missing" ]]; then
        echo ">>> Skipping $name: $missing does not exist"
        FAILED+=("$name")
        continue
    fi
    if [[ "$name" == ppo_disagg && "$NUM_GPUS" -lt 4 ]]; then
        echo ">>> Skipping $name: it needs at least 4 GPUs"
        FAILED+=("$name")
        continue
    fi
    if [[ "$name" == capture ]]; then rm -rf "$REPLAY_DIR"; fi
    echo ">>> Running $name  (cmd=$train_cmd, expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train_wait "$COMMON_ARGS $combo_args" "$train_cmd" || FAILED+=("$name")
    if [[ "$name" == capture && -e "$REPLAY_DIR/rollout/$((LIVE_ROLLOUTS - 1)).pt" ]]; then
        uv run python "$SCRIPT_DIR/prepare_heterogeneous_rollout.py" \
            --input "$REPLAY_DIR"/rollout/[0-9]*.pt \
            --output "$REPLAY_DATA" || FAILED+=("replay_fixture")
    fi
done

echo ">>> Failed combos: ${FAILED[*]:-none}"
