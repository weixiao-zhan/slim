#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate LoRA (PEFT) training on a two-GPU node with colocated rollout.
# Combos run one at a time, and each one waits for the previous one to finish.
# `resume`, `init`, and `kl_adapter` load the checkpoint that `save` writes, and `ppo_resume` loads the one that
# `ppo` writes, so select them together, for example `bash tests/sweep_peft.sh save resume init`.
# Checkpoints go to NVMe scratch instead of the small /tmp overlay, and the script deletes them when it exits.
#
# Compare `train/actor/pg_kl_k3` and `train/actor/pg_clipfrac` of `full` and `lora`, and of `lora` and `fp8`.
# With one optimizer step per rollout they measure the train/rollout mismatch, which checks the merged weights
# that the engines receive.
# The tiny MoE has random weights: its reward is always 0, and failures caused by its output are not bugs.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
# Weightless block-FP8 rollout checkpoint; the first weight sync fills it.
FP8_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B-FP8"
# Random-weight Qwen3.5 MoE with 4 layers and 16 experts.
MOE_MODEL_DIR="$REPO_DIR/models/Qwen3.5-35B-A3B-tiny"
DATASET_DIR="$REPO_DIR/datasets/mixed"
CKPT_DIR="${SWEEP_PEFT_CKPT_DIR:-/mnt/sagemaker-nvme/checkpoints/sweep-peft}"
# A second HF directory with the base weights, so the separate reference starts at KL 0 like the shared one.
REF_MODEL_DIR="$CKPT_DIR/ref-Qwen3.5-4B"

# NeMo PeftConfig fields. alpha / dim = 2, so a merge with a wrong scale shows up as mismatch.
LORA='{"dim":32,"alpha":64,"target_modules":["*_proj"]}'
# `*.experts` adds the grouped experts of every MoE layer.
MOE_LORA='{"dim":32,"alpha":64,"target_modules":["*_proj","*.experts"]}'
# LoRA uses about 10x the full-parameter learning rate.
LORA_ACTOR="--lr 1e-5 --actor-peft-config $LORA"

COMMON_ARGS="
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --num-steps-per-rollout 1
    --rollout-temperature 1
    --rollout-shuffle

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.6
    --sglang-mamba-ssm-dtype float32
    --sglang-mamba-radix-cache-strategy extra_buffer_lazy
    --sglang-max-mamba-cache-size 256
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus $NUM_GPUS
    --activation-checkpointing
    --use-dynamic-batch-size

    --eps-clip 0.2
    --eps-clip-high 0.28
"

# Dense lengths follow template_run.sh.
DENSE="--hf-checkpoint $MODEL_DIR --max-context-len $(K 24) --context-parallel-size 2 --max-tokens-per-gpu $(K 12)"
GRPO="--advantage-estimator grpo --disable-group-advantage-std-normalization"
DENSE_GRPO="$DENSE $(group_advantage_filter_args 16) $GRPO"
MOE="--hf-checkpoint $MOE_MODEL_DIR --max-context-len $(K 8) --expert-model-parallel-size 2 --max-tokens-per-gpu $(K 8)"
# Every MoE reward is 0, so the reward filter would accept no group and the entropy term gives the adapters a gradient.
MOE_GRPO="$MOE $GRPO --use-rollout-routing-replay --entropy-coef 0.01"
# Actor, critic, and rollout share both GPUs.
PPO="--advantage-estimator ppo_gae --value-clip 0.2 --lr 1e-5 --lr-critic 5e-5 --critic-colocate"
PPO_SAVE="--save $CKPT_DIR/ppo/actor --critic-save $CKPT_DIR/ppo/critic --save-interval 2"

# combo -> "expect_actor|expect_critic|combo_args"
declare -A COMBOS
# Full-parameter baseline and LoRA with the same rollout and loss settings.
COMBOS[full]="1|0|$DENSE_GRPO --num-rollout 4 --lr 1e-6"
COMBOS[lora]="1|0|$DENSE_GRPO --num-rollout 4 $LORA_ACTOR"
# Expert LoRA with EP2. As in `kl_shared`, the 1e-3 learning rate moves the policy away from the base, so `kl_loss`
# grows while `pg_kl_k3` stays at its first-step level only if the engines receive the merged expert adapters.
COMBOS[moe]="1|0|$MOE_GRPO --num-rollout 3 --lr 1e-3 --actor-peft-config $MOE_LORA --kl-loss-coef 0.01"
# Block-FP8 rollout quantizes the merged weights.
COMBOS[fp8]="1|0|$DENSE_GRPO --num-rollout 4 $LORA_ACTOR --rollout-load $FP8_MODEL_DIR --sglang-load-format dummy"
# Save at rollout 2, then resume to rollout 4 with the PEFT config read from the checkpoint.
COMBOS[save]="1|0|$DENSE_GRPO --num-rollout 2 $LORA_ACTOR --save $CKPT_DIR/save --save-interval 2"
COMBOS[resume]="1|0|$DENSE_GRPO --num-rollout 4 --lr 1e-5 --load $CKPT_DIR/save --save $CKPT_DIR/save --save-interval 2"
# Initialize the adapters from the adapter directory that `save` wrote, with its PEFT config and a fresh optimizer.
COMBOS[init]="1|0|$DENSE_GRPO --num-rollout 2 --lr 1e-5 --load $CKPT_DIR/save/iter_0000002/model"
# LoRA actor and critic; the critic checkpoint holds its adapters and value head.
COMBOS[ppo]="1|1|$DENSE $PPO --num-rollout 2 --actor-peft-config $LORA --critic-peft-config $LORA $PPO_SAVE"
COMBOS[ppo_resume]="1|1|$DENSE $PPO --num-rollout 4 --load $CKPT_DIR/ppo/actor --critic-load $CKPT_DIR/ppo/critic"
# KL to the base: the same model with its adapters disabled, then a separate reference model.
# The 1e-4 learning rate moves the policy away from the base, so `kl_loss` grows while `pg_kl_k3` stays at the
# `lora` level only if the engines receive the merged adapters.
KL_ACTOR="--lr 1e-4 --actor-peft-config $LORA --kl-loss-coef 0.01"
COMBOS[kl_shared]="1|0|$DENSE_GRPO --num-rollout 3 $KL_ACTOR"
COMBOS[kl_ref]="1|0|$DENSE_GRPO --num-rollout 3 $KL_ACTOR --ref-load $REF_MODEL_DIR"
# KL to an adapter reference: the policy starts from the adapter that `save` wrote, and the reference is the base
# with the same adapter, so `kl_loss` starts at 0.
SAVED_ADAPTER="$CKPT_DIR/save/iter_0000002/model"
COMBOS[kl_adapter]="1|0|$DENSE_GRPO --num-rollout 3 --lr 1e-4 --kl-loss-coef 0.01 --load $SAVED_ADAPTER --ref-load $SAVED_ADAPTER"

# combo -> checkpoint paths that an earlier combo writes. Without them a resume trains from scratch.
declare -A REQUIRES
REQUIRES[resume]="$CKPT_DIR/save/latest_checkpointed_iteration.txt"
REQUIRES[init]="$CKPT_DIR/save/iter_0000002/model/adapter_model.safetensors"
REQUIRES[kl_adapter]="$CKPT_DIR/save/iter_0000002/model/adapter_model.safetensors"
REQUIRES[ppo_resume]="$CKPT_DIR/ppo/actor/latest_checkpointed_iteration.txt $CKPT_DIR/ppo/critic/latest_checkpointed_iteration.txt"

ORDER=(full lora moe fp8 save resume init ppo ppo_resume kl_shared kl_ref kl_adapter)
if [[ $# -gt 0 ]]; then ORDER=("$@"); fi
for name in "${ORDER[@]}"; do
    if [[ -z "${COMBOS[$name]:-}" ]]; then echo "Unknown combo: $name"; exit 2; fi
done

rm -rf "$CKPT_DIR"
mkdir -p "$REF_MODEL_DIR"
ln -s "$MODEL_DIR"/* "$REF_MODEL_DIR"/
trap 'rm -rf "$CKPT_DIR"' EXIT

start_ray

FAILED=()
for name in "${ORDER[@]}"; do
    IFS='|' read -r expect_actor expect_critic combo_args <<< "${COMBOS[$name]}"
    missing=""
    for path in ${REQUIRES[$name]:-}; do
        [[ -e "$path" ]] || missing="$path"
    done
    if [[ -n "$missing" ]]; then
        echo ">>> Skipping $name: $missing does not exist"
        FAILED+=("$name")
        continue
    fi
    echo ">>> Running $name  (expect_actor=$expect_actor, expect_critic=$expect_critic)"
    run_train_wait "$COMMON_ARGS $combo_args" || FAILED+=("$name")
done

echo ">>> Failed combos: ${FAILED[*]:-none}"
