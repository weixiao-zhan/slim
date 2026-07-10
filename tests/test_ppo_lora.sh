#!/usr/bin/env bash
# PPO + LoRA (PEFT) on the mixed (math+vision) dataset with Qwen3.5-2B. --use-peft applies
# LoRA to both actor and critic. Actor, critic, and rollout all colocate on the 8-GPU node.
source "$(dirname "$0")/common.sh"

MODEL_DIR="$REPO_DIR/models/Qwen3.5-2B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
SAVE_DIR="$REPO_DIR/outputs/ppo-lora-qwen35-2b"
mkdir -p "$SAVE_DIR"

start_ray
run_train "
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 4
    --max-context-len 8192
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle

    --rollout-num-gpus-per-replica 1
    --sglang-mem-fraction-static 0.7
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-colocate

    --actor-num-gpus 8
    --critic-num-gpus 8
    --critic-colocate
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --use-peft
    --peft-config {\"r\":128,\"lora_alpha\":256,\"target_modules\":\"all-linear\"}

    --advantage-estimator ppo_gae
    --value-clip 0.2
    --eps-clip 0.2
    --eps-clip-high 0.28

    --optimizer adam
    --lr 3e-6
    --lr-critic 5e-5
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR
    --save-interval 20

    $(wandb_args ppo-lora-qwen35-2b)
"
