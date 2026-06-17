#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
BF16_MODEL_DIR="$REPO_DIR/models/Qwen3.5-4B"
DATASET_DIR="$REPO_DIR/datasets/mixed"
SAVE_DIR="$REPO_DIR/outputs/grpo-mixed-cis-qwen35-4b-profile"
PROFILE_DIR="$REPO_DIR/outputs/profiles"

start_ray
run_train "
    --num-rollout 3
    --rollout-batch-size 8
    --n-samples-per-prompt 2
    --max-context-len 8192
    --apply-chat-template-kwargs {\"enable_thinking\":false}
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --skip-eval-before-train

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --rollout-fault-tolerance
    --colocate

    --actor-num-nodes 1
    --actor-num-gpus-per-node $NUM_GPUS
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384

    --profile-target train_pg rollout
    --profile-step-start 0
    --profile-step-end 1
    --profile-dir $PROFILE_DIR

    --advantage-estimator grpo
    --disable-rewards-std-normalization
    --policy-surrogate cis
    --old-logprob-source rollout
    --eps-clip 1
    --eps-clip-high 1

    --optimizer adam
    --lr 3e-6
    --lr-warmup-iters 0
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $FP8_MODEL_DIR
    --save $SAVE_DIR

    $(wandb_args grpo-mixed-cis-qwen35-4b-fp8-fp32-profile)
"
