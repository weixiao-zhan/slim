#!/usr/bin/env bash
# GSPO on math (DAPO-17k). 8 actor GPUs colocated with rollout.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${LLM_MODEL_DIR:-$HOME/models/Qwen3.5-2B}"
DATASET_DIR="${LLM_DATASET_DIR:-$HOME/datasets/dapo17k}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/gspo-dapo17k-qwen35-2b}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --skip-eval-before-train

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-attention-backend fa3
    --rollout-fault-tolerance
    --colocate

    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768

    --advantage-estimator gspo
    --disable-rewards-std-normalization
    --old-logprob-source rollout
    --eps-clip 3e-4
    --eps-clip-high 4e-4

    --optimizer adam
    --lr 1e-5
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR
    --save-interval 20

    $(wandb_args gspo-dapo17k-qwen35-2b)
"
