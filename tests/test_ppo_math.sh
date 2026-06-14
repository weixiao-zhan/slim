#!/usr/bin/env bash
# PPO on math (GSM8K). 4 actor + 4 critic GPUs colocated with 8 rollout.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${LLM_MODEL_DIR:-$HOME/models/Qwen3-1.7B-Base}"
DATASET_DIR="${LLM_DATASET_DIR:-$HOME/datasets}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/ppo-gsm8k-qwen3-1.7b}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 32
    --n-samples-per-prompt 16
    --max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/gsm8k/train.parquet
    --rm-type math
    --rollout-shuffle
    --eval-log-passrate
    --eval-interval 20
    --eval-prompt-data gsm8k_test $DATASET_DIR/gsm8k/test.parquet
    --eval-n-samples-per-prompt 1

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-attention-backend fa3
    --rollout-fault-tolerance
    --colocate

    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --critic-num-nodes 1
    --critic-num-gpus-per-node 4
    --attn-implementation flash_attention_3
    --master-weight-dtype fp32
    --compute-dtype bf16
    --gradient-checkpointing
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768

    --advantage-estimator ppo_gae
    --gamma 1.0
    --lambd 0.95
    --value-clip 0.2
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28

    --optimizer adam
    --lr 1e-5
    --critic-lr 5e-5
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR
    --save-interval 20

    $(wandb_args ppo-gsm8k-qwen3-1.7b)
"
