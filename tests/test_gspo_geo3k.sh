#!/usr/bin/env bash
# GSPO on Geo3K VLM
source "$(dirname "$0")/common.sh"

MODEL_DIR="${VLM_MODEL_DIR:-$HOME/models/Qwen3.5-2B}"
DATASET_DIR="${VLM_DATASET_DIR:-$HOME/datasets/geo3k}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/gspo-geo3k-qwen35-2b}"

start_ray
run_train "
    --num-rollout 50
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --max-context-len 4096
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --eval-log-passrate
    --eval-interval 20
    --eval-prompt-data geo3k $DATASET_DIR/test.parquet
    --eval-n-samples-per-prompt 1
    --skip-eval-before-train

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6
    --sglang-attention-backend flashinfer
    --sglang-mm-enable-dp-encoder
    --sglang-mamba-scheduler-strategy extra_buffer 
    --sglang-page-size 64
    --use-fault-tolerance

    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    --attn-implementation sdpa
    --gradient-checkpointing
    --colocate
    --use-dynamic-batch-size
    --max-tokens-per-gpu 6144

    --advantage-estimator gspo
    --disable-grpo-std-normalization
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
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

    $(wandb_args gspo-geo3k-qwen35-2b)
"
