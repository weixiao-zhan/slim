#!/usr/bin/env bash
# GSPO on Geo3K VLM. 8 actor GPUs with 8 rollout engines.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${VLM_MODEL_DIR:-/home/ubuntu/models/Qwen3-VL-2B-Instruct}"
DATASET_DIR="${VLM_DATASET_DIR:-/home/ubuntu/datasets/geo3k}"
SAVE_DIR="${SAVE_DIR:-/home/ubuntu/outputs/gspo-geo3k-qwen3vl2b}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 32
    --n-samples-per-prompt 16
    --rollout-max-context-len 2048
    --rollout-temperature 1
    --num-steps-per-rollout 1
    --dynamic-sampling-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 48

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --log-passrate
    --eval-interval 20
    --eval-prompt-data geo3k $DATASET_DIR/test.parquet
    --n-samples-per-eval-prompt 1
    --eval-max-context-len 2048

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6
    --sglang-attention-backend fa3
    --sglang-mm-enable-dp-encoder
    --use-fault-tolerance

    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --attn-implementation flash_attention_3
    --gradient-checkpointing
    --update-weight-buffer-size 536870912
    --colocate
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --train-env-vars '{\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}'

    --advantage-estimator gspo
    --disable-grpo-std-normalization
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28

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

    $(wandb_args gspo-geo3k-qwen3vl2b)
"
