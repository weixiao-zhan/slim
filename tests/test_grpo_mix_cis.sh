#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL_DIR="${VLM_MODEL_DIR:-$HOME/models/Qwen3.5-4B-Base}"
DATASET_DIR="${VLM_DATASET_DIR:-$HOME/datasets/mixed}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/grpo-mix-cis-qwen35-4b}"

start_ray
run_train "
    --num-rollout 20
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --max-context-len 32768
    --rollout-temperature 1
    --num-steps-per-rollout 1
    --rollout-group-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 96

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --eval-log-passrate
    --eval-interval 20
    --eval-prompt-data math $DATASET_DIR/test_math.parquet vision $DATASET_DIR/test_vision.parquet
    --eval-n-samples-per-prompt 2

    --rollout-num-gpus-per-engine 1
    --rollout-concurrency-per-engine 64
    --sglang-mem-fraction-static 0.8
    --sglang-max-running-requests 48
    --sglang-attention-backend fa3
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
    --max-tokens-per-gpu 32768

    --advantage-estimator grpo
    --disable-rewards-std-normalization
    --policy-surrogate cis
    --old-logprob-source rollout
    --eps-clip 1
    --eps-clip-high 1

    --optimizer adam
    --lr 3e-6
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR

    $(wandb_args grpo-mix-cis-qwen35-4b)
"
