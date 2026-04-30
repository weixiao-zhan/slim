#!/usr/bin/env bash
# GSPO on mixed math (DAPO-17k, text) + geometry (Geo3K, VLM). Qwen3.5-2B.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${VLM_MODEL_DIR:-$HOME/models/Qwen3.5-2B}"
DATASET_DIR="${VLM_DATASET_DIR:-$HOME/datasets/mixed_math_vlm}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/gspo-mixed-math-geo3k-qwen35-2b}"

# --dynamic-sampling-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
# --over-sampling-batch-size 64

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --rollout-max-context-len 8192
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --apply-chat-template-kwargs {\"enable_thinking\":false}
    --eval-log-passrate
    --skip-eval-before-train
    --eval-interval 20
    --eval-prompt-data math $DATASET_DIR/test_math.parquet geo3k $DATASET_DIR/test_geo3k.parquet
    --eval-n-samples-per-prompt 1
    --eval-max-context-len 8192

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.6
    --sglang-attention-backend fa3
    --sglang-mm-enable-dp-encoder
    --use-fault-tolerance

    --actor-num-nodes 1
    --actor-num-gpus-per-node $NUM_GPUS
    --attn-implementation flash_attention_3
    --gradient-checkpointing
    --colocate
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384

    --advantage-estimator gspo
    --disable-grpo-std-normalization
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.00
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00

    --optimizer adam
    --lr 1e-6
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR

    $(wandb_args gspo-mixed-math-geo3k-qwen35-2b)
"
