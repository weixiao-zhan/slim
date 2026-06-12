#!/usr/bin/env bash
# GSPO on mixed math (DAPO-17k, text) + geometry (Geo3K, VLM), with FP8 rollout +
# BF16 training and TIS/MIS train-inference mismatch correction. Qwen3.5-2B.
#
# --hf-checkpoint -> forged FP8 model (rollout engine); --load -> BF16 model (training).
# slim quantizes BF16 weights to block-FP8 on each weight sync (see DESIGN_DECISIONS.md).
source "$(dirname "$0")/common.sh"

BF16_MODEL_DIR="${BF16_MODEL_DIR:-$HOME/models/Qwen3.5-2B}"
FP8_MODEL_DIR="${FP8_MODEL_DIR:-$HOME/models/Qwen3.5-2B-FP8}"
DATASET_DIR="${VLM_DATASET_DIR:-$HOME/datasets/mixed_math_vlm}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/gspo-mixed-math-geo3k-qwen35-2b-fp8-tis}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --max-context-len 8192
    --rollout-temperature 1
    --num-steps-per-rollout 1
    --rollout-group-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 96

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --apply-chat-template-kwargs {\"enable_thinking\":false}
    --eval-log-passrate
    --eval-interval 20
    --eval-prompt-data math $DATASET_DIR/test_math.parquet geo3k $DATASET_DIR/test_geo3k.parquet
    --eval-n-samples-per-prompt 1

    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.45
    --sglang-attention-backend flashinfer
    --sglang-mm-enable-dp-encoder
    --rollout-fault-tolerance

    --actor-num-nodes 1
    --actor-num-gpus-per-node $NUM_GPUS
    --attn-implementation flash_attention_2
    --gradient-checkpointing
    --colocate
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192

    --advantage-estimator gspo
    --disable-grpo-std-normalization
    --eps-clip 3e-4
    --eps-clip-high 4e-4
    --entropy-coef 0.00
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00

    --use-tis
    --custom-config-path examples/tis/mis.yaml

    --optimizer adam
    --lr 1e-5
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $FP8_MODEL_DIR
    --load $BF16_MODEL_DIR
    --save $SAVE_DIR

    $(wandb_args gspo-mixed-math-geo3k-qwen35-2b-fp8-tis)
"
