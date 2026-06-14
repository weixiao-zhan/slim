#!/usr/bin/env bash
# LoRA (PEFT) + GSPO on Geo3K VLM with Qwen3-VL-2B-Instruct. 8xH100 colocated.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${VLM_MODEL_DIR:-$HOME/models/Qwen3-VL-2B-Instruct}"
DATASET_DIR="${VLM_DATASET_DIR:-$HOME/datasets/geo3k}"
SAVE_DIR="${SAVE_DIR:-$HOME/outputs/lora-gspo-geo3k-qwen3vl2b}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 32
    --n-samples-per-prompt 16
    --max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 1
    --rollout-group-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 48

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --eval-log-passrate
    --eval-interval 20
    --eval-prompt-data geo3k $DATASET_DIR/test.parquet
    --eval-n-samples-per-prompt 1

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
    --use-peft
    --peft-config '{\"r\": 128, \"lora_alpha\": 256, \"target_modules\": \"all-linear\"}'

    --advantage-estimator gspo
    --disable-grpo-std-normalization
    --use-rollout-logprobs
    --eps-clip 3e-4
    --eps-clip-high 4e-4

    --optimizer adam
    --lr 5e-5
    --lr-warmup-iters 10
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $SAVE_DIR
    --save-interval 20

    $(wandb_args lora-gspo-geo3k-qwen3vl2b)
"
