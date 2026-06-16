#!/usr/bin/env bash
# Routing replay on Qwen3.5-MoE (Qwen3.6-35B-A3B).
# Runs 5 rollout steps so we can compare train/rollout KL with and without
# --use-rollout-routing-replay. Dumps details to $DUMP_DIR for offline diffing.
#
# Toggle replay via: ROUTING_REPLAY=1 (default) | 0
source "$(dirname "$0")/common.sh"

MODEL_DIR="/opt/dlami/nvme/models/Qwen3.6-35B-A3B"
DATASET_DIR="$HOME/datasets/geo3k"
ROUTING_REPLAY="${ROUTING_REPLAY:-1}"
TAG_SUFFIX=$([ "$ROUTING_REPLAY" = "1" ] && echo "replay" || echo "baseline")
DUMP_DIR="${DUMP_DIR:-$HOME/outputs/routing-replay-qwen35moe-$TAG_SUFFIX}"

ROUTING_REPLAY_ARG=""
if [ "$ROUTING_REPLAY" = "1" ]; then
    ROUTING_REPLAY_ARG="--use-rollout-routing-replay"
fi

start_ray
run_train "
    --num-rollout 5
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 1

    --prompt-data $DATASET_DIR/train.parquet
    --rm-type math
    --rollout-shuffle
    --skip-eval-before-train

    --rollout-num-gpus-per-engine 2
    --sglang-ep 2
    --sglang-mem-fraction-static 0.8
    --sglang-attention-backend fa3
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-page-size 64
    --sglang-enforce-disable-flashinfer-allreduce-fusion
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
    --lr-warmup-iters 0
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98

    --hf-checkpoint $MODEL_DIR
    --save $DUMP_DIR/ckpt
    --dump-details $DUMP_DIR

    $ROUTING_REPLAY_ARG
"
