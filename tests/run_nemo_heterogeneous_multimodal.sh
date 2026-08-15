#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Qwen3.5 MoE actor steps with modality-skewed DP ranks across the CP/EP matrix.
source "$(dirname "$0")/common.sh"

MODEL_DIR="${NEMO_HETEROGENEOUS_MODEL_DIR:-$REPO_DIR/models/Qwen3.5-35B-A3B}"
RESULT_DIR="${NEMO_HETEROGENEOUS_RESULT_DIR:-/tmp/slim-nemo-heterogeneous-multimodal}"
ROLLOUT_SOURCE="${NEMO_HETEROGENEOUS_ROLLOUT_SOURCE:-/tmp/slim-nemo-grad-norm-matrix/rollout.pt}"
ROLLOUT_DATA="$RESULT_DIR/rollout.pt"

if [[ "$NUM_GPUS" -ne 8 ]]; then
    echo "heterogeneous multimodal qualification requires exactly 8 GPUs, found $NUM_GPUS" >&2
    exit 1
fi
if [[ ! -f "$ROLLOUT_SOURCE" ]]; then
    echo "fixed rollout source not found: $ROLLOUT_SOURCE" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
uv run python tests/prepare_nemo_heterogeneous_rollout.py \
    --input "$ROLLOUT_SOURCE" \
    --output "$ROLLOUT_DATA"

run_case() {
    local case_name="$1"
    local cp_size="$2"
    local ep_size="$3"
    shift 3
    env -u LD_LIBRARY_PATH uv run torchrun \
        --standalone \
        --nproc-per-node 8 \
        tests/backends/nemo/run_training_qualification.py \
        --rollout-data "$ROLLOUT_DATA" \
        \
        --role actor \
        --context-parallel-size "$cp_size" \
        --expert-parallel-size "$ep_size" \
        --heterogeneous-multimodal \
        --max-tokens-per-gpu "$(K 2)" \
        \
        --checkpoint "$MODEL_DIR" \
        --output "$RESULT_DIR/$case_name.json" \
        "$@" \
        2>&1 | tee "$RESULT_DIR/$case_name.log"
}

cd "$REPO_DIR"
for topology in \
    "cp1_ep4 1 4" \
    "cp1_ep8 1 8"
do
    read -r topology_name cp_size ep_size <<<"$topology"
    run_case "${topology_name}_frozen_vision" "$cp_size" "$ep_size" --freeze-vision-tower
    run_case "${topology_name}_trainable_vision" "$cp_size" "$ep_size"
done
