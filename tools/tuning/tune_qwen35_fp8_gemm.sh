#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tune the block-FP8 W8A8 GEMM Triton kernel for the Qwen3.5 family's FUSED runtime
# GEMM shapes on the local GPU, writing tuned configs into tools/triton_fp8_configs/.
#
# IMPORTANT: shapes are the FUSED shapes sglang actually dispatches at runtime — NOT the
# split nn.Linear (out,in) pairs. sglang fuses q+k+v(+gate) into one qkv_proj, gate+up
# into one gate_up_proj, and the GatedDeltaNet q+k+v+z into in_proj_qkvz. Derived from
# sglang/srt/models/qwen3_5.py and validated against the installed 4B configs:
#   qkv_proj      N = head_dim*(2*num_attn_heads + 2*num_kv_heads)   K = hidden   (attn_output_gate doubles Q)
#   o_proj        N = hidden                                          K = num_attn_heads*head_dim
#   gate_up_proj  N = 2*intermediate                                  K = hidden
#   down_proj     N = hidden                                          K = intermediate
#   in_proj_qkvz  N = 2*key_dim + 2*value_dim                         K = hidden   (FP8 under official recipe)
#   out_proj(la)  N = hidden                                          K = value_dim  (== attn o_proj, deduped)
# 35B-A3B's expert FFN uses the fused-MoE kernel (separate tuning, see tools/triton_moe_configs/).
#
# These match the OFFICIAL Qwen3.5-FP8 recipe (linear_attn in_proj_qkvz / out_proj kept in
# FP8; only conv1d/in_proj_a/in_proj_b stay bf16). See tools/fp8_recipes/qwen35_official.json.
#
# Which models to tune is selected by the first arg (default: all):
#   bash tools/tuning/tune_qwen35_fp8_gemm.sh 4b      # just 4B's shapes
#   bash tools/tuning/tune_qwen35_fp8_gemm.sh all     # union across 4B/9B/27B/35B-A3B
# Accepts 4b | 9b | 27b | 35b | all (matched against each shape's "model:" tag below).
#
# Batch-size grid is trimmed to what an RL rollout actually hits (decode small-M +
# prefill chunks); the upstream default (18 sizes up to 4096) re-runs the full 1280-
# config search per size and takes hours per shape. Override via BATCH_SIZES env.
set -euo pipefail
cd "$(dirname "$0")/../.."

SAVE_PATH="$(pwd)/tools/triton_fp8_configs"
SCRIPT="tools/tuning/tuning_block_wise_kernel.py"
mkdir -p "$SAVE_PATH"

MODEL_SET="${1:-all}"
BATCH_SIZES="${BATCH_SIZES:-1 8 16 32 64 128 256 512 1024 2048}"

# Single source of truth: every fused FP8 GEMM (N K) shape across the family, each tagged
# "model:N K" so a model subset can be selected without a second array. Tuning dedups by
# (N,K) below, so shapes shared between models are tuned once even when MODEL_SET=all.
# Per-model arch params (head_dim=256, attn_output_gate doubles Q):
#   4B:      H=2560 I=9216  nh=16 nkv=4 key_dim=2048 value_dim=4096
#   9B:      H=4096 I=12288 nh=16 nkv=4 value_dim=4096
#   27B:     H=5120 I=17408 nh=24 nkv=4 value_dim=6144
#   35B-A3B: H=2048 nh=16 nkv=2 value_dim=4096 shared_I=512  (routed experts -> fused-MoE)
TAGGED_SHAPES=(
  "4b:10240 2560"    # qkv_proj
  "4b:2560 4096"     # o_proj (== linear_attn out_proj)
  "4b:18432 2560"    # gate_up_proj
  "4b:2560 9216"     # down_proj
  "4b:12288 2560"    # linear_attn in_proj_qkvz
  "9b:10240 4096"    # qkv_proj
  "9b:4096 4096"     # o_proj
  "9b:24576 4096"    # gate_up_proj
  "9b:4096 12288"    # down_proj
  "9b:12288 4096"    # in_proj_qkvz
  "27b:14336 5120"   # qkv_proj
  "27b:5120 6144"    # o_proj
  "27b:34816 5120"   # gate_up_proj
  "27b:5120 17408"   # down_proj
  "27b:16384 5120"   # in_proj_qkvz
  "35b:9216 2048"    # qkv_proj
  "35b:2048 4096"    # o_proj
  "35b:12288 2048"   # in_proj_qkvz
  "35b:1024 2048"    # shared_expert gate_up
  "35b:2048 512"     # shared_expert down
)

# Select shapes for the requested model (or all), dedup (N,K), preserve order.
declare -A _seen=()
SHAPES=()
for tagged in "${TAGGED_SHAPES[@]}"; do
  model="${tagged%%:*}"
  nk="${tagged#*:}"
  case "$MODEL_SET" in
    all|ALL) ;;
    *) [[ "${MODEL_SET,,}" == "$model" ]] || continue ;;
  esac
  [[ -n "${_seen[$nk]:-}" ]] && continue
  _seen[$nk]=1
  SHAPES+=("$nk")
done

# BATCH_SIZES=full -> use the upstream script's full default grid (18 sizes up to
# 4096); this is the comprehensive, PR-quality tune. Any other value is passed as an
# explicit (trimmed) grid for fast validation runs.
batch_flag=(--batch-sizes $BATCH_SIZES)
if [[ "$BATCH_SIZES" == "full" ]]; then
  batch_flag=()
fi

echo "Tuning ${#SHAPES[@]} shapes (set=$MODEL_SET, batch_sizes=[$BATCH_SIZES]) -> $SAVE_PATH"
i=0
for nk in "${SHAPES[@]}"; do
  read -r N K <<< "$nk"
  i=$((i+1))
  echo "=== [$i/${#SHAPES[@]}] N=$N K=$K ==="
  uv run python "$SCRIPT" \
    --N "$N" --K "$K" \
    --input-type fp8 \
    --out-dtype bfloat16 \
    --block-n 128 --block-k 128 \
    "${batch_flag[@]}" \
    --save-path "$SAVE_PATH"
done
echo "Done. Configs in $SAVE_PATH"
