#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tune the block-FP8 W8A8 *fused-MoE* Triton kernel for Qwen3.5-35B-A3B's expert FFN on the
# local GPU, writing the tuned config into tools/triton_moe_configs/triton_<ver>/.
#
# WHY: sglang ships a Blackwell BF16 fused-MoE config (E=256,N=512,...Blackwell.json) in its
# wheel, but NO FP8 variant. So FP8 rollout of the MoE falls back to a default kernel config
# ("Using default MoE kernel config. Performance might be sub-optimal!" in the engine log),
# which understates FP8's decode throughput vs the (upstream-tuned) BF16 path. This script
# produces the missing E=256,N=512,...,dtype=fp8_w8a8,block_shape=[128,128].json (+ its _down
# companion) so the fused-MoE GEMMs run on a tuned schedule, matching how the dense models'
# FP8 GEMMs are already tuned (tools/triton_fp8_configs/, see tune_qwen35_fp8_gemm.sh).
#
# The expert-FFN MoE GEMM is keyed by (E, N) = (num_experts, moe_intermediate_size), NOT by the
# split nn.Linear shapes tuned in tune_qwen35_fp8_gemm.sh. For 35B-A3B: E=256, N=512.
#
# Uses sglang's own fused-MoE tuner (vendored as tuning_fused_moe_triton.py), which reads E/N/topk
# and the FP8 block_shape straight from the model's quantization_config — point it at the FP8 ckpt.
#
#   bash tools/tuning/tune_qwen35_fp8_moe.sh                 # tune 35B-A3B-FP8 on this GPU
#   MODEL_DIR=models/Qwen3.5-122B-A10B-FP8 bash tools/tuning/tune_qwen35_fp8_moe.sh
#
# After tuning, run `uv run python patch_sglang.py` to install the new config into the active
# sglang, then re-benchmark (report/fp8_r3/bench_fp8_vs_bf16.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_DIR="${MODEL_DIR:-models/Qwen3.5-35B-A3B-FP8}"
# Ray copies the script to a temp working dir and runs workers there, so a repo-relative (and
# symlinked) model path won't resolve. Pass the absolute, symlink-resolved path instead.
MODEL_ABS="$(readlink -f "$MODEL_DIR")"
TUNER="tools/tuning/tuning_fused_moe_triton.py"
# Cap the batch-size grid (default 4096) at what an RL rollout actually serves: small decode batches
# + prefill chunks. 2048 keeps a generous prefill-chunk margin while skipping the two slowest M tiles
# (3072/4096) we never hit. Override via MAX_BATCH; set MAX_BATCH=0 for the full upstream grid.
MAX_BATCH="${MAX_BATCH:-2048}"

# Triton version dir, matching sglang's get_moe_configs() lookup (configs/triton_<major_minor_patch>/).
TRITON_VER="$(uv run python -c 'import triton; print(triton.__version__.replace(".", "_"))')"
SAVE_DIR="$(pwd)/tools/triton_moe_configs/triton_${TRITON_VER}"
mkdir -p "$SAVE_DIR"

echo "Tuning FP8 fused-MoE for $MODEL_DIR ($MODEL_ABS) -> $SAVE_DIR (triton $TRITON_VER)"
# The tuner uses Ray, whose uv runtime-env hook requires the CWD to contain pyproject.toml — so we
# MUST run from the repo root, not the save dir. save_configs() writes the tuned JSON to a
# cwd-relative basename (repo root); we move it into $SAVE_DIR afterward.
# Snapshot existing E=*.json in repo root so we can identify what the tuner just produced.
before=$(ls E=*.json 2>/dev/null || true)
maxbatch_flag=()
[ "$MAX_BATCH" != "0" ] && maxbatch_flag=(--max-batch "$MAX_BATCH")
uv run python "$TUNER" \
  --model "$MODEL_ABS" \
  --tp-size 1 \
  --dtype fp8_w8a8 \
  "${maxbatch_flag[@]}" \
  --tune

# Move every freshly written config into the version-controlled save dir.
moved=0
for f in E=*.json; do
  [ -e "$f" ] || continue
  case "$before" in *"$f"*) continue ;; esac   # pre-existing, not ours
  mv "$f" "$SAVE_DIR/"; moved=$((moved+1))
done
echo "Done. Moved $moved tuned config(s) into $SAVE_DIR :"
ls -1 "$SAVE_DIR" | grep -iE "Blackwell|RTX_PRO" || true
echo "Next: uv run python patch_sglang.py   # install into active sglang"
