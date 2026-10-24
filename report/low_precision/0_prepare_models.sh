#!/usr/bin/env bash
# Generate the study's quantized checkpoints from the BF16 sources (models/<base>), one per
# (model, precision) in common.py's registry, into $QUANT_DIR/<base>-<suffix>:
#   fp8, mxfp8 -> tools/convert_hf_to_fp8.py  (data-free, recipe from tools/quant_recipes/)
#   nvfp4      -> tools/convert_hf_to_nvfp4.py (ModelOpt max calibration + HF export)
# Skips existing checkpoints and formats sglang can't serve for that model (MXFP8 MoE on SM120).
#
#   bash report/low_precision/0_prepare_models.sh
#   MODELS="9b" PRECISIONS="nvfp4" bash report/low_precision/0_prepare_models.sh
#   QUANT_DIR=/path bash report/low_precision/0_prepare_models.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=".venv/bin/python"
REG="$PY report/low_precision/common.py"
MODELS="${MODELS:-$($REG models)}"
PRECISIONS="${PRECISIONS:-fp8 mxfp8 nvfp4}"

for m in $MODELS; do
  src="$($REG dir "$m" bf16)"
  for p in $PRECISIONS; do
    if ! $REG supported "$m" "$p"; then echo "[skip] $m $p (unsupported on SM120)"; continue; fi
    dst="$($REG dir "$m" "$p")"
    if [ -s "$dst/config.json" ]; then echo "[skip] $dst (exists)"; continue; fi
    recipe="tools/quant_recipes/$($REG recipe "$p")"
    mkdir -p "$(dirname "$dst")"
    log="$dst.log"
    echo "[make] $dst  (recipe=$recipe, log=$log)"
    if [ "$p" = "nvfp4" ]; then
      $PY tools/convert_hf_to_nvfp4.py --model-dir "$src" --save-dir "$dst" --ref-config "$recipe" > "$log" 2>&1
    else
      $PY tools/convert_hf_to_fp8.py --model-dir "$src" --save-dir "$dst" --ref-config "$recipe" > "$log" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ]; then echo "  FAILED (exit $rc), see $log"; rm -rf "$dst"; fi
  done
done
