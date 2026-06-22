#!/usr/bin/env bash
# Section 1 throughput: decode token/s, FP8 (triton block-GEMM) vs BF16, on the FP8/R3 study models
# Qwen3.5-2B (dense) and Qwen3.5-35B-A3B (MoE), swept at concurrency 1/8/16/32, on this Blackwell
# (sm120) card.
#
# FP8 uses --fp8-gemm-backend triton (DeepGEMM is disabled on SM120). BF16 serves the unquantized
# base model. Attention backend = flashinfer (the rollout default on SM120), mem-fraction matches the
# training rollout config. Uses sglang.bench_one_batch (online path, per the experiment design):
# prefill + decode on a fixed batch; reports median decode token/s. --batch-size is the in-flight
# concurrency.
#
# Usage:
#   bash report/fp8_r3/bench_fp8_vs_bf16.sh                  # both models (2b, 35b)
#   MODELS="2b" bash report/fp8_r3/bench_fp8_vs_bf16.sh      # one model
#   BATCH="1 8 16 32" MODELS="2b" bash report/fp8_r3/bench_fp8_vs_bf16.sh   # custom sweep
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT="$PWD/outputs/fp8_vs_bf16_bench"
mkdir -p "$OUT"

BATCH="${BATCH:-1 8 16 32}"
INPUT_LEN=1024
OUTPUT_LEN=512
MODELS="${MODELS:-2b 35b}"

# logical name -> dir basenames under models/
declare -A BF16_DIR=( [2b]="Qwen3.5-2B"        [35b]="Qwen3.5-35B-A3B" )
declare -A FP8_DIR=(  [2b]="Qwen3.5-2B-FP8"    [35b]="Qwen3.5-35B-A3B-FP8" )

run() {
  local tag="$1" model="$2"; shift 2
  echo "==================== $tag ===================="
  local log="$OUT/bench_$tag.log"
  uv run python -m sglang.bench_one_batch \
    --model-path "$model" \
    --trust-remote-code \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --result-filename "$OUT/result_$tag.jsonl" \
    "$@" \
    > "$log" 2>&1
  echo "  exit=$? -> $log"
  grep -E "Decode\.|median|Benchmark result|Error|Traceback" "$log" | tail -8
  echo
}

for m in $MODELS; do
  run "${m}_bf16" "$PWD/models/${BF16_DIR[$m]}"
  run "${m}_fp8"  "$PWD/models/${FP8_DIR[$m]}" --fp8-gemm-backend triton
done

echo "==================== SUMMARY (decode token/s) ===================="
for m in $MODELS; do
  for prec in bf16 fp8; do
    tag="${m}_${prec}"
    echo "--- $tag ---"
    grep -E "Decode\.|median" "$OUT/bench_$tag.log" 2>/dev/null || echo "  (no decode result — see $OUT/bench_$tag.log)"
  done
done
