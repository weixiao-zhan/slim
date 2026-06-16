#!/usr/bin/env bash
# Compare decode throughput of FP8 block-GEMM backends on the 4B FP8 checkpoint,
# to decide which backend the rollout engine should use on this Blackwell (sm120) card.
#
# All backends here consume our fp32 block scales EXCEPT deep_gemm (ue8m0) — which we
# exclude because slim's online weight-sync emits fp32 scales (would be unsafe). The
# tuned Triton config (tools/triton_fp8_configs/, installed via patch_sglang.py) only affects
# the `triton` backend.
#
# Uses sglang.bench_one_batch: prefill + decode on a fixed batch, reports median decode
# token/s. Decode (small batch) is the rollout bottleneck, so that's the metric.
#
# Usage: bash tools/tuning/bench_fp8_backends.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${FP8_MODEL_DIR:-$HOME/models/Qwen3.5-4B-FP8}"
OUT="$HOME/outputs/fp8_backend_bench"
mkdir -p "$OUT"

# Decode-relevant batch sizes (rollout concurrency is 32; sweep around it).
BATCH="1 8 32 64"
INPUT_LEN=1024
OUTPUT_LEN=64

# Backends to compare. triton appears twice conceptually: the tuned config is already
# installed, so `triton` here = tuned triton. The untuned baseline (~2000 tok/s) was
# the prior run before configs were installed.
BACKENDS="triton flashinfer_cutlass flashinfer_trtllm"

echo "Model: $MODEL"
echo "Batches: [$BATCH]  input_len=$INPUT_LEN  output_len=$OUTPUT_LEN"
echo "Backends: $BACKENDS"
echo

for be in $BACKENDS; do
  echo "==================== backend=$be ===================="
  log="$OUT/bench_$be.log"
  uv run python -m sglang.bench_one_batch \
    --model-path "$MODEL" \
    --trust-remote-code \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --fp8-gemm-backend "$be" \
    --batch-size $BATCH \
    --input-len $INPUT_LEN \
    --output-len $OUTPUT_LEN \
    --result-filename "$OUT/result_$be.jsonl" \
    > "$log" 2>&1
  echo "  exit=$? -> $log"
  grep -E "Decode\.|median throughput|Benchmark|Error|Traceback" "$log" | tail -8
  echo
done

echo "==================== SUMMARY (decode token/s) ===================="
for be in $BACKENDS; do
  echo "--- $be ---"
  grep -E "Decode\." "$OUT/bench_$be.log" 2>/dev/null || echo "  (no decode result — see $OUT/bench_$be.log)"
done
