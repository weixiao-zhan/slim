#!/usr/bin/env bash
# Compare sglang MoE-runner backends for the 35B-A3B's block-FP8 (w8a8, 128x128) experts on this
# SM120 card. The main throughput sweep (bench_fp8_vs_bf16.sh) leaves --moe-runner-backend at its
# default (auto -> triton); this script sweeps the backends listed at
#   https://docs.sglang.io/docs/advanced_features/expert_parallelism#backends-for-moe-computation
# to see which actually run with block-FP8 on Blackwell and how their decode throughput compares to
# the (now tuned) Triton kernel. Backends that don't support block-FP8 / SM120 are expected to error;
# we record that as a result ("unsupported") rather than a silent omission.
#
# Run AFTER tune_qwen35_fp8_moe.sh + patch_sglang.py, so `triton` here = the tuned kernel.
#
#   bash report/fp8_r3/bench_moe_backends.sh
#   BACKENDS="triton cutlass" BATCH="16 32" bash report/fp8_r3/bench_moe_backends.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT="$PWD/outputs/fp8_vs_bf16_bench/moe_backends"
mkdir -p "$OUT"

MODEL="$PWD/models/Qwen3.5-35B-A3B-FP8"
BATCH="${BATCH:-16 32}"
INPUT_LENS="512"
OUTPUT_LENS="1024"
CONTEXT_LEN="${CONTEXT_LEN:-1536}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
# Candidate MoE runner backends. `auto` shows what sglang picks unprompted; `triton` is the tuned
# baseline. The flashinfer_*/cutlass/deep_gemm ones are Blackwell/FP8-oriented but may reject 128x128
# block-FP8 — that rejection is the finding. Override via BACKENDS env.
BACKENDS="${BACKENDS:-auto triton triton_kernel cutlass deep_gemm flashinfer_cutlass flashinfer_trtllm}"

run() {
  local backend="$1"
  local tag="35b_fp8_moe_${backend}"
  local log="$OUT/bench_${tag}.log"
  echo "==================== $backend ===================="
  uv run python -m sglang.bench_one_batch \
    --model-path "$MODEL" \
    --trust-remote-code \
    --attention-backend flashinfer \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --batch-size $BATCH \
    --input-len $INPUT_LENS \
    --output-len $OUTPUT_LENS \
    --fp8-gemm-backend triton \
    --moe-runner-backend "$backend" \
    --result-filename "$OUT/result_${tag}.jsonl" \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$OUT/result_${tag}.jsonl" ]; then
    echo "  OK (exit $rc)"
    grep -E "MoE kernel config|Decode\.|median" "$log" | tail -4
  else
    echo "  FAILED (exit $rc) — likely unsupported on SM120+block-FP8; see $log"
    grep -iE "not support|unsupported|assert|error|raise|valueerror|runtimeerror" "$log" | grep -ivE "Capturing|Multi-thread" | tail -4
  fi
  echo
}

for b in $BACKENDS; do run "$b"; done

echo "==================== SUMMARY (decode tok/s @ each batch) ===================="
for b in $BACKENDS; do
  tag="35b_fp8_moe_${b}"
  f="$OUT/result_${tag}.jsonl"
  if [ -s "$f" ]; then
    python3 -c "
import json
print('--- $b ---')
for line in open('$f'):
    r=json.loads(line)
    print(f\"  bs={r['batch_size']:>3}  prefill={r['prefill_throughput']:>8.0f}  decode={r['median_decode_throughput']:>8.0f} tok/s\")
"
  else
    echo "--- $b ---"
    echo "  (no result — unsupported or errored; see $OUT/bench_${tag}.log)"
  fi
done
