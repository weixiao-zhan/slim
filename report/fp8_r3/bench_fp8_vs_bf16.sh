#!/usr/bin/env bash
# Benchmark FP8 vs BF16 decode/prefill throughput on Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-35B-A3B.
# Sweeps BATCH (default 1/2/4/8/16/32/64) at input=512, output=1024 (context=1536); outputs prefill
# and median decode token/s to outputs/fp8_vs_bf16_bench/result_<tag>.jsonl.
# The BF16 35B-A3B is skipped at batch 64: its 65GB of weights starve the GDN linear-attention
# recurrent-state pool below 64 concurrent slots, and the mem-fraction that reaches 64 slots leaves
# no room for prefill activations (hard OOM). FP8 halves the weights and fits.
# Override: MODELS="2b" or BATCH="1 32" bash report/fp8_r3/bench_fp8_vs_bf16.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT="$PWD/outputs/fp8_vs_bf16_bench"
mkdir -p "$OUT"

BATCH="${BATCH:-1 2 4 8 16 32 64}"
INPUT_LENS="512"
OUTPUT_LENS="1024"
CONTEXT_LEN="${CONTEXT_LEN:-1536}"
# Static KV/mamba pool fraction. The BF16 35B-A3B (65GB weights) needs a larger pool than 0.8 to
# fit the batch-32 req_to_token / mamba slots; smaller models have ample headroom at any value.
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MODELS="${MODELS:-2b 4b 9b 35b}"

# logical name -> dir basenames under models/
declare -A BF16_DIR=( [2b]="Qwen3.5-2B"        [4b]="Qwen3.5-4B"      [9b]="Qwen3.5-9B"      [35b]="Qwen3.5-35B-A3B" )
declare -A FP8_DIR=(  [2b]="Qwen3.5-2B-FP8"    [4b]="Qwen3.5-4B-FP8"  [9b]="Qwen3.5-9B-FP8"  [35b]="Qwen3.5-35B-A3B-FP8" )

run() {
  local tag="$1" model="$2" batch="$3"; shift 3
  echo "==================== $tag (batch: $batch) ===================="
  local log="$OUT/bench_$tag.log"
  uv run python -m sglang.bench_one_batch \
    --model-path "$model" \
    --trust-remote-code \
    --attention-backend flashinfer \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --batch-size $batch \
    --input-len $INPUT_LENS \
    --output-len $OUTPUT_LENS \
    --result-filename "$OUT/result_$tag.jsonl" \
    "$@" \
    > "$log" 2>&1
  echo "  exit=$? -> $log"
  grep -E "Prefill|Decode\.|median|Benchmark result|Error|Traceback" "$log" | tail -16
  echo
}

for m in $MODELS; do
  # BF16 35B-A3B can't fit batch 64 on one card (see header); cap its grid at 32.
  bf16_batch="$BATCH"
  if [ "$m" = "35b" ]; then
    bf16_batch="$(echo "$BATCH" | tr ' ' '\n' | awk '$1<=32' | tr '\n' ' ')"
  fi
  run "${m}_bf16" "$PWD/models/${BF16_DIR[$m]}" "$bf16_batch"
  run "${m}_fp8"  "$PWD/models/${FP8_DIR[$m]}" "$BATCH" --fp8-gemm-backend triton
done

echo "==================== SUMMARY (prefill + decode token/s) ===================="
for m in $MODELS; do
  for prec in bf16 fp8; do
    tag="${m}_${prec}"
    echo "--- $tag ---"
    python3 -c "
import json
for line in open('$OUT/result_$tag.jsonl'):
    r = json.loads(line)
    print(f\"  bs={r['batch_size']:3d}  in={r['input_len']:5d}  out={r['output_len']:5d}  prefill={r['prefill_throughput']:8.0f} tok/s  decode={r['median_decode_throughput']:8.0f} tok/s\")
" 2>/dev/null || echo "  (no result — see $OUT/bench_${tag}.log)"
  done
done
