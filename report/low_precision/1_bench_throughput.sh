#!/usr/bin/env bash
# Benchmark prefill/decode throughput of every (model, precision) in common.py's registry
# (models: 9b 27b 35b; precisions: bf16 fp8 mxfp8 nvfp4) with sglang.bench_one_batch.
# Sweeps BATCH (default 1/2/4/8/16/32/64) at input=512, output=1024 (context=1536); writes
# prefill and median decode token/s to $OUT/result_<model>_<precision>.jsonl.
# Run 0_prepare_models.sh first. Formats sglang can't serve on SM120 (MXFP8 MoE) are skipped.
# Every run sizes the GDN state pool to the sweep's max batch (common.py mamba-flags: 4 slots per
# request, lazy extra-buffer radix cache). The BF16 27B and 35B-A3B are capped at batch 32: with
# 52-65GB of weights, the state pool for 64 concurrent requests does not fit on one card.
#
#   bash report/low_precision/1_bench_throughput.sh
#   MODELS="9b" PRECISIONS="fp8 nvfp4" BATCH="1 32" bash report/low_precision/1_bench_throughput.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=".venv/bin/python"
REG="$PY report/low_precision/common.py"
OUT="${OUT:-/mnt/sagemaker-nvme/experiments/low_precision/throughput}"
mkdir -p "$OUT"

BATCH="${BATCH:-1 2 4 8 16 32 64}"
INPUT_LENS="512"
OUTPUT_LENS="1024"
CONTEXT_LEN="${CONTEXT_LEN:-1536}"
# Static KV/mamba pool fraction. The BF16 35B-A3B (65GB weights) needs a larger pool than 0.8 to
# fit the batch-32 req_to_token / mamba slots; smaller footprints have ample headroom at any value.
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MODELS="${MODELS:-$($REG models)}"
PRECISIONS="${PRECISIONS:-$($REG precisions)}"

run() {
  local tag="$1" model="$2" batch="$3"; shift 3
  echo "==================== $tag (batch: $batch) ===================="
  local log="$OUT/bench_$tag.log"
  $PY -m sglang.bench_one_batch \
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
  for p in $PRECISIONS; do
    if ! $REG supported "$m" "$p"; then echo "[skip] ${m}_${p} (unsupported on SM120)"; continue; fi
    batch="$BATCH"
    if [ "$p" = "bf16" ] && [ "$m" != "9b" ]; then
      batch="$(echo "$BATCH" | tr ' ' '\n' | awk '$1<=32' | tr '\n' ' ')"
    fi
    max_batch="$(echo "$batch" | tr ' ' '\n' | sort -n | tail -1)"
    # shellcheck disable=SC2046  # flags are intentionally word-split
    run "${m}_${p}" "$($REG dir "$m" "$p")" "$batch" $($REG flags "$p") $($REG mamba-flags "$max_batch")
  done
done

echo "==================== SUMMARY (prefill + decode token/s) ===================="
for m in $MODELS; do
  for p in $PRECISIONS; do
    tag="${m}_${p}"
    [ -s "$OUT/result_$tag.jsonl" ] || continue
    echo "--- $tag ---"
    python3 -c "
import json
for line in open('$OUT/result_$tag.jsonl'):
    r = json.loads(line)
    print(f\"  bs={r['batch_size']:3d}  in={r['input_len']:5d}  out={r['output_len']:5d}  prefill={r['prefill_throughput']:8.0f} tok/s  decode={r['median_decode_throughput']:8.0f} tok/s\")
"
  done
done
