#!/usr/bin/env bash
# Drive the full FP8/R3 experiment matrix (Sections 2 & 3). Section 1 (throughput) is a separate,
# already-run sweep: report/fp8_r3/bench_fp8_vs_bf16.sh.
#
# Idempotent / resumable: each step skips if its output already exists (records.jsonl for stage 1,
# kl/summary.json for stage 2). Delete a run dir under $ROOT (below) to redo it.
#
# Stage 1 (inference + accuracy, offline sglang Engine), per (model, precision, split):
#   - 2b : {bf16,fp8} x {math,vision}                      (dense, no experts)
#   - 35b: {bf16,fp8} x {math,vision}, R3-capture ON       (one capture feeds replay AND no-replay KL)
#
# Stage 2 (forward-only KL, eager SDPA, no torch.compile, BF16 weights), per (model, precision, split):
#   - 2b : baseline, no replay
#   - 35b: replay AND no-replay, both reading the R3-captured rollouts (only --r3 + out-dir differ)
#
# Usage:
#   bash report/fp8_r3/run_matrix.sh                  # everything
#   STAGES=1 bash report/fp8_r3/run_matrix.sh         # only stage 1
#   STAGES=2 MODELS=2b bash report/fp8_r3/run_matrix.sh
#   ROOT=/path/to/run_data bash report/fp8_r3/run_matrix.sh   # override where run data lives
set -uo pipefail
# This script lives at report/fp8_r3/; cd up two levels to the repo root, since the Python stages
# are invoked by their repo-root-relative paths and read datasets/ relative to the CWD.
cd "$(dirname "$0")/../.."

STAGES="${STAGES:-1 2}"
MODELS="${MODELS:-2b 35b}"
SPLITS="${SPLITS:-math vision}"
# This script OWNS the on-disk layout: it builds every run-dir path and hands each stage an explicit
# --out-dir/--src-dir. The Python scripts hardcode no paths. Override the parent with ROOT=...
ROOT="${ROOT:-/opt/dlami/nvme/experiments/fp8_r3}"
PY="uv run python"
TORCHRUN="uv run torchrun --nproc_per_node=1"

# Run-dir naming convention: base case carries no suffix; an R3 run gets the `_r3` subscript.
run_dir() {  # model precision split r3(0/1) -> path under $ROOT
  local key="${1}_${2}_${3}"; [ "$4" = "1" ] && key="${key}_r3"
  echo "$ROOT/$key"
}

stage1() {  # model precision split r3(0/1) [extra flags...]
  local model="$1" prec="$2" split="$3" r3="$4"; shift 4
  local dir; dir="$(run_dir "$model" "$prec" "$split" "$r3")"
  if [ -s "$dir/records.jsonl" ]; then echo "[skip stage1] $(basename "$dir") (records exist)"; return; fi
  echo "[stage1] $(basename "$dir")"
  local r3flag=(); [ "$r3" = "1" ] && r3flag=(--r3)
  $PY report/fp8_r3/stage1_infer.py --out-dir "$dir" \
    --model "$model" --precision "$prec" --split "$split" "${r3flag[@]}" "$@"
}

stage2() {  # model precision split replay(0/1) src_dir
  local model="$1" prec="$2" split="$3" replay="$4" src_dir="$5"
  local dir; dir="$(run_dir "$model" "$prec" "$split" "$replay")"
  if [ -s "$dir/kl/summary.json" ]; then echo "[skip stage2] $(basename "$dir") (kl exists)"; return; fi
  echo "[stage2] $(basename "$dir")  (src=$(basename "$src_dir"))"
  local r3flag=(); [ "$replay" = "1" ] && r3flag=(--r3)
  $TORCHRUN report/fp8_r3/stage2_kl.py --src-dir "$src_dir" --out-dir "$dir" \
    --model "$model" --precision "$prec" --split "$split" "${r3flag[@]}"
}

for s in $STAGES; do
  if [ "$s" = "1" ]; then
    echo "########## STAGE 1: inference + accuracy ##########"
    for split in $SPLITS; do
      for prec in bf16 fp8; do
        for m in $MODELS; do
          if [ "$m" = "35b" ]; then
            stage1 35b "$prec" "$split" 1    # MoE: always capture experts (r3=1)
          else
            stage1 2b "$prec" "$split" 0
          fi
        done
      done
    done
  fi

  if [ "$s" = "2" ]; then
    echo "########## STAGE 2: forward-only KL ##########"
    for split in $SPLITS; do
      for prec in bf16 fp8; do
        for m in $MODELS; do
          if [ "$m" = "35b" ]; then
            # Both KL runs read the R3-captured stage-1 dir; only replay-on/off and out-dir differ.
            src="$(run_dir 35b "$prec" "$split" 1)"
            stage2 35b "$prec" "$split" 1 "$src"   # replay
            stage2 35b "$prec" "$split" 0 "$src"   # no replay (same source rollouts)
          else
            src="$(run_dir 2b "$prec" "$split" 0)"
            stage2 2b "$prec" "$split" 0 "$src"    # dense baseline
          fi
        done
      done
    done
  fi
done

echo "########## DONE. Summaries: ##########"
for d in "$ROOT"/*/; do
  [ -f "$d/meta.json" ] && printf "%-28s acc=%s\n" "$(basename "$d")" \
    "$($PY -c "import json;print(json.load(open('$d/meta.json')).get('accuracy'))" 2>/dev/null)"
  [ -f "$d/kl/summary.json" ] && printf "  KL %-25s tok_mean=%s seq_mean=%s\n" "$(basename "$d")" \
    "$($PY -c "import json;print(round(json.load(open('$d/kl/summary.json')).get('kl_k3_token_mean',0),4))" 2>/dev/null)" \
    "$($PY -c "import json;print(round(json.load(open('$d/kl/summary.json')).get('kl_k3_seq_mean',0),3))" 2>/dev/null)"
done
