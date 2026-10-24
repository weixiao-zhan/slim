#!/usr/bin/env bash
# Drive the accuracy (2_infer.py) and TIM KL (3_kl.py) experiment matrix over every
# (model, precision) in common.py's registry. Throughput is a separate sweep: 1_bench_throughput.sh.
# Run 0_prepare_models.sh first to generate the quantized checkpoints.
#
# Idempotent / resumable: each step skips if its output already exists (records.jsonl for step 2,
# kl/summary.json for step 3). Delete a run dir under $ROOT (below) to redo it.
#
# Step 2 (2_infer.py: inference + accuracy, offline sglang Engine), per (model, precision, split):
#   - dense (9b, 27b) : no experts
#   - MoE (35b)       : R3-capture ON (one capture feeds replay AND no-replay KL)
# Step 3 (3_kl.py: forward-only KL, eager SDPA, no torch.compile, BF16 weights), per (model, precision, split):
#   - dense : baseline, no replay
#   - MoE   : replay AND no-replay, both reading the R3-captured rollouts (only --r3 + out-dir differ)
# (model, precision) pairs sglang can't serve on SM120 (MXFP8 MoE) are skipped.
#
# Usage:
#   bash report/low_precision/run_matrix.sh                       # everything
#   STEPS=2 bash report/low_precision/run_matrix.sh               # only step 2
#   STEPS=3 MODELS=9b PRECISIONS="bf16 nvfp4" bash report/low_precision/run_matrix.sh
#   ROOT=/path/to/run_data bash report/low_precision/run_matrix.sh   # override where run data lives
set -uo pipefail
# This script lives at report/low_precision/; cd up two levels to the repo root, since the Python
# steps are invoked by their repo-root-relative paths and read datasets/ relative to the CWD.
cd "$(dirname "$0")/../.."

PY="uv run python"
TORCHRUN="uv run torchrun --nproc_per_node=1"
REG=".venv/bin/python report/low_precision/common.py"
STEPS="${STEPS:-2 3}"
MODELS="${MODELS:-$($REG models)}"
PRECISIONS="${PRECISIONS:-$($REG precisions)}"
SPLITS="${SPLITS:-math vision}"
# This script OWNS the on-disk layout: it builds every run-dir path and hands each step an explicit
# --out-dir/--src-dir. The Python scripts hardcode no paths. Override the parent with ROOT=...
ROOT="${ROOT:-/mnt/sagemaker-nvme/experiments/low_precision/runs}"

# Run-dir naming convention: base case carries no suffix; an R3 run gets the `_r3` subscript.
run_dir() {  # model precision split r3(0/1) -> path under $ROOT
  local key="${1}_${2}_${3}"; [ "$4" = "1" ] && key="${key}_r3"
  echo "$ROOT/$key"
}

is_moe() { $REG moe "$1"; }

infer() {  # model precision split r3(0/1) [extra flags...]
  local model="$1" prec="$2" split="$3" r3="$4"; shift 4
  local dir; dir="$(run_dir "$model" "$prec" "$split" "$r3")"
  if [ -s "$dir/records.jsonl" ]; then echo "[skip infer] $(basename "$dir") (records exist)"; return; fi
  echo "[infer] $(basename "$dir")"
  local r3flag=(); [ "$r3" = "1" ] && r3flag=(--r3)
  $PY report/low_precision/2_infer.py --out-dir "$dir" \
    --model "$model" --precision "$prec" --split "$split" "${r3flag[@]}" "$@"
}

kl() {  # model precision split replay(0/1) src_dir
  local model="$1" prec="$2" split="$3" replay="$4" src_dir="$5"
  local dir; dir="$(run_dir "$model" "$prec" "$split" "$replay")"
  if [ -s "$dir/kl/summary.json" ]; then echo "[skip kl] $(basename "$dir") (kl exists)"; return; fi
  echo "[kl] $(basename "$dir")  (src=$(basename "$src_dir"))"
  local r3flag=(); [ "$replay" = "1" ] && r3flag=(--r3)
  $TORCHRUN report/low_precision/3_kl.py --src-dir "$src_dir" --out-dir "$dir" \
    --model "$model" --precision "$prec" --split "$split" "${r3flag[@]}"
}

for s in $STEPS; do
  if [ "$s" = "2" ]; then
    echo "########## STEP 2: inference + accuracy ##########"
    for split in $SPLITS; do
      for prec in $PRECISIONS; do
        for m in $MODELS; do
          if ! $REG supported "$m" "$prec"; then echo "[skip] ${m}_${prec} (unsupported on SM120)"; continue; fi
          if is_moe "$m"; then
            infer "$m" "$prec" "$split" 1    # MoE: always capture experts (r3=1)
          else
            infer "$m" "$prec" "$split" 0    # dense: no experts
          fi
        done
      done
    done
  fi

  if [ "$s" = "3" ]; then
    echo "########## STEP 3: forward-only KL ##########"
    for split in $SPLITS; do
      for prec in $PRECISIONS; do
        for m in $MODELS; do
          if ! $REG supported "$m" "$prec"; then continue; fi
          if is_moe "$m"; then
            # Both KL runs read the R3-captured 2_infer.py dir; only replay-on/off and out-dir differ.
            src="$(run_dir "$m" "$prec" "$split" 1)"
            kl "$m" "$prec" "$split" 1 "$src"   # replay
            kl "$m" "$prec" "$split" 0 "$src"   # no replay (same source rollouts)
          else
            src="$(run_dir "$m" "$prec" "$split" 0)"
            kl "$m" "$prec" "$split" 0 "$src"   # dense baseline
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
