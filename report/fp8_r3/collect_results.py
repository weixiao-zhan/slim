"""Aggregate stage-1 accuracy + stage-2 KL into per-task (Math / Geo3k) Markdown tables.

Enumerates the study's run dirs under --runs-dir and prints rows for fp8_r3_throughput_kl.md.

Args:
  --runs-dir   parent dir holding the per-run subdirs (named by run_matrix.sh's convention).
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ["2b", "35b"]
PRECS = ["bf16", "fp8"]
SPLITS = ["math", "vision"]  # math = Math (text), vision = Geo3k
EOS_ID = 248046  # Qwen3.5 <|im_end|>


def _run_dir(runs_dir, model, prec, split, r3):
    # The run-dir naming convention (shared with run_matrix.sh): base case carries no suffix, an R3
    # run gets the `_r3` subscript. This aggregator is layout-aware by nature (it enumerates runs).
    return Path(runs_dir) / (f"{model}_{prec}_{split}" + ("_r3" if r3 else ""))


def _trunc_stats(runs_dir, model, prec, split, r3):
    """Return (accuracy, accuracy_untruncated, truncation_rate) for a run, or (None,)*3.

    Prefers the per-record `truncated`/`finish_type` field (newer runs); falls back to EOS-token
    detection (last token == <|im_end|>) for older records that predate that field.
    """
    d = _run_dir(runs_dir, model, prec, split, r3)
    p = d / "records.jsonl"
    if not p.exists():
        return None, None, None
    rewards, truncs = [], []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rewards.append(float(rec.get("reward", 0.0)))
        if "truncated" in rec:
            truncs.append(bool(rec["truncated"]))
        else:
            # fallback: a naturally-finished sequence ends with the EOS token
            truncs.append(rec["tokens"][-1] != EOS_ID)
    if not rewards:
        return None, None, None
    rewards = np.array(rewards)
    truncs = np.array(truncs, dtype=bool)
    acc = float(rewards.mean())
    keep = ~truncs
    acc_unt = float(rewards[keep].mean()) if keep.any() else None
    return acc, acc_unt, float(truncs.mean())


def _kl(runs_dir, model, prec, split, r3):
    d = _run_dir(runs_dir, model, prec, split, r3)
    p = d / "kl" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=4):
    return "_tbd_" if x is None else f"{x:.{nd}f}"


def accuracy_tables(runs_dir):
    print("### Section 2 — Accuracy (per task)\n")
    print("acc = raw accuracy over all 400 samples; acc(unt) = accuracy over only sequences that "
          "ended naturally (not length-truncated at 16k); trunc = fraction length-truncated.\n")
    for split, label in [("math", "Math"), ("vision", "Geo3k")]:
        print(f"#### {label}")
        print("| Model | Prec | acc | acc(unt) | trunc rate |")
        print("|-------|------|-----|----------|------------|")
        for model in MODELS:
            # 35b stage-1 always runs with r3 capture; 2b has no r3.
            r3 = model == "35b"
            for prec in PRECS:
                acc, acc_unt, tr = _trunc_stats(runs_dir, model, prec, split, r3)
                print(f"| {model} | {prec.upper()} | {fmt(acc,3)} | {fmt(acc_unt,3)} | {fmt(tr,3)} |")
        print()


def kl_tables(runs_dir):
    print("### Section 3 — Rollout/training KL (per task)\n")
    # (label, model, precision, replay) rows
    rows = [
        ("2B baseline", "2b", "bf16", False),
        ("2B baseline", "2b", "fp8", False),
        ("35B", "35b", "bf16", False),
        ("35B", "35b", "fp8", False),
        ("35B R3", "35b", "bf16", True),
        ("35B R3", "35b", "fp8", True),
    ]
    for split, label in [("math", "Math"), ("vision", "Geo3k")]:
        print(f"#### {label}")
        print("| Config | Precision | per-token K3 mean | median | p99 | seq K3 mean | median |")
        print("|--------|-----------|-------------------|--------|-----|-------------|--------|")
        for name, model, prec, r3 in rows:
            s = _kl(runs_dir, model, prec, split, r3)
            if s is None:
                print(f"| {name} | {prec.upper()} | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |")
            else:
                print(
                    f"| {name} | {prec.upper()} | {fmt(s.get('kl_k3_token_mean'))} | "
                    f"{fmt(s.get('kl_k3_token_median'))} | {fmt(s.get('kl_k3_token_p99'))} | "
                    f"{fmt(s.get('kl_k3_seq_mean'),3)} | {fmt(s.get('kl_k3_seq_median'),3)} |"
                )
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="parent dir holding the per-run subdirs")
    args = ap.parse_args()
    accuracy_tables(args.runs_dir)
    kl_tables(args.runs_dir)
