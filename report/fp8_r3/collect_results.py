"""Aggregate stage-1 accuracy + stage-2 KL into per-task (Math / Geo3k) Markdown tables.

Enumerates the study's run dirs under --runs-dir and prints rows for fp8_r3_throughput_kl.md.

Args:
  --runs-dir   parent dir holding the per-run subdirs (named by run_matrix.sh's convention).
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ["2b", "4b", "9b", "35b"]
PRECS = ["bf16", "fp8"]
SPLITS = ["math", "vision"]  # math = Math (text), vision = Geo3k


def _run_dir(runs_dir, model, prec, split, r3):
    # Naming convention (shared with run_matrix.sh): base case has no suffix, R3 runs get `_r3`.
    return Path(runs_dir) / (f"{model}_{prec}_{split}" + ("_r3" if r3 else ""))


def _load_records(runs_dir, model, prec, split, r3):
    """Return list of dicts with keys: reward, truncated. None if run missing."""
    d = _run_dir(runs_dir, model, prec, split, r3)
    p = d / "records.jsonl"
    if not p.exists():
        return None
    records = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records.append({
            "example_idx": rec["example_idx"],
            "reward": float(rec.get("reward", 0.0)),
            "truncated": bool(rec["truncated"]),
        })
    return records or None


def _pass_at_k(records, k):
    """Unbiased pass@k estimator (Chen et al. 2021): E[1 - C(n-c,k)/C(n,k)]."""
    from math import comb
    by_prompt = {}
    for r in records:
        by_prompt.setdefault(r["example_idx"], []).append(r["reward"])
    scores = []
    for samples in by_prompt.values():
        n = len(samples)
        c = sum(1 for s in samples if s > 0)
        if n < k:
            continue
        scores.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(scores)) if scores else None


def _trunc_stats(records):
    """Return (acc_untruncated, truncation_rate) from loaded records."""
    rewards = np.array([r["reward"] for r in records])
    truncs = np.array([r["truncated"] for r in records], dtype=bool)
    keep = ~truncs
    acc_unt = float(rewards[keep].mean()) if keep.any() else None
    return acc_unt, float(truncs.mean())


def _kl(runs_dir, model, prec, split, r3):
    d = _run_dir(runs_dir, model, prec, split, r3)
    p = d / "kl" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=4):
    return "_tbd_" if x is None else f"{x:.{nd}f}"


def accuracy_tables(runs_dir):
    print("### Section 2 — Accuracy (per task)\n")
    print("pass@k = fraction of prompts solved by at least 1 of k samples (100 prompts × 4 samples, "
          "0/1 boxed-answer reward); acc(unt) = pass@1 over non-truncated sequences only; "
          "trunc = fraction length-truncated at 16k.\n")
    for split, label in [("math", "Math"), ("vision", "Geo3k")]:
        print(f"#### {label}")
        print("| Model | Prec | pass@1 | pass@2 | pass@4 | acc(unt) | trunc rate |")
        print("|-------|------|--------|--------|--------|----------|------------|")
        for model in MODELS:
            # 35b stage-1 always runs with r3 capture; 2b has no r3.
            r3 = model == "35b"
            for prec in PRECS:
                records = _load_records(runs_dir, model, prec, split, r3)
                if records is None:
                    print(f"| {model} | {prec.upper()} | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |")
                    continue
                p1 = _pass_at_k(records, 1)
                p2 = _pass_at_k(records, 2)
                p4 = _pass_at_k(records, 4)
                acc_unt, tr = _trunc_stats(records)
                print(f"| {model} | {prec.upper()} | {fmt(p1,3)} | {fmt(p2,3)} | {fmt(p4,3)} | {fmt(acc_unt,3)} | {fmt(tr,3)} |")
        print()


def kl_tables(runs_dir):
    print("### Section 3 — Rollout/training KL (per task)\n")
    # (label, model, precision, replay) rows
    rows = [
        ("2B", "2b", "bf16", False),
        ("2B", "2b", "fp8", False),
        ("4B", "4b", "bf16", False),
        ("4B", "4b", "fp8", False),
        ("9B", "9b", "bf16", False),
        ("9B", "9b", "fp8", False),
        ("35B", "35b", "bf16", False),
        ("35B", "35b", "fp8", False),
        ("35B R3", "35b", "bf16", True),
        ("35B R3", "35b", "fp8", True),
    ]
    for split, label in [("math", "Math"), ("vision", "Geo3k")]:
        print(f"#### {label}")
        # report mean (headline) + p99 (tail) for each K3 granularity (median ~0, std outlier-dominated).
        print("| Config | Precision | tok K3 mean | tok K3 p99 | seq K3 mean | seq K3 p99 |")
        print("|--------|-----------|-------------|------------|-------------|------------|")
        for name, model, prec, r3 in rows:
            s = _kl(runs_dir, model, prec, split, r3)
            if s is None:
                print(f"| {name} | {prec.upper()} | _tbd_ | _tbd_ | _tbd_ | _tbd_ |")
            else:
                print(
                    f"| {name} | {prec.upper()} | {fmt(s.get('kl_k3_token_mean'))} | "
                    f"{fmt(s.get('kl_k3_token_p99'))} | "
                    f"{fmt(s.get('kl_k3_seq_mean'))} | {fmt(s.get('kl_k3_seq_p99'))} |"
                )
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="parent dir holding the per-run subdirs")
    args = ap.parse_args()
    accuracy_tables(args.runs_dir)
    kl_tables(args.runs_dir)
