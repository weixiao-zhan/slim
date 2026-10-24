"""Aggregate throughput, accuracy (2_infer.py), and KL (3_kl.py) into Markdown tables for README.md.

Enumerates the study's run dirs under --runs-dir (and bench results under --bench-dir) over every
(model, precision) in common.py's registry.

Args:
  --runs-dir   parent dir holding the per-run subdirs (named by run_matrix.sh's convention).
  --bench-dir  dir with 1_bench_throughput.sh's result_<model>_<precision>.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import common as C

MODELS = list(C.MODELS)
PRECS = list(C.PRECISIONS)
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
            # MoE 2_infer.py always runs with r3 capture; dense models have no r3.
            r3 = model in C.MOE_MODELS
            for prec in PRECS:
                if not C.is_supported(model, prec):
                    continue
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
    print("### Section 3 — TIM KL (per task)\n")
    # (label, model, precision, replay) rows; MoE models also get an R3 (replay) row per precision.
    configs = [(m.upper(), m, False) for m in MODELS] + [(f"{m.upper()} R3", m, True) for m in MODELS if m in C.MOE_MODELS]
    rows = [(name, m, p, r3) for name, m, r3 in configs for p in PRECS if C.is_supported(m, p)]
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


def _throughput(bench_dir, model, prec):
    """{batch_size: (prefill tok/s, decode tok/s)}; last run wins."""
    p = Path(bench_dir) / f"result_{model}_{prec}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[r["batch_size"]] = (r["prefill_throughput"], r["median_decode_throughput"])
    return out


def throughput_tables(bench_dir):
    print("### Section 1 — Throughput (input=512, output=1024)\n")
    for idx, label in [(1, "Decode"), (0, "Prefill")]:
        print(f"#### {label} tok/s (x = speedup vs BF16)")
        for model in MODELS:
            thru = {p: _throughput(bench_dir, model, p) for p in PRECS if C.is_supported(model, p)}
            batches = sorted({b for d in thru.values() for b in d})
            if not batches:
                continue
            print(f"\n{model.upper()}\n")
            print("| Prec | " + " | ".join(f"bs={b}" for b in batches) + " |")
            print("|------|" + "|".join("------" for _ in batches) + "|")
            base = thru.get("bf16", {})
            for prec, d in thru.items():
                cells = []
                for b in batches:
                    if b not in d:
                        cells.append("—")
                    elif prec != "bf16" and b in base:
                        cells.append(f"{d[b][idx]:.0f} ({d[b][idx] / base[b][idx]:.2f}x)")
                    else:
                        cells.append(f"{d[b][idx]:.0f}")
                print(f"| {prec.upper()} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/mnt/sagemaker-nvme/experiments/low_precision/runs",
                    help="parent dir holding the per-run subdirs")
    ap.add_argument("--bench-dir", default="/mnt/sagemaker-nvme/experiments/low_precision/throughput",
                    help="dir with 1_bench_throughput.sh results")
    args = ap.parse_args()
    throughput_tables(args.bench_dir)
    accuracy_tables(args.runs_dir)
    kl_tables(args.runs_dir)
