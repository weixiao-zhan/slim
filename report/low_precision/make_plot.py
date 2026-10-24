"""Render the three figures for README.md straight from the raw run data.

Reads (no hardcoded numbers):
  accuracy   <- $RUNS_ROOT/<run>/records.jsonl        (pass@k, trunc)
  KL         <- $RUNS_ROOT/<run>/kl/summary.json       (token/seq K3 mean)
  throughput <- $BENCH_DIR/result_<model>_<prec>.jsonl (sglang.bench_one_batch output)

Run-dir naming mirrors run_matrix.sh / collect_results.py: `<model>_<prec>_<split>` with a
`_r3` suffix for replay runs; 2_infer.py accuracy uses the r3-captured dir for MoE models.

Produces (next to this script):
  throughput.png  - prefill / decode tok/s vs concurrency (log2 axes; total on the left axis, per
                    user = total / batch size on the right axis) + decode speedup vs BF16,
                    one column per model
  accuracy.png    - pass@1 / pass@4 / trunc rate per task, bars grouped by model
  kl.png          - token / seq K3 mean per task, bars grouped by model config (MoE also with R3)

Shared style: color encodes precision (fixed categorical slots: BF16 blue, FP8 orange, MXFP8
aqua, NVFP4 yellow, validated for CVD separation); models are facets / x groups. Bars carry
their values as labels; the shared legend below each figure names the precisions.

Env overrides:
  RUNS_ROOT  parent of the per-run dirs (default /mnt/sagemaker-nvme/experiments/low_precision/runs)
  BENCH_DIR  dir with result_<tag>.jsonl (default /mnt/sagemaker-nvme/experiments/low_precision/throughput)

Usage:  python make_plot.py [throughput|accuracy|kl|all]
"""

import json
import os
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import common as C  # noqa: E402

EXP = Path("/mnt/sagemaker-nvme/experiments/low_precision")
ROOT = Path(os.environ.get("RUNS_ROOT", EXP / "runs"))
BENCH_DIR = Path(os.environ.get("BENCH_DIR", EXP / "throughput"))

CONC = [1, 2, 4, 8, 16, 32, 64]  # concurrency sweep (input=512, output=1024)
BENCH_IN, BENCH_OUT = 512, 1024  # bench workload these plots read

MODELS = list(C.MODELS)
MODEL_LABEL = {"9b": "9B", "27b": "27B", "35b": "35B-A3B"}
PRECS = list(C.PRECISIONS)
PREC_LABEL = {"bf16": "BF16", "fp8": "128 Block FP8", "mxfp8": "MXFP8", "nvfp4": "NVFP4"}
# Categorical slots 1-4 of the reference palette, in fixed order (validated: adjacent CVD dE >= 9.1).
COLOR = {"bf16": "#2a78d6", "fp8": "#eb6834", "mxfp8": "#1baf7a", "nvfp4": "#eda100"}
SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"


plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.8,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelcolor": INK_2,
    "font.size": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
})


def style_axes(ax, grid_axis="y", box=False):
    """Recessive dotted grid; open top/right spines unless `box`."""
    ax.spines[["top", "right"]].set_visible(box)
    ax.grid(True, axis=grid_axis, linestyle=":", linewidth=0.6, color="#d8d7d2", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, legend_extra=()):
    """Lay out, add the precision legend (+ `legend_extra` handles) below, write OUT/name."""
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    handles = [Patch(facecolor=COLOR[p], label=PREC_LABEL[p]) for p in PRECS] + list(legend_extra)
    fig.legend(handles=handles, ncol=len(handles), loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.5, columnspacing=2.0, fontsize=13)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ====================================================================================
# Raw-data readers
# ====================================================================================
def _run_dir(model, prec, split, r3):
    return ROOT / (f"{model}_{prec}_{split}" + ("_r3" if r3 else ""))


def _jsonl(p):
    return [json.loads(line) for line in p.open() if line.strip()] if p.exists() else []


def load_records(model, prec, split):
    """Return [{ex, reward, trunc}] for an accuracy run, or None if missing."""
    # 2_infer.py always captures r3 for MoE models; matches run_matrix.sh / collect_results.py.
    rows = _jsonl(_run_dir(model, prec, split, model in C.MOE_MODELS) / "records.jsonl")
    return [{"ex": r["example_idx"], "reward": float(r.get("reward", 0.0)), "trunc": bool(r["truncated"])}
            for r in rows] or None


def pass_at_k(records, k):
    """Unbiased pass@k (Chen et al. 2021): E[1 - C(n-c,k)/C(n,k)]."""
    by = {}
    for r in records:
        by.setdefault(r["ex"], []).append(r["reward"])
    scores = []
    for samples in by.values():
        n, c = len(samples), sum(1 for s in samples if s > 0)
        if n >= k:
            scores.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(scores)) if scores else None


def load_kl_summary(model, prec, split, r3):
    p = _run_dir(model, prec, split, r3) / "kl" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_throughput(model, prec):
    """{batch_size: (prefill_tok_s, decode_tok_s)} for the bench workload; last run wins."""
    return {r["batch_size"]: (r["prefill_throughput"], r["median_decode_throughput"])
            for r in _jsonl(BENCH_DIR / f"result_{model}_{prec}.jsonl")
            if r.get("input_len") == BENCH_IN and r.get("output_len") == BENCH_OUT}


# ====================================================================================
# Shared mark helpers
# ====================================================================================
def grouped_bars(ax, groups, values, fmt, label_size=9):
    """Bars per precision grouped on x. values: {(group, prec): v}; absent pairs leave a gap."""
    n = len(PRECS)
    slot = 0.8 / n
    x = np.arange(len(groups))
    for pi, prec in enumerate(PRECS):
        for gi, g in enumerate(groups):
            v = values.get((g, prec))
            if v is None:
                continue
            xb = x[gi] - 0.4 + slot * (pi + 0.5)
            # 2px surface gap between adjacent bars (edge in surface color).
            ax.bar(xb, v, slot, color=COLOR[prec], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(xb, v, fmt(v), ha="center", va="bottom", fontsize=label_size, rotation=90,
                    color=INK_2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(axis="x", length=0)


# ====================================================================================
# Throughput  (log2 tok/s vs concurrency with per-user tok/s on a right axis,
#              + decode speedup vs BF16)
# ====================================================================================
# Line styles: total = solid + circles, per user = dashed + squares (also the legend handles).
# Square side = circle diameter * sqrt(pi) / 2, so both markers have the same area.
SOLID = {"linestyle": "-", "linewidth": 2, "marker": "o", "markersize": 6}
DASHED = {"linestyle": "--", "linewidth": 1.5, "marker": "s", "markersize": 6 * np.pi ** 0.5 / 2}


def _line(ax, xs, ys, prec, style=SOLID):
    ax.plot(xs, ys, **style, color=COLOR[prec], markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)


def _log2_y(ax):
    """Log2 y axis with a tick at every power of two; K = 1024."""
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(LogLocator(base=2, numticks=30))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:g}K" if v >= 1024 else f"{v:g}"))


def plot_throughput():
    thru = {m: {p: load_throughput(m, p) for p in PRECS} for m in MODELS}
    models = [m for m in MODELS if any(thru[m].values())]
    if not models:
        print("no throughput results under", BENCH_DIR)
        return
    fig, axes = plt.subplots(3, len(models), figsize=(5.6 * len(models), 12.5), squeeze=False,
                             sharex=True, sharey="row")

    # Rows 0-1: total tok/s (solid, left axis) and tok/s/user = total / batch size (dashed, right
    # axis). Only the last column labels the right axis; the others keep its spine.
    for r, stage in enumerate(["Prefill", "Decode"]):
        rights = [ax.twinx() for ax in axes[r]]
        total, per_user = [], []
        for ax, right, m in zip(axes[r], rights, models, strict=True):
            right.sharey(rights[0])
            for prec, d in thru[m].items():
                xs = [b for b in CONC if b in d]
                ys = [d[b][r] for b in xs]
                ys_user = [y / b for y, b in zip(ys, xs, strict=True)]
                _line(ax, xs, ys, prec)
                _line(right, xs, ys_user, prec, DASHED)
                total += ys
                per_user += ys_user
            _log2_y(ax)
            _log2_y(right)
            if right is not rights[-1]:
                right.tick_params(axis="y", which="both", right=False, labelright=False)
        axes[r, 0].set_ylabel(f"{stage} tok/s (solid)")
        rights[-1].set_ylabel(f"{stage} tok/s/user (dashed)")
        # Right axis = left axis / 2^n: a whole-octave offset puts the power-of-two ticks of both
        # axes on the same grid lines. Pick n that needs the smallest common span.
        lo_l, hi_l = np.log2(min(total)), np.log2(max(total))
        lo_r, hi_r = np.log2(min(per_user)), np.log2(max(per_user))
        n = min(range(-12, 13), key=lambda k: max(hi_l, hi_r + k) - min(lo_l, lo_r + k))
        lo, hi = min(lo_l, lo_r + n), max(hi_l, hi_r + n)
        pad = 0.05 * (hi - lo)
        axes[r, 0].set_ylim(2 ** (lo - pad), 2 ** (hi + pad))
        rights[0].set_ylim(2 ** (lo - pad - n), 2 ** (hi + pad - n))

    # Row 2: decode speedup vs BF16 at the same batch size.
    for ax, m in zip(axes[2], models, strict=True):
        base = thru[m]["bf16"]
        for prec, d in thru[m].items():
            xs = [b for b in CONC if b in d and b in base]
            _line(ax, xs, [d[b][1] / base[b][1] for b in xs], prec)
    axes[2, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}x"))
    axes[2, 0].set_ylabel("Decode speedup vs BF16")

    # Shared x: log2 batch size, half a log2 step past each end; tick labels on every row.
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(CONC, [str(b) for b in CONC])
    axes[0, 0].xaxis.set_minor_locator(NullLocator())
    axes[0, 0].set_xlim(2 ** -0.5, 64 * 2 ** 0.5)
    for (_, c), ax in np.ndenumerate(axes):
        style_axes(ax, grid_axis="both", box=True)
        ax.tick_params(axis="x", labelbottom=True)
        if c:  # rows share the grid, so inner columns keep the left spine only
            ax.tick_params(axis="y", which="both", left=False)
    for ax, m in zip(axes[0], models, strict=True):
        ax.set_title(MODEL_LABEL[m])
    for ax in axes[-1]:
        ax.set_xlabel("concurrency (batch size)")
    save(fig, "throughput.png", [
        Line2D([], [], **SOLID, color=INK_2, label="total (left axis)"),
        Line2D([], [], **DASHED, color=INK_2, label="per user (right axis)"),
    ])


# ====================================================================================
# Accuracy  (pass@1 / pass@4 / trunc rate, bars grouped by model)
# ====================================================================================
ACC_METRICS = [("pass@1", lambda r: pass_at_k(r, 1)), ("pass@4", lambda r: pass_at_k(r, 4)),
               ("trunc rate", lambda r: float(np.mean([x["trunc"] for x in r])))]
TASKS = [("math", "Math (text)"), ("vision", "Geo3k (vision)")]


def plot_accuracy():
    fig, axes = plt.subplots(len(TASKS), len(ACC_METRICS), figsize=(16, 9), squeeze=False)
    groups = [MODEL_LABEL[m] for m in MODELS]
    for r, (split, task) in enumerate(TASKS):
        records = {(m, p): load_records(m, p, split) for m in MODELS for p in PRECS}
        for c, (metric, fn) in enumerate(ACC_METRICS):
            ax = axes[r, c]
            values = {(MODEL_LABEL[m], p): fn(rec) for (m, p), rec in records.items() if rec}
            grouped_bars(ax, groups, values, lambda v: f"{v * 100:.0f}%")
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
            ax.set_title(f"{task} - {metric}")
            style_axes(ax)
    save(fig, "accuracy.png")


# ====================================================================================
# TIM (K3 KL) - bars grouped by model config, log y shared per statistic row
# ====================================================================================
KL_GRAN = [("token K3 (mean)", "kl_k3_token_mean"), ("seq K3 (mean)", "kl_k3_seq_mean")]
# x groups: every model without replay, plus each MoE model with R3.
KL_GROUPS = [(MODEL_LABEL[m], m, False) for m in MODELS] + \
            [(f"{MODEL_LABEL[m]} R3", m, True) for m in MODELS if m in C.MOE_MODELS]


def plot_kl():
    fig, axes = plt.subplots(len(KL_GRAN), len(TASKS), figsize=(14, 9), squeeze=False, sharey="row")
    for r, (gtitle, gkey) in enumerate(KL_GRAN):
        row_values = []
        for c, (split, task) in enumerate(TASKS):
            ax = axes[r, c]
            values = {}
            for label, m, r3 in KL_GROUPS:
                for p in PRECS:
                    s = load_kl_summary(m, p, split, r3)
                    if s is not None and s.get(gkey) is not None:
                        values[(label, p)] = float(s[gkey])
            grouped_bars(ax, [g for g, _, _ in KL_GROUPS], values, lambda v: f"{v:.2g}")
            row_values += values.values()
            if r == 0:
                ax.set_title(task)
            if c == 0:
                ax.set_ylabel(gtitle)
            style_axes(ax)
        # K3 spans ~4 decades across formats: log scale keeps the 8-bit bars readable. The row
        # shares y, so its limits cover both tasks.
        axes[r, 0].set_yscale("log")
        axes[r, 0].set_ylim(min(row_values) / 3, max(row_values) * 10)
    save(fig, "kl.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "throughput"):
        plot_throughput()
    if which in ("all", "accuracy"):
        plot_accuracy()
    if which in ("all", "kl"):
        plot_kl()
