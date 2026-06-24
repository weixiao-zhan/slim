"""Render the three figures for fp8_r3_throughput_kl.md straight from the raw run data.

Reads (no hardcoded numbers):
  accuracy   <- $FP8_R3_ROOT/<run>/records.jsonl        (pass@k, acc-untrunc, trunc)
  KL         <- $FP8_R3_ROOT/<run>/kl/summary.json       (token/seq K3 mean & p99)
  throughput <- $BENCH_DIR/result_<model>_<prec>.jsonl   (sglang.bench_one_batch output)

Run-dir naming mirrors run_matrix.sh / collect_results.py: `<model>_<prec>_<split>` with a
`_r3` suffix for replay runs; stage-1 accuracy uses the r3-captured dir for 35b (none for 2b).

Produces (next to this script):
  throughput.png  - prefill/decode tok/s vs concurrency, log-log axes
  accuracy.png    - pass@k curves (k=1..4) + acc(untrunc)/trunc bars per task
  kl.png          - token/seq K3 mean & p99 per task, linear y, shared y-lim per statistic

Shared style: hue encodes the model (2B = blue, 9B = green, 35B = orange); shade encodes
precision (deep = BF16, light = FP8). R3 reuses the 35B hue with a hatch overlay.

Env overrides:
  FP8_R3_ROOT  parent of the per-run dirs (default /opt/dlami/nvme/experiments/fp8_r3)
  BENCH_DIR    dir with result_<tag>.jsonl (default <repo>/outputs/fp8_vs_bf16_bench)

Usage:  python make_plot.py [throughput|accuracy|kl|all]
"""

import json
import os
import sys
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ROOT = Path(os.environ.get("FP8_R3_ROOT", "/opt/dlami/nvme/experiments/fp8_r3"))
BENCH_DIR = Path(os.environ.get("BENCH_DIR", REPO / "outputs" / "fp8_vs_bf16_bench"))

CONC = [1, 2, 4, 8, 16, 32, 64]  # concurrency sweep (input=512, output=1024)
KS = [1, 2, 3, 4]      # pass@k
BENCH_IN, BENCH_OUT = 512, 1024  # bench workload these plots read

# ---- palette: <model>_<precision> -> hex (deep = BF16, light = FP8) -------------------
# hue = model (2B blue, 9B green, 35B orange); shade = precision (deep BF16, light FP8).
C = {
    "2b_bf16": "#1f4e79",
    "2b_fp8": "#9ecae1",
    "9b_bf16": "#1b6b3a",
    "9b_fp8": "#a1d99b",
    "35b_bf16": "#b35900",
    "35b_fp8": "#fdbe85",
}
# accuracy / throughput series: (palette key, legend label)
SERIES = [("2b_bf16", "2B BF16"), ("2b_fp8", "2B FP8"),
          ("9b_bf16", "9B BF16"), ("9b_fp8", "9B FP8"),
          ("35b_bf16", "35B BF16"), ("35b_fp8", "35B FP8")]


def shade(color, f):
    """Multiply RGB by f (<1 darkens) for marker rings / hatch edges."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r * f, g * f, b * f)


def base_style():
    """Larger, more legible labels/legend than mpl defaults (shared by all three figures)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "font.size": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def style_axes(ax, grid_axis="y"):
    """Clean spines + light dotted grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle=":", linewidth=0.7, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)


# Shared legend placement/style for all three figures (ncol varies — KL has the extra R3 entries —
# but position, font, and spacing are uniform). Each plot reserves bottom margin via subplots_adjust.
LEGEND_Y = 0.0
LEGEND_FONTSIZE = 15


def legend_below(fig, handles, ncol, y=LEGEND_Y):
    fig.legend(handles=handles, ncol=ncol, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=1.5, columnspacing=2.0,
               fontsize=LEGEND_FONTSIZE)


# ====================================================================================
# Raw-data readers
# ====================================================================================
def _run_dir(model, prec, split, r3):
    return ROOT / (f"{model}_{prec}_{split}" + ("_r3" if r3 else ""))


def _acc_r3(model):
    # stage-1 captures r3 for 35b (no r3 variant for 2b); matches collect_results.py.
    return model == "35b"


def load_records(model, prec, split):
    """Return [{example_idx, reward, truncated}] for an accuracy run, or None if missing."""
    p = _run_dir(model, prec, split, _acc_r3(model)) / "records.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append({"ex": r["example_idx"], "reward": float(r.get("reward", 0.0)),
                    "trunc": bool(r["truncated"])})
    return out or None


def pass_at_k(records, k):
    """Unbiased pass@k (Chen et al. 2021): E[1 - C(n-c,k)/C(n,k)]."""
    by = {}
    for r in records:
        by.setdefault(r["ex"], []).append(r["reward"])
    scores = []
    for samples in by.values():
        n = len(samples)
        c = sum(1 for s in samples if s > 0)
        if n < k:
            continue
        scores.append(1.0 - comb(n - c, k) / comb(n, k))
    return float(np.mean(scores)) if scores else None


def acc_stats(records):
    """(acc over non-truncated, truncation rate)."""
    rew = np.array([r["reward"] for r in records])
    tr = np.array([r["trunc"] for r in records], dtype=bool)
    keep = ~tr
    return (float(rew[keep].mean()) if keep.any() else None), float(tr.mean())


def load_accuracy(split):
    """{palette_key: dict(passk=[k1..k4], acc=, trunc=)} for one split."""
    data = {}
    for model in ("2b", "9b", "35b"):
        for prec in ("bf16", "fp8"):
            recs = load_records(model, prec, split)
            if recs is None:
                continue
            acc, tr = acc_stats(recs)
            data[f"{model}_{prec}"] = dict(passk=[pass_at_k(recs, k) for k in KS], acc=acc, trunc=tr)
    return data


def load_kl_summary(model, prec, split, r3):
    p = _run_dir(model, prec, split, r3) / "kl" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_throughput(model, prec):
    """{batch_size: (prefill_tok_s, decode_tok_s)} for the bench workload; last run wins."""
    p = BENCH_DIR / f"result_{model}_{prec}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("input_len") != BENCH_IN or r.get("output_len") != BENCH_OUT:
            continue
        out[r["batch_size"]] = (r["prefill_throughput"], r["median_decode_throughput"])
    return out


# ====================================================================================
# Accuracy  (pass@1..4 curves + acc/trunc bars)
# ====================================================================================
def plot_accuracy():
    base_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.4))  # width unchanged (was 18)
    panels = [("math", "Math (text)"), ("vision", "Geo3k (vision)")]
    xk = np.arange(len(KS))

    for col, (split, title) in enumerate(panels):
        data = load_accuracy(split)
        ax_line, ax_bar = axes[col * 2], axes[col * 2 + 1]

        # --- pass@k curves ---
        for key, _ in SERIES:
            if key not in data:
                continue
            ax_line.plot(xk, data[key]["passk"], marker="o", markersize=8, linewidth=2.4,
                         color=C[key], markerfacecolor=C[key], markeredgecolor=shade(C[key], 0.7),
                         markeredgewidth=1.0, zorder=3)
        ax_line.set_title(title)
        # x ticks self-label as pass@1..4, so no separate xlabel (which would collide with the legend).
        ax_line.set_xticks(xk)
        ax_line.set_xticklabels([f"pass@{k}" for k in KS])
        ax_line.set_ylim(0, 1.0)
        ax_line.set_yticks(np.arange(0, 1.01, 0.2))
        style_axes(ax_line)

        # --- acc(untrunc) / trunc rate bars ---
        groups = ["acc(unt)", "trunc rate"]
        x = np.arange(len(groups))
        keys = [k for k, _ in SERIES if k in data]
        n = len(keys)
        w = 0.8 / max(n, 1)
        for i, key in enumerate(keys):
            vals = [data[key]["acc"], data[key]["trunc"]]
            offs = x - 0.4 + w * (i + 0.5)
            bars = ax_bar.bar(offs, vals, w * 0.92, color=C[key], zorder=3)
            for b, v in zip(bars, vals):
                ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=10)
        ax_bar.set_title(title)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(groups)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_yticks(np.arange(0, 1.01, 0.2))
        style_axes(ax_bar)

    handles = [Patch(facecolor=C[k], label=lab) for k, lab in SERIES]
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.subplots_adjust(bottom=0.14)
    legend_below(fig, handles, ncol=6)
    fig.savefig(OUT / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "accuracy.png")


# ====================================================================================
# Throughput  (log-log: tok/s vs concurrency)
# ====================================================================================
def plot_throughput():
    base_style()
    # Figure width is matched to the two ~square equal-aspect boxes (each ~0.8 w:h); a wider figure
    # would shrink the boxes and leave a large central gap.
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.8), gridspec_kw={"wspace": 0.16})
    thru = {key: load_throughput(*key.split("_", 1)) for key, _ in SERIES}

    # Shared x window (both panels span the same concurrency decades).
    xpad = 0.06
    xlo = min(CONC) / 10 ** xpad
    xhi = max(CONC) * 10 ** xpad
    x_dec = np.log10(xhi / xlo)

    # Per-panel data y-range and one common y decade-span (the larger), so both panels get identical
    # x- and y-decade spans -> identical equal-aspect boxes.
    yranges = {}
    for idx, title in [(0, "Prefill"), (1, "Decode")]:
        vals = [d[c][idx] for key, _ in SERIES for d in [thru[key]] for c in CONC if c in d]
        yranges[title] = (min(vals), max(vals))
    ypad = 0.08
    target_dec = max(np.log10(hi / lo) for lo, hi in yranges.values()) + 2 * ypad

    for ax, (idx, title) in zip(axes, [(0, "Prefill"), (1, "Decode")]):
        for key, _ in SERIES:
            d = thru[key]
            xs = [c for c in CONC if c in d]
            ys = [d[c][idx] for c in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6, color=C[key],
                    markerfacecolor=C[key], markeredgecolor=shade(C[key], 0.7),
                    markeredgewidth=0.8, zorder=3)
        ax.set_title(title)
        # No x-label: the powers-of-2 concurrency ticks are self-explanatory. y-label only on the left.
        if idx == 0:
            ax.set_ylabel("K tok/s")
        # Same log base on both axes so equal aspect gives 1:1 decade-per-decade pixels (slope-1 ==
        # 45 degrees). x ticks stay at the powers-of-2 concurrencies.
        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_xticks(CONC)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        # denser y ticks: 1-2-5 within each decade. Positions stay in tok/s; labels show K tok/s.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.tick_params(axis="x", which="minor", bottom=False)

        # Shared x window; y window centered on this panel's data, stretched to the common decade span.
        ax.set_xlim(xlo, xhi)
        lo, hi = yranges[title]
        center = np.sqrt(lo * hi)
        ax.set_ylim(center / 10 ** (target_dec / 2), center * 10 ** (target_dec / 2))

        style_axes(ax, grid_axis="both")
        # 1:1 decade-per-decade pixels -> slope-1 (linear scaling) renders at 45 degrees; faint guide.
        ax.set_aspect("equal", adjustable="box")
        y0 = ax.get_ylim()[0]
        ax.plot([xlo, xhi], [y0, y0 * 10 ** x_dec], linestyle=(0, (4, 4)), linewidth=1.2,
                color="#999999", zorder=1)

    # ncol=3 wraps the 6 series into two rows: BF16 on top, FP8 below, one model per column.
    handles = [Patch(facecolor=C[k], label=lab) for k, lab in SERIES]
    fig.subplots_adjust(bottom=0.24)
    legend_below(fig, handles, ncol=3, y=0.0)
    fig.savefig(OUT / "throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "throughput.png")


# ====================================================================================
# Rollout-train KL (K3) — linear y, shared y-lim per statistic across the two tasks
# ====================================================================================
# (palette key, legend label, face color, hatch) — R3 = 35B hue + hatch overlay.
KL_SERIES = [
    ("2b_bf16", "2B BF16", C["2b_bf16"], None, "2b", "bf16", False),
    ("2b_fp8", "2B FP8", C["2b_fp8"], None, "2b", "fp8", False),
    ("9b_bf16", "9B BF16", C["9b_bf16"], None, "9b", "bf16", False),
    ("9b_fp8", "9B FP8", C["9b_fp8"], None, "9b", "fp8", False),
    ("35b_bf16", "35B BF16", C["35b_bf16"], None, "35b", "bf16", False),
    ("35b_fp8", "35B FP8", C["35b_fp8"], None, "35b", "fp8", False),
    ("35b_r3_bf16", "35B R3 BF16", C["35b_bf16"], "///", "35b", "bf16", True),
    ("35b_r3_fp8", "35B R3 FP8", C["35b_fp8"], "///", "35b", "fp8", True),
]
# rows: (granularity title, summary.json key). y-ticks are derived from the data (see _nice_ticks).
KL_GRAN = [
    ("token K3 (mean)", "kl_k3_token_mean"),
    ("seq K3 (mean)", "kl_k3_seq_mean"),
]
KL_TASKS = [("math", "Math (text)"), ("vision", "Geo3k (vision)")]
# x clusters (model config), each holding the two precision bars (BF16 deep, FP8 light).
KL_CLUSTERS = [("2B", "2b", False), ("9B", "9b", False), ("35B", "35b", False), ("35B R3", "35b", True)]
KL_PRECS = ["bf16", "fp8"]


def _kfmt(v):
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _nice_ticks(top):
    """Evenly spaced ticks from 0 up to at least `top`, with a 1/2/5 x 10^n step."""
    import math
    raw = top / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    ticks, t = [], 0.0
    while t < top + step * 0.5:
        ticks.append(round(t, 10))
        t += step
    return ticks


def plot_kl():
    base_style()
    # rows = granularity (token K3 / seq K3), cols = task (Math / Geo3k). Each granularity ROW
    # shares one y-lim across both task columns (no log), so the token K3 of Math vs Geo3k sit on
    # the same scale and the cross-task / cross-precision token difference is read directly.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # pull every value first so we can fix per-row y-lims.
    vals = {}  # (gran_key, split, series_key) -> value
    for split, _ in KL_TASKS:
        for _, _, _, _, model, prec, r3 in KL_SERIES:
            s = load_kl_summary(model, prec, split, r3)
            if s is None:
                continue
            skey = f"{model}{'_r3' if r3 else ''}_{prec}"
            for _, gkey in KL_GRAN:
                if s.get(gkey) is not None:
                    vals[(gkey, split, skey)] = float(s[gkey])

    # Per-row ticks derived from the data max (so the axis always extends above the tallest bar, incl.
    # the higher-divergence 9B), then y-lim set above the top tick to leave headroom for value labels.
    row_ticks, row_top = {}, {}
    for _, gkey in KL_GRAN:
        gvals = [v for (gk, sp, sk), v in vals.items() if gk == gkey]
        gmax = max(gvals) if gvals else 1.0
        row_ticks[gkey] = _nice_ticks(gmax)
        row_top[gkey] = max(row_ticks[gkey][-1], gmax) * 1.12

    # one cluster per model config; the two precision bars sit side-by-side within it.
    # slot/width/step chosen so inter-cluster gap : cluster width ~= accuracy.png's 0.27.
    slot = 0.8 / 4          # per-bar slot, matches accuracy's w
    bw = slot * 0.92        # drawn bar width, matches accuracy
    step = 2.45 * slot      # cluster center-to-center spacing
    xc = np.arange(len(KL_CLUSTERS)) * step  # cluster centers
    for r, (gtitle, gkey) in enumerate(KL_GRAN):
        gticks = row_ticks[gkey]
        for c, (split, task_label) in enumerate(KL_TASKS):
            ax = axes[r, c]
            for ci, (clabel, model, r3) in enumerate(KL_CLUSTERS):
                for pi, prec in enumerate(KL_PRECS):
                    skey = f"{model}{'_r3' if r3 else ''}_{prec}"
                    v = vals.get((gkey, split, skey))
                    if v is None:
                        continue
                    color = C[f"{model}_{prec}"]
                    hatch = "///" if r3 else None
                    xb = xc[ci] + (pi - 0.5) * slot
                    ax.bar(xb, v, bw, color=color, hatch=hatch, zorder=3,
                           edgecolor=shade(color, 0.5) if hatch else "none",
                           linewidth=0.7 if hatch else 0.0)
                    ax.text(xb, v + row_top[gkey] * 0.012, _kfmt(v),
                            ha="center", va="bottom", fontsize=10, rotation=90)
            ax.set_ylim(0, row_top[gkey])
            ax.set_yticks(gticks)
            ax.set_xticks(xc)
            ax.set_xticklabels([cl for cl, _, _ in KL_CLUSTERS])
            ax.tick_params(axis="x", length=0)
            if r == 0:
                ax.set_title(task_label)
            if c == 0:
                ax.set_ylabel(gtitle, fontweight="bold")
            style_axes(ax)

    handles = [Patch(facecolor=color, hatch=hatch,
                     edgecolor=shade(color, 0.5) if hatch else color, label=lab)
               for _, lab, color, hatch, *_ in KL_SERIES]
    # ncol=4 -> two rows (KL has the extra R3 entries); position/font match the other figures.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.subplots_adjust(bottom=0.13)
    legend_below(fig, handles, ncol=4)
    fig.savefig(OUT / "kl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "kl.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "throughput"):
        plot_throughput()
    if which in ("all", "accuracy"):
        plot_accuracy()
    if which in ("all", "kl"):
        plot_kl()
