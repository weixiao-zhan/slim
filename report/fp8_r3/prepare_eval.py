"""Build the eval parquets for the FP8/R3 study, using slim's VLM prompt schema: test_math.parquet
(open-r1/DAPO-Math-17k) and test_vision.parquet (hiyouga/geometry3k), --n-eval examples each.

Args:
  --n-eval     examples per task (default 500).
  --out-dir    output dir for the parquets (default repo datasets/eval/).
"""

import argparse
from pathlib import Path

from datasets import Image, Sequence, load_dataset

# This file lives at report/fp8_r3/; the eval set defaults to the repo-root datasets/ dir, which is
# parents[2] (report/fp8_r3 -> report -> slim).
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "eval"


def transform_dapo(row):
    row["prompt"] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Solve the math problem, and put the final answer in \\boxed{...}."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": row["prompt"].strip()},
            ],
        },
    ]
    row["label"] = row["reward_model"]["ground_truth"]
    row["images"] = []
    return row


def transform_geo3k(row):
    text = row["problem"].replace("<image>", "").strip()
    row["prompt"] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Solve the geometry problem, and put the final answer in \\boxed{...}."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "text": ""},
                {"type": "text", "text": text},
            ],
        },
    ]
    row["label"] = row["answer"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=500, help="examples per task to take")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="output dir for the parquets (default: repo datasets/eval/)")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Text math: take the FIRST --n-eval (disjoint from tests/prepare_mixed.py's train slice, which
    # uses range(100, len) — these may overlap that train slice, but for a standalone eval study that
    # is fine since we are not training on it here).
    dapo_all = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train").shuffle(seed=42)
    cols_to_drop = [c for c in dapo_all.column_names if c not in ("prompt", "label", "images")]
    dapo_test = (
        dapo_all.select(range(args.n_eval))
        .map(transform_dapo, remove_columns=cols_to_drop)
        .cast_column("images", Sequence(Image()))
    )
    dapo_test.to_parquet(out_dir / "test_math.parquet")
    print(f"Wrote test_math.parquet ({len(dapo_test)} rows)")

    # Vision geometry: geo3k test has 601; take up to --n-eval.
    geo3k = load_dataset("hiyouga/geometry3k")
    n_vision = min(args.n_eval, len(geo3k["test"]))
    geo3k_test = (
        geo3k["test"].select(range(n_vision)).map(transform_geo3k).select_columns(["prompt", "label", "images"])
    )
    geo3k_test.to_parquet(out_dir / "test_vision.parquet")
    print(f"Wrote test_vision.parquet ({len(geo3k_test)} rows)")


if __name__ == "__main__":
    main()
