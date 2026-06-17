"""Prepare a mixed math (text) + geometry (vision) dataset for hybrid GRPO training.

Combines two well-understood, fast-to-fetch sources:
  - open-r1/DAPO-Math-17k-Processed (English, text-only math)
  - hiyouga/geometry3k              (geometry reasoning with images)

All prompts use VLM processor format (content as list of {type, text} dicts). Image blocks
use {"type": "image", "text": ""} for Arrow schema compatibility; the Qwen VL processor
produces identical output with or without the text key.

Outputs (the geometry split is the "vision" eval task, so the test scripts' eval keys and
paths stay unchanged):
  <repo>/datasets/mixed/train.parquet        — mixed text + vision (for training)
  <repo>/datasets/mixed/test_math.parquet    — text-only math (for eval)
  <repo>/datasets/mixed/test_vision.parquet  — geometry vision reasoning (for eval)

Usage:
    uv run python tests/prepare_mixed.py
"""

from pathlib import Path

from datasets import Image, Sequence, concatenate_datasets, load_dataset

# Write into the repo's own datasets/ dir (this file lives in <repo>/tests/).
OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mixed"

N_EVAL = 100

def transform_dapo(row):
    """Pure-text math problem in VLM format."""
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
    """Vision geometry problem with one image in VLM format."""
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Text-only math: DAPO-Math-17k (hold out N_EVAL for eval) ----
    dapo_all = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train").shuffle(seed=42)
    cols_to_drop = [c for c in dapo_all.column_names if c not in ("prompt", "label", "images")]
    dapo_test = (
        dapo_all.select(range(N_EVAL))
        .map(transform_dapo, remove_columns=cols_to_drop)
        .cast_column("images", Sequence(Image()))
    )
    dapo_train = (
        dapo_all.select(range(N_EVAL, len(dapo_all)))
        .map(transform_dapo, remove_columns=cols_to_drop)
        .cast_column("images", Sequence(Image()))
    )
    print(f"DAPO-Math-17k (en): {len(dapo_train)} train, {len(dapo_test)} test")

    # ---- Vision geometry: Geometry3K (cap test split at N_EVAL) ----
    geo3k = load_dataset("hiyouga/geometry3k")
    geo3k_train = geo3k["train"].map(transform_geo3k).select_columns(["prompt", "label", "images"])
    geo3k_test = (
        geo3k["test"].select(range(N_EVAL)).map(transform_geo3k).select_columns(["prompt", "label", "images"])
    )
    print(f"Geometry3K: {len(geo3k_train)} train, {len(geo3k_test)} test")

    # Train set — mixed, same schema so concat works
    train = concatenate_datasets([dapo_train, geo3k_train]).shuffle(seed=42)
    train.to_parquet(OUT_DIR / "train.parquet")
    print(f"Wrote train.parquet ({len(train)} rows: {len(dapo_train)} text + {len(geo3k_train)} vision)")

    # Separate test files for per-task eval reporting
    dapo_test.to_parquet(OUT_DIR / "test_math.parquet")
    print(f"Wrote test_math.parquet ({len(dapo_test)} rows)")

    geo3k_test.to_parquet(OUT_DIR / "test_vision.parquet")
    print(f"Wrote test_vision.parquet ({len(geo3k_test)} rows)")


if __name__ == "__main__":
    main()
