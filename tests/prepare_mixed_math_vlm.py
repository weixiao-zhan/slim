"""Prepare mixed math (text) + geometry (VLM) dataset for hybrid GRPO training.

Combines:
  - open-r1/DAPO-Math-17k-Processed (English, text-only math)
  - hiyouga/geometry3k (geometry with images)

All prompts use VLM processor format (content as list of {type, text} dicts).
Image blocks use {"type": "image", "text": ""} for Arrow schema compatibility;
the Qwen VL processor produces identical output with or without the text key.

Outputs:
  train.parquet      — mixed text + vision (for training)
  test_math.parquet  — text-only math (for eval)
  test_geo3k.parquet — vision geometry (for eval)

Usage:
    uv run python tests/prepare_mixed_math_vlm.py
"""

from pathlib import Path

from datasets import Features, Image, Sequence, Value, concatenate_datasets, load_dataset

OUT_DIR = Path.home() / "datasets" / "mixed_math_vlm"


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
    """Vision math problem with image in VLM format."""
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

    # Text-only math — hold out 500 for test
    dapo_all = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    dapo_all = dapo_all.shuffle(seed=42)
    cols_to_drop = [c for c in dapo_all.column_names if c not in ("prompt", "label", "images")]
    dapo_test = dapo_all.select(range(500)).map(transform_dapo, remove_columns=cols_to_drop).cast_column("images", Sequence(Image()))
    dapo_train = dapo_all.select(range(500, len(dapo_all))).map(transform_dapo, remove_columns=cols_to_drop).cast_column("images", Sequence(Image()))
    print(f"DAPO-Math-17k (en): {len(dapo_train)} train, {len(dapo_test)} test")

    # Vision math
    geo3k = load_dataset("hiyouga/geometry3k")
    geo3k_train = geo3k["train"].map(transform_geo3k).select_columns(["prompt", "label", "images"])
    geo3k_test = geo3k["test"].map(transform_geo3k).select_columns(["prompt", "label", "images"])
    print(f"Geometry3K: {len(geo3k_train)} train, {len(geo3k_test)} test")

    # Train set — mixed, same schema so concat works
    train = concatenate_datasets([dapo_train, geo3k_train]).shuffle(seed=42)
    train.to_parquet(OUT_DIR / "train.parquet")
    print(f"Wrote train.parquet ({len(train)} rows: {len(dapo_train)} text + {len(geo3k_train)} vision)")

    # Separate test files for per-task eval reporting
    dapo_test.to_parquet(OUT_DIR / "test_math.parquet")
    print(f"Wrote test_math.parquet ({len(dapo_test)} rows)")

    geo3k_test.to_parquet(OUT_DIR / "test_geo3k.parquet")
    print(f"Wrote test_geo3k.parquet ({len(geo3k_test)} rows)")


if __name__ == "__main__":
    main()
