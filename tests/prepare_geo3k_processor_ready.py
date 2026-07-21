# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Geometry3K dataset as parquet for VLM PPO training.

Usage:
    uv run python tests/prepare_geo3k_processor_ready.py
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "geo3k"

N_EVAL = 100 


def transform(row):
    text = row["problem"].replace("<image>", "").strip()
    row["prompt"] = [
        {
            "role": "system",
            "content": [
                {'type': 'text', 'text':"Solve the geometry problem, and put the final answer in \\boxed{...}."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": text}
            ],
        }
    ]
    row["label"] = row["answer"]
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("hiyouga/geometry3k")
    for split in ds:
        split_ds = ds[split]
        # Cap the eval (test) split at N_EVAL examples; keep train split intact.
        if split == "test":
            split_ds = split_ds.select(range(min(N_EVAL, len(split_ds))))
        split_ds.map(transform).select_columns(["prompt", "label", "images"]).to_parquet(OUT_DIR / f"{split}.parquet")
        print(f"Wrote {split}.parquet ({len(split_ds)} rows)")


if __name__ == "__main__":
    main()
