"""Prepare DAPO-Math-17k dataset as parquet for GRPO training.

Usage:
    uv run python tests/prepare_dapo17k_tokenizer_ready.py
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "dapo17k"


def transform(row):
    row["prompt"] = [
        {
            "role": "system",
            "content": "Solve the math problem, and put the final answer in \\boxed{...}.",
        },
        {
            "role": "user",
            "content": row["prompt"].strip(),
        },
    ]
    row["label"] = row["reward_model"]["ground_truth"]
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    ds = ds.map(transform).select_columns(["prompt", "label"])
    ds.to_parquet(OUT_DIR / "train.parquet")
    print(f"Wrote train.parquet ({len(ds)} rows)")


if __name__ == "__main__":
    main()
