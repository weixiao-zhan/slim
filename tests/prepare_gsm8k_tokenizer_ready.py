"""Prepare GSM8K dataset as parquet for PPO training.

Usage:
    uv run python tests/prepare_gsm8k_tokenizer_ready.py
"""

from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("/home/ubuntu/datasets/gsm8k")


def transform(row):
    parts = row["answer"].split("####")
    row["prompt"] = [
        {
            "role": "system",
            "content": "Solve the math problem, and put the final answer in \\boxed{...}."
        },
        {
            "role": "user",
            "content": row["question"].strip(),
        }
    ]
    row["label"] = int(parts[-1].strip().replace(",", ""))
    row["metadata"] = {"source": "gsm8k"}
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    for split in ds:
        ds[split].map(transform).select_columns(["prompt", "label", "metadata"]).to_parquet(OUT_DIR / f"{split}.parquet")
        print(f"Wrote {split}.parquet")


if __name__ == "__main__":
    main()
