# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a math (text) + geometry (vision) dataset for the Strands calculator rollout.

Same two sources as `prepare_mixed.py`:
  - open-r1/DAPO-Math-17k-Processed (English, text-only math)
  - hiyouga/geometry3k              (geometry reasoning with images)

The rows carry the flat schema `examples/strands_calculator/generate.py` reads: `prompt`
and `system` as plain strings, `images` as PIL images of the user turn, `label` as the
ground truth for `--rm-type math`. The agent framework owns everything above that: Strands
renders the turns and advertises the calculator, and the SGLang engine applies the chat
template.

Outputs:
  <repo>/datasets/strands_calculator/train.parquet        — mixed text + vision (training)
  <repo>/datasets/strands_calculator/test_math.parquet    — text-only math (eval)
  <repo>/datasets/strands_calculator/test_vision.parquet  — geometry vision reasoning (eval)

Usage:
    uv run python examples/strands_calculator/prepare_dataset.py
"""

from pathlib import Path

from datasets import Image, Sequence, concatenate_datasets, load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "datasets" / "strands_calculator"

N_EVAL = 100

SYSTEM_PROMPT = (
    "Solve the problem. Use the calculator tool for any arithmetic you are unsure of, "
    "then put the final answer in \\boxed{...}."
)


def transform_dapo(row):
    """Pure-text math problem."""
    row["system"] = SYSTEM_PROMPT
    row["prompt"] = row["prompt"].strip()
    row["label"] = row["reward_model"]["ground_truth"]
    row["images"] = []
    return row


def transform_geo3k(row):
    """Geometry problem whose figure travels in `images`."""
    row["system"] = SYSTEM_PROMPT
    row["prompt"] = row["problem"].replace("<image>", "").strip()
    row["label"] = row["answer"]
    return row


COLUMNS = ["prompt", "system", "label", "images"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Text-only math, shuffled once so the eval hold-out is a random slice.
    dapo = (
        load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
        .shuffle(seed=42)
        .map(transform_dapo)
        .select_columns(COLUMNS)
        .cast_column("images", Sequence(Image()))
    )
    dapo_test, dapo_train = dapo.select(range(N_EVAL)), dapo.select(range(N_EVAL, len(dapo)))

    # Vision geometry, whose own test split supplies the eval rows.
    geo3k = (
        load_dataset("hiyouga/geometry3k")
        .map(transform_geo3k)
        .select_columns(COLUMNS)
    )
    geo3k_train, geo3k_test = geo3k["train"], geo3k["test"].select(range(N_EVAL))

    splits = {
        "train": concatenate_datasets([dapo_train, geo3k_train]).shuffle(seed=42),
        "test_math": dapo_test,
        "test_vision": geo3k_test,
    }
    for name, split in splits.items():
        split.to_parquet(OUT_DIR / f"{name}.parquet")
        print(f"Wrote {name}.parquet ({len(split)} rows)")


if __name__ == "__main__":
    main()
