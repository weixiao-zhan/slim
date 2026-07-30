# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed mixed-modality prompt set for trainer comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-data", required=True)
    parser.add_argument("--vision-data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts-per-modality", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    math_data = Dataset.from_parquet(args.math_data).select(range(args.prompts_per_modality))
    vision_data = Dataset.from_parquet(args.vision_data).select(range(args.prompts_per_modality))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = concatenate_datasets([math_data, vision_data])
    prompts.to_parquet(args.output)
    print(f"saved {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
