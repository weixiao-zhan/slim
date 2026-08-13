# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed rollout fixture with vision episodes assigned only to DP1."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vision-episodes", type=int, default=4)
    parser.add_argument("--layout-stride", type=int, default=8)
    return parser.parse_args()


def _has_vision(record: dict) -> bool:
    return any(traj.get("multimodal_inputs") for traj in record["trajectories"])


def build_heterogeneous_records(
    records: list[dict],
    *,
    batch_size: int = 256,
    vision_episodes: int = 4,
    layout_stride: int = 8,
) -> tuple[list[dict], list[int]]:
    if batch_size < layout_stride or batch_size % layout_stride:
        raise ValueError(f"batch size {batch_size} must be a positive multiple of layout stride {layout_stride}")
    if layout_stride % 2:
        raise ValueError("layout stride must be even so no episode reward equals its group mean")
    if vision_episodes < 1 or vision_episodes > batch_size // layout_stride:
        raise ValueError(f"vision episode count {vision_episodes} does not fit batch size {batch_size}")

    text_records = [record for record in records if not _has_vision(record)]
    vision_records = [record for record in records if _has_vision(record)]
    if not text_records or not vision_records:
        raise ValueError("source rollout must contain both text and vision episodes")

    batch = [copy.deepcopy(text_records[index % len(text_records)]) for index in range(batch_size)]
    target_indices = [1 + layout_stride * index for index in range(vision_episodes)]
    for vision_offset, target in enumerate(target_indices):
        batch[target] = copy.deepcopy(vision_records[vision_offset % len(vision_records)])

    reward_denominator = layout_stride - 1
    for index, record in enumerate(batch):
        group_index, group_offset = divmod(index, layout_stride)
        record["reward"] = ((group_offset + group_index) % layout_stride) / reward_denominator

    for dp_size in (8, 4):
        vision_dp_ranks = {index % dp_size for index, record in enumerate(batch) if _has_vision(record)}
        if vision_dp_ranks != {1}:
            raise RuntimeError(f"vision episodes map to DP ranks {sorted(vision_dp_ranks)} for DP size {dp_size}")

    return batch, target_indices


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    records = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{args.input} does not contain an episodes list")

    episodes, vision_indices = build_heterogeneous_records(
        records,
        batch_size=args.batch_size,
        vision_episodes=args.vision_episodes,
        layout_stride=args.layout_stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "episodes": episodes,
            "heterogeneous_layout": {
                "vision_dp_rank": 1,
                "vision_episode_indices": vision_indices,
            },
        },
        args.output,
    )
    print(f"saved {len(episodes)} episodes to {args.output}; vision indices {vision_indices}, all other episodes text-only")


if __name__ == "__main__":
    main()
