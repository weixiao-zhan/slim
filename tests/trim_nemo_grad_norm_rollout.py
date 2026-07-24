# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound response lengths in a saved debug rollout fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slim.utils.types import Episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    return parser.parse_args()


def _slice_edges(value, edge_count: int):
    if value is None:
        return None
    return value[:edge_count]


def trim_record(record: dict, max_response_tokens: int) -> tuple[dict, int]:
    episode = Episode(**record)
    episode.ensure_edge_alignment()
    mask = torch.as_tensor(episode.loss_mask)
    generated_edges = torch.nonzero(mask, as_tuple=False).flatten()
    if generated_edges.numel() == 0:
        return record, len(episode.tokens)

    first_generated_edge = int(generated_edges[0].item())
    edge_count = min(episode.num_edges, first_generated_edge + max_response_tokens)
    token_count = edge_count + 1
    trimmed = dict(record)
    trimmed["tokens"] = episode.tokens[:token_count]
    trimmed["loss_mask"] = _slice_edges(episode.loss_mask, edge_count)
    trimmed["rollout_log_probs"] = _slice_edges(episode.rollout_log_probs, edge_count)
    trimmed["rollout_routed_experts"] = _slice_edges(episode.rollout_routed_experts, edge_count)
    trimmed["max_tokens"] = token_count
    if token_count < len(episode.tokens):
        trimmed["text"] = None
        trimmed["generated_text"] = None
        trimmed["status"] = Episode.Status.TRUNCATED

    Episode(**trimmed).ensure_edge_alignment()
    return trimmed, token_count


def main() -> None:
    args = parse_args()
    if args.max_response_tokens < 1:
        raise ValueError("--max-response-tokens must be positive")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    records = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{args.input} does not contain an episodes list")

    original_tokens = sum(len(record["tokens"]) for record in records)
    trimmed_records = []
    token_counts = []
    for record in records:
        trimmed, token_count = trim_record(record, args.max_response_tokens)
        trimmed_records.append(trimmed)
        token_counts.append(token_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "episodes": trimmed_records}, args.output)
    print(
        f"saved {len(trimmed_records)} episodes to {args.output}; "
        f"tokens {original_tokens} -> {sum(token_counts)}, max sequence {max(token_counts, default=0)}"
    )


if __name__ == "__main__":
    main()
