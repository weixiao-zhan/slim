# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound response lengths in a saved debug rollout fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from slim.utils.types import Episode, Trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    return parser.parse_args()


def _slice_predictions(value, prediction_count: int):
    if value is None:
        return None
    return value[:prediction_count]


def _trim_span(span: dict, max_response_tokens: int) -> tuple[dict, int]:
    """Bound one span's response, returning the trimmed span and its token count."""
    trajectory = Trajectory(**span)
    mask = torch.as_tensor(trajectory.loss_mask)
    generated_positions = torch.nonzero(mask, as_tuple=False).flatten()
    if generated_positions.numel() == 0:
        return span, len(trajectory.token_ids)

    first_generated_position = int(generated_positions[0].item())
    prediction_count = min(
        len(trajectory.token_ids) - 1,
        first_generated_position + max_response_tokens,
    )
    token_count = prediction_count + 1
    trimmed = dict(span)
    trimmed["token_ids"] = trajectory.token_ids[:token_count]
    trimmed["loss_mask"] = _slice_predictions(trajectory.loss_mask, prediction_count)
    trimmed["rollout_log_probs"] = _slice_predictions(
        trajectory.rollout_log_probs,
        prediction_count,
    )
    trimmed["rollout_routed_experts"] = _slice_predictions(
        trajectory.rollout_routed_experts,
        prediction_count,
    )
    if token_count < len(trajectory.token_ids):
        trimmed["text"] = None
        trimmed["generated_text"] = None

    Trajectory(**trimmed).finalize_source_token_alignment()
    return trimmed, token_count


def trim_record(record: dict, max_response_tokens: int) -> tuple[dict, int]:
    """Bound every span of one attempt."""
    spans = []
    token_counts = []
    for span in record["trajectories"]:
        trimmed, token_count = _trim_span(span, max_response_tokens)
        spans.append(trimmed)
        token_counts.append(token_count)

    trimmed_record = dict(record)
    trimmed_record["trajectories"] = spans
    trimmed_record["max_tokens"] = max(token_counts, default=0)
    if any(len(span["token_ids"]) < len(original["token_ids"]) for span, original in zip(spans, record["trajectories"], strict=True)):
        trimmed_record["status"] = Episode.Status.TRUNCATED
    return trimmed_record, sum(token_counts)


def main() -> None:
    args = parse_args()
    if args.max_response_tokens < 1:
        raise ValueError("--max-response-tokens must be positive")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    records = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{args.input} does not contain an episodes list")

    original_tokens = sum(
        len(span["token_ids"]) for record in records for span in record["trajectories"]
    )
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
