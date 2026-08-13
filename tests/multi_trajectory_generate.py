# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generate function that produces multi-trajectory episodes.

Each attempt makes two generation calls. The second span starts from a compressed
view of the first rather than extending it, so the two are separate contiguous
spans with no order relation the layout needs to know about. This exercises the
flatten-pad-partition path with $k > 1$ documents per episode, which the
append-only sweeps do not reach.

Wire it in with ``--custom-generate-function-path tests.multi_trajectory_generate.generate``.
"""

from __future__ import annotations

from slim.rollout.sglang_rollout import generate as generate_span
from slim.utils.types import Episode, Trajectory

# Tokens of the first span kept as the second span's prompt.
_COMPRESSED_PROMPT_TOKENS = 64
# Generation budget for the second span.
_SPAN_MAX_NEW_TOKENS = 256


async def generate(state, episode: Episode) -> Episode:
    """Generate two spans per attempt, the second over a compressed context."""
    episode = await generate_span(state, episode)
    first = episode.trajectories[0]
    if episode.status != Episode.Status.COMPLETED or len(first.token_ids) < 2:
        return episode

    # A fresh span whose prompt is the tail of the first, not a continuation of it.
    prompt = list(first.token_ids[-_COMPRESSED_PROMPT_TOKENS:])
    edge_length = max(len(prompt) - 1, 0)
    episode.trajectories.append(
        Trajectory(
            token_ids=prompt,
            loss_mask=[0] * edge_length,
            rollout_log_probs=[0.0] * edge_length,
        )
    )

    # generate() writes into episode.trajectory, so hand it only the new span. Its
    # budget counts from the compressed prompt, not the first span's full length.
    tail = Episode(
        example=episode.example,
        trajectories=[episode.trajectories[-1]],
        max_tokens=len(prompt) + _SPAN_MAX_NEW_TOKENS,
        session_id=episode.session_id,
        sampling_seed=episode.sampling_seed,
    )
    tail = await generate_span(state, tail)
    episode.trajectories[-1] = tail.trajectory
    # The attempt is truncated if either span hit its budget.
    if tail.status == Episode.Status.TRUNCATED:
        episode.status = Episode.Status.TRUNCATED
    return episode
