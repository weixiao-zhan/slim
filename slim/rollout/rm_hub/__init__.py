# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random

from slim.utils.misc import load_function
from slim.utils.types import Episode

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def _rule_based_reward(args, episode: Episode, **kwargs) -> int | float:
    metadata = episode.example.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        metadata = json.loads(metadata)
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    # the built-in scorer reads the last trajectory holding the attempt's answer;
    response = episode.trajectories[-1].generated_text or ""
    label = episode.example.get("label")
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def async_rm(args, episode: Episode, **kwargs) -> None:
    """Score one attempt in place, on `episode.reward` or on each `trajectory.reward`."""
    if args.custom_rm_path is not None:
        await load_function(args.custom_rm_path)(args, episode, **kwargs)
    else:
        episode.reward = await _rule_based_reward(args, episode, **kwargs)


async def batched_async_rm(args, episodes: list[Episode], **kwargs) -> None:
    """Score a whole group in place with `--group-rm`"""
    if args.custom_rm_path is not None:
        await load_function(args.custom_rm_path)(args, episodes, **kwargs)
        return
    await asyncio.gather(*[async_rm(args, episode, **kwargs) for episode in episodes])

