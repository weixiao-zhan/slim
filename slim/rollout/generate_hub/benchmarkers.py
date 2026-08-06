# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random

from slim.rollout.sglang_rollout import generate as _generate_base
from slim.utils.types import Episode

logger = logging.getLogger(__name__)


async def generate_with_random_osl(state, episode: Episode) -> Episode:
    # TODO: make it configurable after we have an enhanced arg parser
    min_osl = 32 * 1024
    max_osl = 64 * 1024

    if episode._sampling_params is None:
        raise RuntimeError("Episode._sampling_params must be initialized before generation.")
    episode._sampling_params["ignore_eos"] = True
    episode._sampling_params["max_new_tokens"] = random.randrange(min_osl, max_osl)

    ans = await _generate_base(state, episode)

    logger.info(f"generate_with_random_osl response_length={ans.trajectory.response_length}")
    return ans
