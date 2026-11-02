# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import Namespace
from copy import deepcopy

from slim.utils import logging_utils
from slim.utils.metric_utils import compute_rollout_step
from slim.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_perf_data_raw(rollout_id: int, args: Namespace, is_primary_rank: bool) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    counter_dict_raw = deepcopy(timer_instance.counter_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    log_dict.update({f"perf/{key}": val for key, val in counter_dict_raw.items()})

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict)
