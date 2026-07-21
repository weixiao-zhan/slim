# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from slim.utils.types import Episode


@dataclass
class RolloutFnTrainOutput:
    episodes: list[Episode]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, list[Episode]]
    metrics: dict[str, Any] = None
