from dataclasses import dataclass
from typing import Any

from slime.utils.types import Episode


@dataclass
class RolloutFnTrainOutput:
    episodes: list[Episode]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, list[Episode]]
    metrics: dict[str, Any] = None
