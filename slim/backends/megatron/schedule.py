"""MCore forward and backward schedule selection without virtual pipelines."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from .topology import require_single_model_chunk


def get_forward_backward_func(pp_size: int) -> Callable[..., Any]:
    """Select the MCore schedule with an explicit non-interleaved topology."""

    if isinstance(pp_size, bool) or not isinstance(pp_size, int) or pp_size < 1:
        raise ValueError(f"pp_size must be a positive integer, got {pp_size!r}.")
    pipeline_parallel = importlib.import_module("megatron.core.pipeline_parallel")
    return pipeline_parallel.get_forward_backward_func(pp_size=pp_size, vp_size=None)


def run_forward_backward(
    *,
    pp_size: int,
    model: Sequence[Any],
    forward_step_func: Callable[..., Any],
    data_iterator: Iterator[Any] | Sequence[Iterator[Any]],
    **schedule_kwargs: Any,
) -> Any:
    """Run one MCore schedule while preserving the one-chunk model contract."""

    model_chunk = require_single_model_chunk(model)
    schedule = get_forward_backward_func(pp_size)
    return schedule(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=[model_chunk],
        **schedule_kwargs,
    )
