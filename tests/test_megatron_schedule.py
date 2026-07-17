from __future__ import annotations

import sys
from types import ModuleType

import pytest

from slim.backends.megatron.schedule import get_forward_backward_func, run_forward_backward


pytestmark = pytest.mark.unit


@pytest.fixture
def fake_pipeline_parallel(monkeypatch):
    module = ModuleType("megatron.core.pipeline_parallel")
    calls = []
    schedule_calls = []

    def schedule(**kwargs):
        schedule_calls.append(kwargs)
        return "scheduled"

    def fake_get_forward_backward_func(**kwargs):
        calls.append(kwargs)
        return schedule

    module.get_forward_backward_func = fake_get_forward_backward_func
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls, schedule_calls


def test_schedule_selection_explicitly_disables_vp(fake_pipeline_parallel):
    calls, _ = fake_pipeline_parallel

    selected = get_forward_backward_func(2)

    assert callable(selected)
    assert calls == [{"pp_size": 2, "vp_size": None}]


def test_run_forward_backward_preserves_one_model_chunk(fake_pipeline_parallel):
    _, schedule_calls = fake_pipeline_parallel
    chunk = object()
    iterator = iter([object()])

    def forward_step(*_args):
        return None

    result = run_forward_backward(
        pp_size=1,
        model=[chunk],
        forward_step_func=forward_step,
        data_iterator=iterator,
        num_microbatches=1,
        seq_length=8,
        micro_batch_size=1,
    )

    assert result == "scheduled"
    assert schedule_calls[0]["model"] == [chunk]
    assert schedule_calls[0]["data_iterator"] is iterator
    assert schedule_calls[0]["forward_step_func"] is forward_step


def test_run_forward_backward_rejects_multiple_chunks(fake_pipeline_parallel):
    with pytest.raises(ValueError, match="exactly one model chunk"):
        run_forward_backward(
            pp_size=2,
            model=[object(), object()],
            forward_step_func=lambda *_args: None,
            data_iterator=iter(()),
        )


@pytest.mark.parametrize("pp_size", [0, -1, True, 1.5])
def test_schedule_rejects_invalid_pp_size(fake_pipeline_parallel, pp_size):
    with pytest.raises(ValueError, match="positive integer"):
        get_forward_backward_func(pp_size)
