from types import SimpleNamespace

import pytest
import torch

from slim.backends.megatron.weight_sync import (
    MegatronWeightStreamer,
    bucket_named_tensors,
    iter_hf_weights,
)


class _Bridge:
    def export_hf_weights(self, model, cpu, show_progress):
        assert len(model) == 1
        assert not cpu
        assert not show_progress
        yield ("a", torch.ones(2, dtype=torch.float32))
        yield ("b", torch.ones(3, dtype=torch.float32))
        yield ("c", torch.ones(1, dtype=torch.float32))


def test_bucket_named_tensors_is_bounded_except_single_large_tensor():
    buckets = list(bucket_named_tensors(iter_hf_weights(_Bridge(), [object()]), max_bytes=12))
    assert [[name for name, _ in bucket.tensors] for bucket in buckets] == [["a"], ["b"], ["c"]]
    assert [bucket.num_bytes for bucket in buckets] == [8, 12, 4]


def test_bucket_named_tensors_rejects_nonpositive_limit():
    try:
        list(bucket_named_tensors([], max_bytes=0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_iter_hf_weights_rejects_virtual_chunks():
    try:
        list(iter_hf_weights(_Bridge(), [object(), object()]))
    except ValueError as exc:
        assert "one local model chunk" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_weight_streamer_resumes_engines_and_does_not_commit_failed_version(
    monkeypatch,
):
    events = []

    class _RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self):
            events.append(self.name)
            return self.name

    engine = SimpleNamespace(
        pause_generation=_RemoteMethod("pause"),
        flush_cache=_RemoteMethod("flush"),
        continue_generation=_RemoteMethod("continue"),
    )

    class _Transport:
        rollout_engines = [engine]
        weight_version = 0

        def wait_and_update_bucket_weights(self, _bucket):
            raise RuntimeError("transport failed")

    import ray
    import slim.utils.distributed_utils as distributed_utils

    monkeypatch.setattr(ray, "get", lambda values: values)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: object())
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)

    streamer = MegatronWeightStreamer(
        bridge=_Bridge(),
        model=[object()],
        transport=_Transport(),
        max_bytes=64,
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        streamer.update_weights()

    assert events == ["pause", "flush", "continue"]
    assert streamer.weight_version == 0
