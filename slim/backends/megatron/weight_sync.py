"""Stream Bridge-converted HuggingFace weights to SGLang transports."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

import torch


class NamedTensorBucket(NamedTuple):
    tensors: list[tuple[str, torch.Tensor]]
    num_bytes: int


def bucket_named_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    max_bytes: int,
) -> Iterator[NamedTensorBucket]:
    """Group a tensor stream without materializing the full state dict."""

    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}.")

    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_bytes = 0
    for name, tensor in tensors:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if bucket and bucket_bytes + tensor_bytes > max_bytes:
            yield NamedTensorBucket(bucket, bucket_bytes)
            bucket = []
            bucket_bytes = 0
        bucket.append((name, tensor))
        bucket_bytes += tensor_bytes
    if bucket:
        yield NamedTensorBucket(bucket, bucket_bytes)


def iter_hf_weights(
    bridge: Any,
    model: list[torch.nn.Module],
    *,
    quantizer: Any = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield canonical HF names from Bridge, applying Slim quantization if configured."""

    if len(model) != 1:
        raise ValueError(
            "The slim Megatron backend requires one local model chunk; "
            f"received {len(model)}."
        )
    for item in bridge.export_hf_weights(model, cpu=False, show_progress=False):
        name, tensor = item[:2]
        converted = quantizer.quantize(name, tensor) if quantizer is not None else ((name, tensor),)
        yield from converted


class MegatronWeightStreamer:
    """Drive an existing Slim weight transport from a Bridge export stream."""

    def __init__(
        self,
        *,
        bridge: Any,
        model: list[torch.nn.Module],
        transport: Any,
        max_bytes: int,
        quantizer: Any = None,
    ) -> None:
        if len(model) != 1:
            raise ValueError(
                "The slim Megatron backend requires one local model chunk; "
                f"received {len(model)}."
            )
        self.bridge = bridge
        self.model = model
        self.transport = transport
        self.max_bytes = max_bytes
        self.quantizer = quantizer
        self.weight_version = 0

    def connect_rollout_engines(self, *args, **kwargs) -> None:
        self.transport.connect_rollout_engines(*args, **kwargs)

    @torch.no_grad()
    def update_weights(self) -> None:
        """Export and send bounded buckets through the selected Slim transport."""

        import ray
        import torch.distributed as dist

        from slim.utils.distributed_utils import get_gloo_group

        next_weight_version = self.weight_version + 1
        self.transport.weight_version = next_weight_version
        rank = dist.get_rank()
        paused = False
        completed = False
        try:
            if rank == 0:
                paused = True
                ray.get(
                    [
                        engine.pause_generation.remote()
                        for engine in self.transport.rollout_engines
                    ]
                )
                ray.get(
                    [
                        engine.flush_cache.remote()
                        for engine in self.transport.rollout_engines
                    ]
                )
            dist.barrier(group=get_gloo_group())

            stream = iter_hf_weights(self.bridge, self.model, quantizer=self.quantizer)
            for bucket in bucket_named_tensors(stream, self.max_bytes):
                self.transport.wait_and_update_bucket_weights(bucket.tensors)

            dist.barrier(group=get_gloo_group())
            torch.cuda.ipc_collect()
            self.weight_version = next_weight_version
            completed = True
        finally:
            if rank == 0 and paused:
                ray.get(
                    [
                        engine.continue_generation.remote()
                        for engine in self.transport.rollout_engines
                    ]
                )
        if completed:
            dist.barrier(group=get_gloo_group())
