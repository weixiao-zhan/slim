# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
import socket
from argparse import Namespace
from collections.abc import Sequence
from collections.abc import Iterator

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from slim.utils.distributed_utils import get_gloo_group, init_process_group

logger = logging.getLogger(__name__)


def _full_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=True)
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module, quantizer=None) -> None:
        self.args = args
        self.model = model
        self.quantizer = quantizer
        self.state_dict_adapter = model.state_dict_adapter
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        pass

    def _hf_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        device = torch.device("cuda", torch.cuda.current_device())
        for name, state_tensor in self.model.state_dict().items():
            if name.endswith("_extra_state"):
                continue

            tensor = _full_tensor(state_tensor, device)
            converted = self.state_dict_adapter.convert_single_tensor_to_hf(
                name,
                tensor,
                exclude_key_regex=r".*_extra_state.*",
                quantization=False,
            )
            for hf_name, hf_tensor in converted:
                hf_tensor = _full_tensor(hf_tensor, device).contiguous()
                yield hf_name, hf_tensor

    def update_weights(self) -> None:
        """Convert and stream the current policy weights to rollout engines."""
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        bucket = []
        bucket_size = 0
        for name, param in self._hf_tensors():
            named_tensors = self.quantizer.quantize(name, param) if self.quantizer else ((name, param),)
            for tensor_name, tensor in named_tensors:
                tensor_size = tensor.numel() * tensor.element_size()
                if bucket and bucket_size + tensor_size >= self.args.update_weight_buffer_size:
                    self.wait_and_update_bucket_weights(bucket)
                    del bucket
                    torch.cuda.ipc_collect()
                    bucket = []
                    bucket_size = 0
                bucket.append((tensor_name, tensor))
                bucket_size += tensor_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()

        if rank == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def wait_and_update_bucket_weights(self, bucket):
        bucket = [
            (name, param.wait().contiguous()) if hasattr(param, "wait") else (name, param.contiguous())
            for name, param in bucket
        ]
        self.update_bucket_weights(bucket, weight_version=self.weight_version)
        dist.barrier(group=get_gloo_group())

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        pass


class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using tensors.

    Streams parameters in size-bounded buckets; optionally groups tensors by dtype
    and flattens per dtype, gathers per-rank blobs to the source, and issues one
    RPC per dtype per bucket (or one per bucket if not flattened).
    """

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Attach rollout engines and create per-engine IPC (Gloo) groups.

        Sets the gather source rank and engine handle within the engine's local
        group.
        """
        self.rollout_engines = rollout_engines
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_replica] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Split engines into colocated (on actor GPUs, use Gloo IPC) vs
        # distributed (on other GPUs e.g. critic, use NCCL broadcast).
        # Follows the v0.2.2 Megatron use_distribute pattern.
        world_size = dist.get_world_size()
        distributed_engines = []
        distributed_gpu_counts = []

        for i, engine in enumerate(self.rollout_engines):
            start_rank = engine_gpu_offsets[i]
            end_rank = start_rank + engine_gpu_counts[i]
            group_ranks = list(range(start_rank, end_rank))

            if any(r >= world_size for r in group_ranks):
                distributed_engines.append(engine)
                distributed_gpu_counts.append(engine_gpu_counts[i])
                continue

            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine

        self._distributed_updater = None
        if distributed_engines:
            self._distributed_updater = UpdateWeightFromDistributed(self.args, self.model)
            self._distributed_updater.connect_rollout_engines(
                distributed_engines,
                rollout_engine_lock,
                engine_gpu_counts=distributed_gpu_counts,
            )

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        if self._ipc_gather_group is not None:
            monkey_patch_torch_reductions()
            named_tensors_by_dtype = {}
            for name, tensor in named_tensors:
                named_tensors_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            serialized_tensors = []
            long_live_tensors = []
            for tensors in named_tensors_by_dtype.values():
                flattened = FlattenedTensorBucket(named_tensors=tensors)
                payload = {
                    "flattened_tensor": flattened.get_flattened_tensor(),
                    "metadata": flattened.get_metadata(),
                }
                long_live_tensors.append(payload)
                serialized_tensors.append(MultiprocessingSerializer.serialize(payload, output_str=True))

            if self._ipc_gather_src == dist.get_rank():
                gathered = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
            else:
                gathered = None
            dist.gather_object(
                obj=serialized_tensors,
                object_gather_list=gathered,
                dst=self._ipc_gather_src,
                group=self._ipc_gather_group,
            )

            if dist.get_rank() == self._ipc_gather_src:
                for index in range(len(gathered[0])):
                    ray.get(
                        self._ipc_engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=[tensors[index] for tensors in gathered],
                            load_format="flattened_bucket",
                            flush_cache=False,
                            weight_version=str(weight_version),
                        )
                    )

        # Update engines on non-actor GPUs via NCCL broadcast
        if self._distributed_updater is not None:
            self._distributed_updater.update_bucket_weights(named_tensors, weight_version=weight_version)


class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast."""
        self.rollout_engines = rollout_engines

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_replica] * len(rollout_engines)

        if self._is_src_rank:
            self._group_name = "slim"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = sum(engine_gpu_counts) + 1

            # Compute cumulative rank offsets.
            cumulative = [0]
            for c in engine_gpu_counts:
                cumulative.append(cumulative[-1] + c)

            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    cumulative[i] + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast tensors.

        Ensures tensors are contiguous; when `world_size == 1`, converts DTensors
        to full tensors prior to `dist.broadcast`.
        """
        if not self._is_src_rank or not named_tensors:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[param.dtype for _, param in named_tensors],
                shapes=[param.shape for _, param in named_tensors],
                group_name=self._group_name,
                weight_version=str(weight_version),
            )
            for engine in self.rollout_engines
        ]

        handles = []
        # Broadcast parameters one by one with memory management
        for _name, param in named_tensors:
            torch.cuda.empty_cache()
            # Ensure tensor is contiguous and on the right device
            param_data = param.data.contiguous()

            # avoid `DTensor._op_dispatcher.dispatch` has `assert compute_mesh is not None` error
            if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                param_data = param_data.full_tensor()

            # Synchronous broadcast to avoid memory buildup
            handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

        for handle in handles:
            handle.wait()
        ray.get(refs)
