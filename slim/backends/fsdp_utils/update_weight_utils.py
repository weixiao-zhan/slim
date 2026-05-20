import abc
import logging
import socket
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor, Replicate

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

from slim.utils.distributed_utils import get_gloo_group, init_process_group

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]


logger = logging.getLogger(__name__)


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
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

    def update_weights(self, peft_remap: bool = False) -> None:
        """Sync model weights to rollout engines.

        Args:
            peft_remap: If True, the model is a merged PeftModel. Keys are remapped to
                HF-compatible names and adapter-only keys are skipped. Caller is responsible
                for calling merge_adapter() before and unmerge_adapter() after.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        peft_prefix = self.model.base_model.prefix if peft_remap else ""

        bucket = []
        bucket_size = 0
        for name, param in self.model.state_dict().items():
            if peft_remap:
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if peft_prefix and peft_prefix in name:
                    continue
                if "original_module" in name:
                    continue

            param_size = param.numel() * param.element_size()
            if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                self.wait_and_update_bucket_weights(bucket)
                del bucket
                torch.cuda.ipc_collect()
                bucket = []
                bucket_size = 0

            param = param.cuda()
            if isinstance(param, DTensor):
                param = param.redistribute(
                    placements=[Replicate()] * param.device_mesh.ndim,
                    async_op=True,
                ).to_local()
            bucket.append((name, param))
            bucket_size += param_size

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

        Sets the gather source rank, engine handle, and `tp_rank` within the
        engine's local group.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
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
                self.tp_rank = dist.get_rank() - start_rank

        self._distributed_updater = None
        if distributed_engines:
            self._distributed_updater = UpdateWeightFromDistributed(self.args, self.model)
            self._distributed_updater.connect_rollout_engines(
                distributed_engines,
                rollout_engine_lock,
                engine_gpu_counts=distributed_gpu_counts,
            )

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
        # gather_object is only collective among group members, so we skip entirely.
        if self._ipc_gather_group is None:
            return

        monkey_patch_torch_reductions()
        # Use flattened bucket approach for efficient weight transfer
        logger.info("Using flattened tensor bucket")
        # Group tensors by dtype
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                named_tensors_by_dtypes[dtype] = []
            named_tensors_by_dtypes[dtype].append((name, tensor))

        # Create flattened bucket for each dtype group
        serialized_tensors = []
        long_live_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            long_live_tensors.append(flattened_tensor_data)
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            # On rank 0, prepare a list to hold the gathered batches from all ranks.
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        # Gather the serialized batches from all ranks to rank 0.
        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            # Handle flattened bucket format
            # Each rank may have multiple dtype buckets
            # TODO: here we assume all ranks have the same number of dtypes
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                    "weight_version": str(weight_version),
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)

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
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)

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
