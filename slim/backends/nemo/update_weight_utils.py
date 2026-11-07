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
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from ray.actor import ActorHandle
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from sglang.srt.utils import MultiprocessingSerializer, init_custom_process_group
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from slim.utils.distributed_utils import get_gloo_group
from slim.utils.quant import FUSED_EXPERT_SUFFIXES
from slim.utils.timer import timer

from .model import LORA_MODULES

logger = logging.getLogger(__name__)


def _bucket_order(named_tensor: tuple[str, torch.Tensor]) -> tuple[int, bool]:
    """Sort key that puts a bucket's tensors in the order the engine needs.

    The engine receives a bucket as one flattened byte buffer and views each tensor back at its
    byte offset, so descending element size keeps every offset aligned to the dtype. The buffer
    loads in one `load_weights` call, and SGLang's Qwen3.5 loader drops every per-expert tensor
    that follows a fused expert weight in a call. So fused expert weights go last among the tensors
    of their element size; the per-expert scales of an FP8 bucket are wider and go ahead of them.
    """
    name, tensor = named_tensor
    return -tensor.element_size(), name.endswith(FUSED_EXPERT_SUFFIXES)


def _full_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=True)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor


def _compute_dtypes(model: torch.nn.Module) -> dict[str, torch.dtype]:
    """The dtype FSDP2 gathers each parameter in for the forward, by parameter name.

    A parameter computes in the policy of its nearest FSDP unit: bf16 for the bulk of the
    fp32 master weights, fp32 for units that must compute in fp32. FSDP2 keeps the policy in
    its private state, which AutoModel reads the same way. The names are the state dict's,
    which drop the activation checkpoint wrapper that `named_parameters` keeps.
    """
    unit_dtypes = {
        prefix: module._get_fsdp_state()._mp_policy.param_dtype
        for prefix, module in model.named_modules()
        if isinstance(module, FSDPModule)
    }
    dtypes = {}
    # Tied weights, such as `lm_head` sharing the embedding, appear under every name in the state dict.
    for name, _ in model.named_parameters(remove_duplicate=False):
        parts = name.split(".")
        for depth in range(len(parts) - 1, -1, -1):
            prefix = ".".join(parts[:depth])
            if prefix in unit_dtypes:
                if unit_dtypes[prefix] is not None:
                    dtypes[name.replace(_CHECKPOINT_PREFIX, "")] = unit_dtypes[prefix]
                break
    return dtypes


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module, quantizer=None) -> None:
        self.args = args
        self.model = model
        self.quantizer = quantizer
        self.state_dict_adapter = model.state_dict_adapter
        self.compute_dtypes = _compute_dtypes(model)
        # Under PEFT the engines receive every adapted weight merged into its base, and no adapter tensors.
        self.is_peft = any(isinstance(module, LORA_MODULES) for module in model.modules())
        # The engine holds one matrix for tied word embeddings, so it cannot run an adapted `lm_head`.
        if self.is_peft and model.config.tie_word_embeddings and isinstance(model.get_output_embeddings(), LinearLoRA):
            raise ValueError("LoRA on lm_head requires untied word embeddings")
        self.merged: dict[str, torch.Tensor] | None = None
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        engine_actor_ranks: Sequence[list[int] | None],
    ) -> None:
        """Attach the node-0 rollout engines.

        ``engine_actor_ranks[i]`` lists the actor ranks on the GPUs of engine ``i``,
        or is None if not all of these GPUs are actor GPUs.
        """

    @timer
    @torch.no_grad()
    def stage_merged_weights(self, offload: bool) -> None:
        """Fold every adapter into its base weight while the model is on the GPU.

        AutoModel's `materialize_effective_weight(s)` computes the merged weight from the sharded tensors.
        Each result takes the compute dtype of its base weight and, with `offload`, waits for the sync on the CPU
        like the offloaded model.
        """
        training = self.model.training
        # AutoModel folds no adapter whose dropout is active.
        self.model.eval()
        self.merged = {}
        for module_name, module in self.model.named_modules():
            if not isinstance(module, LORA_MODULES):
                continue
            if isinstance(module, LinearLoRA):
                weights = {"weight": module.materialize_effective_weight()}
            else:
                # The DeepEP grouped experts hold the same adapter tensors as `GroupedExpertsLoRA`.
                merged = GroupedExpertsLoRA.materialize_effective_weights(module)
                weights = dict(zip(("gate_and_up_projs", "down_projs"), merged, strict=True))
            prefix = f"{module_name}.".replace(_CHECKPOINT_PREFIX, "")
            for name, weight in weights.items():
                weight = weight.to(self.compute_dtypes.get(prefix + name, getattr(module, name).dtype))
                self.merged[prefix + name] = weight.to("cpu") if offload else weight
        self.model.train(training)

    def _state_tensors(self) -> dict[str, torch.Tensor]:
        """The trainer state that the engines receive, by state dict name.

        Colocated engines release their weights while the actor trains, so every synchronization sends every weight.
        Under PEFT the merged weights of the last staging replace the adapted weights, and no adapter tensor goes out.
        """
        state_dict = self.model.state_dict()
        if not self.is_peft:
            return state_dict
        # AutoModel names every adapter tensor with a `lora_` component.
        base = {
            name: tensor
            for name, tensor in state_dict.items()
            if not any(part.startswith("lora_") for part in name.split("."))
        }
        return base | self.merged

    def _hf_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        device = torch.device("cuda", torch.cuda.current_device())
        for name, state_tensor in self._state_tensors().items():
            if name.endswith("_extra_state"):
                continue

            # The engine runs what the trainer computes with, gathered in that dtype. The cast runs on
            # the GPU after the copy, because an offloaded shard is in CPU memory.
            tensor = _full_tensor(state_tensor, device, self.compute_dtypes.get(name))
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
            ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
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
            ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        self.merged = None

    def wait_and_update_bucket_weights(self, bucket):
        bucket = [
            (name, param.wait().contiguous()) if hasattr(param, "wait") else (name, param.contiguous())
            for name, param in bucket
        ]
        bucket.sort(key=_bucket_order)
        self.update_bucket_weights(bucket, weight_version=self.weight_version)
        dist.barrier(group=get_gloo_group())

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        pass


class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines using tensors.

    Streams parameters in size-bounded buckets; flattens each bucket into one tensor,
    gathers the per-rank blobs to the source, and issues one RPC per bucket.
    """

    def __init__(self, args: Namespace, model: torch.nn.Module, quantizer=None) -> None:
        super().__init__(args, model, quantizer)
        self._ipc_groups = None
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._distributed_updater = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        engine_actor_ranks: Sequence[list[int] | None],
    ) -> None:
        """Attach rollout engines: per-engine IPC (Gloo) groups on actor GPUs, NCCL broadcast elsewhere.

        The actor ranks on the GPUs of an engine gather their weights to the first of them,
        which sends them to the engine.
        """
        self.rollout_engines = rollout_engines

        # An engine restarts on the same GPUs, so the Gloo groups of the first connection stay valid.
        if self._ipc_groups is None:
            self._ipc_groups = [
                None if ranks is None else dist.new_group(ranks=ranks, backend="gloo") for ranks in engine_actor_ranks
            ]

        # Split engines into colocated (on actor GPUs, use Gloo IPC) vs
        # distributed (on other GPUs e.g. critic, use NCCL broadcast).
        # Follows the v0.2.2 Megatron use_distribute pattern.
        distributed_engines = []
        for engine, ranks, group in zip(rollout_engines, engine_actor_ranks, self._ipc_groups, strict=True):
            if ranks is None:
                distributed_engines.append(engine)
            elif dist.get_rank() in ranks:
                self._ipc_gather_src = ranks[0]
                self._ipc_gather_group = group
                self._ipc_engine = engine

        if distributed_engines:
            # One updater for all connections, so a reconnect replaces its NCCL group.
            if self._distributed_updater is None:
                self._distributed_updater = UpdateWeightFromDistributed(self.args, self.model)
            self._distributed_updater.connect_rollout_engines(distributed_engines, [None] * len(distributed_engines))

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        if self._ipc_gather_group is not None:
            monkey_patch_torch_reductions()
            flattened = FlattenedTensorBucket(named_tensors=named_tensors)
            # The payload keeps the flattened tensor alive while the engine reads it.
            payload = {
                "flattened_tensor": flattened.get_flattened_tensor(),
                "metadata": flattened.get_metadata(),
            }
            serialized = MultiprocessingSerializer.serialize(payload, output_str=True)

            if self._ipc_gather_src == dist.get_rank():
                gathered = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
            else:
                gathered = None
            dist.gather_object(
                obj=serialized,
                object_gather_list=gathered,
                dst=self._ipc_gather_src,
                group=self._ipc_gather_group,
            )

            if dist.get_rank() == self._ipc_gather_src:
                ray.get(
                    self._ipc_engine.update_weights_from_tensor.remote(
                        serialized_named_tensors=gathered,
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

    def __init__(self, args: Namespace, model: torch.nn.Module, quantizer=None) -> None:
        super().__init__(args, model, quantizer)
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        engine_actor_ranks: Sequence[list[int] | None],
    ) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast.

        Every engine receives the broadcast from rank 0, so ``engine_actor_ranks`` is not used.
        """
        self.rollout_engines = rollout_engines

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0

        if self._is_src_rank:
            # A reconnect after an engine restart replaces the group of the last connection.
            if self._model_update_groups is not None:
                dist.destroy_process_group(self._model_update_groups)
            self._group_name = "slim"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            # Rank 0 of the group is the trainer. The num_gpus GPUs of engine i follow from rank i * num_gpus + 1.
            num_gpus = self.args.rollout_num_gpus_per_replica
            world_size = len(self.rollout_engines) * num_gpus + 1
            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * num_gpus + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast the bucket as one flattened tensor."""
        if not self._is_src_rank or not named_tensors:
            return

        flattened = FlattenedTensorBucket(named_tensors=named_tensors)
        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[param.dtype for _, param in named_tensors],
                shapes=[param.shape for _, param in named_tensors],
                group_name=self._group_name,
                weight_version=str(weight_version),
                load_format="flattened_bucket",
            )
            for engine in self.rollout_engines
        ]
        dist.broadcast(flattened.get_flattened_tensor(), 0, group=self._model_update_groups)
        ray.get(refs)
