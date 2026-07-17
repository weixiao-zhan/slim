"""Ray-facing actor trainer built on Megatron Bridge and Megatron Core."""

from __future__ import annotations

import logging
import os
import random
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist

from slim.ray.ray_worker import RayWorker
from slim.utils import logging_utils
from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group, init_gloo_group
from slim.utils.logging_utils import configure_logger, init_tracking
from slim.utils.memory_utils import clear_memory
from slim.utils.quant import Quantizer

from .batch_layout import (
    PackedLayout,
    build_packed_layout,
    packed_sequence_alignment,
)
from .checkpoint import CheckpointMetadata, MegatronCheckpointManager
from .loss import objective_weights, policy_loss, selected_log_probs, vocab_parallel_entropy
from .model import MegatronModelBundle, build_megatron_model
from .routing_replay import ModelScopedRouterReplay
from .schedule import run_forward_backward
from .weight_sync import MegatronWeightStreamer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PreparedBatch:
    """One packed microbatch and its rank-local sequence views."""

    layout: PackedLayout
    episodes: Sequence[Any]
    tensors: dict[str, torch.Tensor]
    visual_kwargs: dict[str, torch.Tensor]
    packed_seq_params: Any
    cp_indices: torch.Tensor
    sequence_ids: torch.Tensor
    sequence_active_counts: torch.Tensor


def _is_hf_checkpoint(path: str | Path | None) -> bool:
    return path is not None and (Path(path) / "config.json").is_file()


def _as_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _split_by_capacity(
    episodes: Sequence[Any],
    capacity: int,
    *,
    alignment: int = 1,
) -> list[list[Any]]:
    """Pack episodes with first-fit placement while retaining per-bin order."""

    if capacity < 1:
        raise ValueError(f"Token capacity must be positive, got {capacity}.")
    if alignment < 1:
        raise ValueError(f"Token alignment must be positive, got {alignment}.")
    bins: list[list[Any]] = []
    bin_tokens: list[int] = []
    for episode in episodes:
        logical_length = len(episode.tokens)
        physical_length = (
            (logical_length + alignment - 1) // alignment
        ) * alignment
        if physical_length > capacity:
            raise ValueError(
                f"Aligned episode length {physical_length} exceeds the configured "
                f"token capacity {capacity}."
            )
        for index, used in enumerate(bin_tokens):
            if used + physical_length <= capacity:
                bins[index].append(episode)
                bin_tokens[index] += physical_length
                break
        else:
            bins.append([episode])
            bin_tokens.append(physical_length)
    return bins


def _rebalance_microbatches(
    batches: Sequence[Sequence[Any]],
    target_count: int,
) -> list[list[Any]]:
    """Split existing bins until every DP rank has the same schedule length."""

    result = [list(batch) for batch in batches]
    if target_count < len(result):
        raise ValueError(
            f"target_count {target_count} is smaller than local batch count {len(result)}."
        )
    while len(result) < target_count:
        split_index = next(
            (index for index, batch in enumerate(result) if len(batch) > 1),
            None,
        )
        if split_index is None:
            raise ValueError(
                f"Cannot split {sum(map(len, result))} episodes into "
                f"{target_count} nonempty microbatches."
            )
        result.append([result[split_index].pop()])
    return result


def _resolve_distributed_checkpoint(
    path: str | Path | None,
    *,
    is_checkpoint=None,
) -> Path | None:
    """Resolve either a direct DCP directory or the latest rollout below a root."""

    if path is None:
        return None
    checkpoint_path = Path(path)
    if is_checkpoint is None:
        from megatron.core.dist_checkpointing import (
            check_is_distributed_checkpoint,
        )

        is_checkpoint = check_is_distributed_checkpoint
    if is_checkpoint(str(checkpoint_path)):
        return checkpoint_path
    if checkpoint_path.is_dir():
        candidates = sorted(
            checkpoint_path.glob("rollout_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"),
            reverse=True,
        )
        for candidate in candidates:
            if candidate.is_dir() and is_checkpoint(str(candidate)):
                return candidate
    raise ValueError(
        f"{checkpoint_path} is neither a Megatron distributed checkpoint nor "
        "a save root containing one."
    )


class MegatronTrainer(RayWorker):
    """Slim actor semantics executed by one MCore model-parallel rank."""

    def __init__(self, world_size: int, rank: int, master_addr: str | None, master_port: int | None):
        configure_logger()
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, int(master_port)
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

    def _init_distributed(self, args: Namespace) -> None:
        from slim.backends.fsdp_utils.fsdp_helpers import get_local_gpu_id

        local_rank = int(get_local_gpu_id())
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

    def _provider_config(self) -> dict[str, Any]:
        values: dict[str, Any] = {"calculate_per_token_loss": True}
        if self.args.gradient_checkpointing:
            values.update(
                recompute_granularity="full",
                recompute_method="uniform",
                recompute_num_layers=1,
            )
        return values

    def _optimizer_config(self) -> dict[str, Any]:
        return {
            "optimizer": self.args.optimizer,
            "lr": self.args.lr_actor,
            "min_lr": self.args.lr_min,
            "weight_decay": self.args.weight_decay,
            "adam_beta1": self.args.adam_beta1,
            "adam_beta2": self.args.adam_beta2,
            "adam_eps": self.args.adam_eps,
            "clip_grad": self.args.clip_grad,
        }

    def _scheduler_config(self) -> dict[str, Any]:
        num_rollout = self.args.num_rollout or 1
        train_steps = max(
            1,
            num_rollout
            * self.args.rollout_batch_size
            * self.args.n_samples_per_prompt
            // self.args.global_batch_size,
        )
        decay_iters = self.args.lr_decay_iters or train_steps
        return {
            "lr_decay_style": self.args.lr_decay_style,
            "lr_decay_steps": max(1, decay_iters * self.args.global_batch_size),
            "lr_warmup_steps": self.args.lr_warmup_iters * self.args.global_batch_size,
            "wd_incr_steps": max(1, decay_iters * self.args.global_batch_size),
        }

    def _configure_model_runtime(self) -> None:
        from megatron.core.distributed import finalize_model_grads
        from megatron.core.utils import get_model_config

        for model_chunk in self.model:
            config = get_model_config(model_chunk)
            config.calculate_per_token_loss = True
            config.finalize_model_grads_func = partial(
                finalize_model_grads,
                pg_collection=self.pg_collection,
            )
            config.grad_scale_func = self.optimizer.scale_loss

    def _checkpoint_metadata(self, rollout_id: int, next_rollout_id: int) -> CheckpointMetadata:
        topology = self.bundle.topology
        return CheckpointMetadata(
            rollout_id=rollout_id,
            next_rollout_id=next_rollout_id,
            global_step=self.global_step,
            world_size=topology.world_size,
            tensor_model_parallel_size=topology.tensor_model_parallel_size,
            pipeline_model_parallel_size=topology.pipeline_model_parallel_size,
            context_parallel_size=topology.context_parallel_size,
            expert_model_parallel_size=topology.expert_model_parallel_size,
            expert_tensor_parallel_size=topology.expert_tensor_parallel_size,
            sequence_parallel=topology.sequence_parallel,
        )

    def _restore_checkpoint(self) -> int:
        if self._checkpoint_load_dir is None or _is_hf_checkpoint(self._checkpoint_load_dir):
            return int(self.args.start_rollout_id or 0)

        expected = self._checkpoint_metadata(
            rollout_id=-1,
            next_rollout_id=int(self.args.start_rollout_id or 0),
        )
        loaded = self.checkpoint_manager.load(self._checkpoint_load_dir, expected)
        self.global_step = loaded.global_step
        return loaded.next_rollout_id

    def _setup_weight_streamer(self) -> None:
        from slim.backends.fsdp_utils.update_weight_utils import UpdateWeightFromDistributed

        quantizer = Quantizer.maybe_from_checkpoint(self.args.hf_checkpoint)
        transport = UpdateWeightFromDistributed(self.args, self.model[0], quantizer)
        self.weight_updater = MegatronWeightStreamer(
            bridge=self.bundle.bridge,
            model=self.model,
            transport=transport,
            max_bytes=self.args.update_weight_buffer_size,
            quantizer=quantizer,
        )

    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:
        """Initialize distributed state, Bridge model, optimizer, and checkpoint state."""

        if role != "actor":
            raise ValueError("The slim Megatron backend supports only the actor role.")
        if with_ref:
            raise ValueError("The slim Megatron backend does not support a reference model.")

        self.args = args
        self.role = role
        self.with_ref = with_ref
        self._init_distributed(args)

        if args.debug_rollout_only:
            self.dp_size = args.world_size
            self.dp_rank = args.rank
            self.train_parallel_config = {"dp_size": self.dp_size}
            return 0

        checkpoint_source = args.load if _is_hf_checkpoint(args.load) else args.hf_checkpoint
        self._checkpoint_load_dir = (
            None
            if args.load is None or _is_hf_checkpoint(args.load)
            else _resolve_distributed_checkpoint(args.load)
        )
        self._checkpoint_save_dir = args.save

        self.bundle: MegatronModelBundle = build_megatron_model(
            checkpoint_source,
            world_size=args.world_size,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            sequence_parallel=args.sequence_parallel,
            expert_model_parallel_size=args.expert_model_parallel_size,
            use_distributed_optimizer=args.use_distributed_optimizer,
            use_rollout_routing_replay=args.use_rollout_routing_replay,
            seed=args.seed,
            provider_config_kwargs=self._provider_config(),
            ddp_config_kwargs={"average_in_collective": False},
            optimizer_config_kwargs=self._optimizer_config(),
            scheduler_config_kwargs=self._scheduler_config(),
        )
        self.model = self.bundle.model
        self.optimizer = self.bundle.optimizer
        self.scheduler = self.bundle.scheduler
        self.pg_collection = self.bundle.pg_collection
        self.dp_size = self.bundle.topology.data_parallel_size
        self.dp_rank = self.pg_collection.dp.rank()
        self.train_parallel_config = {"dp_size": self.dp_size}
        self._configure_model_runtime()

        self.is_vlm = bool(getattr(self.bundle.provider, "modality_keys", None))
        self.placeholder_token_ids = {}
        if hasattr(self.bundle.provider, "image_token_id"):
            self.placeholder_token_ids["image"] = self.bundle.provider.image_token_id
        if hasattr(self.bundle.provider, "video_token_id"):
            self.placeholder_token_ids["video"] = self.bundle.provider.video_token_id
        self.pad_token_id = int(getattr(self.bundle.provider, "pad_token_id", 0) or 0)

        self.router_replay = (
            ModelScopedRouterReplay(self.model) if args.use_rollout_routing_replay else None
        )
        self.checkpoint_manager = MegatronCheckpointManager(
            self.model,
            self.optimizer,
            self.scheduler,
            use_distributed_optimizer=args.use_distributed_optimizer,
            pg_collection=self.pg_collection,
        )
        self.global_step = 0
        self.args.start_rollout_id = self._restore_checkpoint()
        self._pending_episodes: list[Any] | None = None
        self._setup_weight_streamer()

        if dist.get_rank() == 0:
            init_tracking(args, primary=False, disable_stats=True)
        return int(self.args.start_rollout_id)

    def set_rollout_manager(self, rollout_manager: Any) -> None:
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and dist.get_rank() == 0:
            ray.get(
                rollout_manager.set_train_parallel_config.remote(self.train_parallel_config)
            )

    def connect_actor_critic(self, _critic_handle: Any) -> None:
        raise ValueError("The slim Megatron backend does not support a critic.")

    def compute_values(self, _rollout_id: int, _rollout_data_ref: list) -> None:
        raise ValueError("The slim Megatron backend does not support a critic.")

    def clear_memory(self) -> None:
        if torch.cuda.is_available():
            clear_memory()

    def sleep(self) -> None:
        """Remain resident because Megatron colocated rollout is not supported."""

    def wake_up(self) -> None:
        """Remain resident because Megatron colocated rollout is not supported."""

    def _episode_microbatches(
        self,
        episodes: Sequence[Any],
        *,
        for_log_probs: bool,
    ) -> list[list[Any]]:
        if self.args.use_dynamic_batch_size:
            capacity = (
                self.args.log_probs_max_tokens_per_gpu
                if for_log_probs
                else self.args.max_tokens_per_gpu
            )
            topology = self.bundle.topology
            alignment = packed_sequence_alignment(
                tensor_model_parallel_size=topology.tensor_model_parallel_size,
                context_parallel_size=topology.context_parallel_size,
                sequence_parallel=topology.sequence_parallel,
            )
            batches = _split_by_capacity(
                episodes,
                capacity,
                alignment=alignment,
            )
            count = torch.tensor(
                len(batches),
                dtype=torch.int,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(
                count,
                op=dist.ReduceOp.MAX,
                group=self.pg_collection.dp,
            )
            return _rebalance_microbatches(batches, int(count.item()))
        return [
            list(episodes[index : index + self.args.micro_batch_size])
            for index in range(0, len(episodes), self.args.micro_batch_size)
        ]

    def _build_layout(self, episodes: Sequence[Any]) -> PackedLayout:
        return build_packed_layout(
            episodes,
            tensor_model_parallel_size=self.bundle.topology.tensor_model_parallel_size,
            context_parallel_size=self.bundle.topology.context_parallel_size,
            sequence_parallel=self.bundle.topology.sequence_parallel,
            pad_token_id=self.pad_token_id,
            placeholder_token_ids=self.placeholder_token_ids or None,
        )

    def _prepare_batch(self, episodes: Sequence[Any]) -> _PreparedBatch:
        from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

        layout = self._build_layout(episodes)
        device = torch.device("cuda", torch.cuda.current_device())

        tensors: dict[str, torch.Tensor] = {
            "tokens": layout.tokens.to(device, non_blocking=True),
            "labels": layout.labels.to(device, non_blocking=True),
            "loss_mask": layout.loss_mask.to(device, non_blocking=True),
            "position_ids": layout.position_ids.to(device, non_blocking=True),
            "token_mask": layout.token_mask.to(device, non_blocking=True),
            "edge_mask": layout.edge_mask.to(device, non_blocking=True),
            "cu_seqlens_q": layout.cu_seqlens_q.to(device, non_blocking=True),
            "cu_seqlens_kv": layout.cu_seqlens_kv.to(device, non_blocking=True),
            "max_seqlen_q": layout.max_seqlen_q.to(device, non_blocking=True),
            "max_seqlen_kv": layout.max_seqlen_kv.to(device, non_blocking=True),
        }
        if layout.cu_seqlens_q_padded is not None:
            tensors["cu_seqlens_q_padded"] = layout.cu_seqlens_q_padded.to(
                device, non_blocking=True
            )
        if layout.cu_seqlens_kv_padded is not None:
            tensors["cu_seqlens_kv_padded"] = layout.cu_seqlens_kv_padded.to(
                device, non_blocking=True
            )
        tensors.update(
            {
                name: value.to(device, non_blocking=True)
                for name, value in layout.edge_features.items()
            }
        )
        if layout.rollout_routed_experts is not None:
            tensors["rollout_routed_experts"] = layout.rollout_routed_experts.to(
                device, non_blocking=True
            )

        packed_metadata = {
            key: tensors[key]
            for key in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
                "max_seqlen_q",
                "max_seqlen_kv",
            )
            if key in tensors
        }
        packed_metadata["total_tokens"] = layout.total_tokens
        packed_seq_params = get_packed_seq_params(packed_metadata)
        packed_seq_params.cp_group = self.pg_collection.cp

        cp_indices = layout.get_context_parallel_indices(
            cp_size=self.bundle.topology.context_parallel_size,
            cp_rank=self.pg_collection.cp.rank(),
            device=device,
            cp_group=self.pg_collection.cp,
        )

        sequence_ids = torch.empty_like(tensors["tokens"], dtype=torch.long)
        sequence_active_counts = torch.empty(
            len(layout.trajectories),
            dtype=torch.float32,
            device=device,
        )
        for trajectory in layout.trajectories:
            sequence_ids[
                :,
                trajectory.physical_tokens.start : trajectory.physical_tokens.stop,
            ] = trajectory.episode_index
            sequence_active_counts[trajectory.episode_index] = tensors["loss_mask"][
                :,
                trajectory.physical_tokens.start : trajectory.physical_tokens.stop,
            ].sum()

        visual_kwargs: dict[str, torch.Tensor] = {}
        if layout.visual_inputs is not None:
            visual_kwargs = {
                key: value.to(device, non_blocking=True)
                for key, value in layout.visual_inputs.normalized_for_model().items()
            }

        return _PreparedBatch(
            layout=layout,
            episodes=episodes,
            tensors=tensors,
            visual_kwargs=visual_kwargs,
            packed_seq_params=packed_seq_params,
            cp_indices=cp_indices,
            sequence_ids=sequence_ids,
            sequence_active_counts=sequence_active_counts,
        )

    def _prepare_batches(
        self,
        episodes: Sequence[Any],
        *,
        for_log_probs: bool,
    ) -> list[_PreparedBatch]:
        return [
            self._prepare_batch(microbatch)
            for microbatch in self._episode_microbatches(
                episodes,
                for_log_probs=for_log_probs,
            )
        ]

    @staticmethod
    def _select_cp(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(1, indices)

    def _local_routes(self, batch: _PreparedBatch) -> torch.Tensor | None:
        routes = batch.tensors.get("rollout_routed_experts")
        if routes is None:
            return None
        routes = self._select_cp(routes, batch.cp_indices).squeeze(0)
        if self.bundle.topology.sequence_parallel:
            tp_size = self.bundle.topology.tensor_model_parallel_size
            tp_rank = self.pg_collection.tp.rank()
            routes = torch.tensor_split(routes, tp_size, dim=0)[tp_rank].contiguous()
        return routes

    def _model_forward(
        self,
        model: Any,
        batch: _PreparedBatch,
        *,
        training: bool,
        replay_routes: bool,
    ) -> torch.Tensor:
        tensors = batch.tensors
        if self.is_vlm:
            tokens = tensors["tokens"]
            position_ids = tensors["position_ids"]
        else:
            tokens = self._select_cp(tensors["tokens"], batch.cp_indices)
            position_ids = self._select_cp(tensors["position_ids"], batch.cp_indices)

        if replay_routes and self.router_replay is not None:
            routes = self._local_routes(batch)
            if routes is None:
                raise ValueError(
                    "Routing replay is enabled but the batch has no rollout_routed_experts."
                )
            self.router_replay.prepare_forward(routes)

        output = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch.packed_seq_params,
            runtime_gather_output=False,
            **batch.visual_kwargs,
        )
        if isinstance(output, tuple):
            output = output[0]
        if training and replay_routes and self.router_replay is not None:
            self.router_replay.register_backward_hook(output)
        return output

    def _local_policy_inputs(
        self,
        logits: torch.Tensor,
        batch: _PreparedBatch,
        *,
        global_active_tokens: int,
        global_num_sequences: int,
    ) -> dict[str, torch.Tensor]:
        tensors = batch.tensors
        indices = batch.cp_indices
        labels = self._select_cp(tensors["labels"], indices)
        safe_labels = labels.masked_fill(labels < 0, 0)
        entropy = None
        if self.args.entropy_coef != 0.0:
            entropy = vocab_parallel_entropy(
                logits,
                tp_group=self.pg_collection.tp,
            )
        current_log_probs = selected_log_probs(
            logits,
            safe_labels,
            tp_group=self.pg_collection.tp,
            temperature=self.args.rollout_temperature,
        )
        loss_mask = self._select_cp(tensors["loss_mask"], indices)
        sequence_ids = self._select_cp(batch.sequence_ids, indices)
        weights = objective_weights(
            loss_mask=loss_mask,
            sequence_ids=sequence_ids,
            sequence_active_counts=batch.sequence_active_counts,
            global_active_tokens=global_active_tokens,
            global_num_sequences=global_num_sequences,
            calculate_per_token_loss=self.args.calculate_per_token_loss,
        )
        sample_mean_weights = objective_weights(
            loss_mask=loss_mask,
            sequence_ids=sequence_ids,
            sequence_active_counts=batch.sequence_active_counts,
            global_active_tokens=global_active_tokens,
            global_num_sequences=global_num_sequences,
            calculate_per_token_loss=False,
        )
        old_key = (
            "actor_old_log_probs"
            if self.args.old_logprob_source == "actor"
            else "rollout_log_probs"
        )
        if old_key not in tensors:
            raise KeyError(f"{old_key} is required for Megatron policy training.")

        result = {
            "current_log_probs": current_log_probs,
            "old_log_probs": self._select_cp(tensors[old_key], indices),
            "advantages": self._select_cp(tensors["advantages"], indices),
            "loss_mask": loss_mask,
            "objective_weight": weights,
            "sample_mean_weight": sample_mean_weights,
        }
        if entropy is not None:
            result["entropy"] = entropy
        return result

    def _forward_step(
        self,
        *,
        global_active_tokens: int,
        global_num_sequences: int,
    ):
        def forward_step(data_iterator, model, _checkpoint_activations_microbatch=None):
            batch = next(data_iterator)
            output = self._model_forward(
                model,
                batch,
                training=True,
                replay_routes=self.router_replay is not None,
            )

            def loss_func(logits):
                inputs = self._local_policy_inputs(
                    logits,
                    batch,
                    global_active_tokens=global_active_tokens,
                    global_num_sequences=global_num_sequences,
                )
                result = policy_loss(
                    **inputs,
                    eps_clip=self.args.eps_clip,
                    eps_clip_high=self.args.eps_clip_high,
                    policy_surrogate=self.args.policy_surrogate,
                    eps_clip_c=self.args.eps_clip_c,
                    entropy_coef=self.args.entropy_coef,
                    kl_loss_coef=0.0,
                )
                metrics = {"loss": result.loss.detach(), **result.metrics}
                return (
                    result.loss,
                    result.num_active_tokens.to(dtype=torch.int),
                    metrics,
                )

            return output, loss_func

        return forward_step

    def _log_prob_forward_step(self):
        def forward_step(data_iterator, model, _checkpoint_activations_microbatch=None):
            batch = next(data_iterator)
            output = self._model_forward(
                model,
                batch,
                training=False,
                replay_routes=self.router_replay is not None,
            )

            def collect(logits, non_loss_data=False):
                if not non_loss_data:
                    raise RuntimeError("Log-probability forward requires non-loss collection.")
                labels = self._select_cp(batch.tensors["labels"], batch.cp_indices)
                labels = labels.masked_fill(labels < 0, 0)
                local = selected_log_probs(
                    logits,
                    labels,
                    tp_group=self.pg_collection.tp,
                    temperature=self.args.rollout_temperature,
                )
                full = torch.zeros_like(batch.tensors["loss_mask"], dtype=local.dtype)
                full.index_copy_(1, batch.cp_indices, local)
                if self.pg_collection.cp.size() > 1:
                    dist.all_reduce(full, group=self.pg_collection.cp)
                return {"log_probs": full}

            return output, collect

        return forward_step

    def _run_schedule(
        self,
        batches: Sequence[_PreparedBatch],
        *,
        forward_step_func,
        forward_only: bool,
        collect_non_loss_data: bool = False,
    ):
        if not batches:
            raise ValueError("Megatron schedule requires at least one microbatch.")
        max_sequence_length = max(batch.layout.total_tokens for batch in batches)
        return run_forward_backward(
            pp_size=self.bundle.topology.pipeline_model_parallel_size,
            model=self.model,
            forward_step_func=forward_step_func,
            data_iterator=iter(batches),
            num_microbatches=len(batches),
            seq_length=max_sequence_length,
            decoder_seq_length=max_sequence_length,
            micro_batch_size=1,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            pg_collection=self.pg_collection,
        )

    def _attach_log_probs(
        self,
        batches: Sequence[_PreparedBatch],
        outputs: Sequence[dict[str, torch.Tensor]],
        *,
        attribute: str,
    ) -> None:
        if self.pg_collection.pp.rank() != self.pg_collection.pp.size() - 1:
            return
        if len(outputs) != len(batches):
            raise RuntimeError(
                f"Expected {len(batches)} log-probability outputs, received {len(outputs)}."
            )
        for batch, output in zip(batches, outputs, strict=True):
            full = output["log_probs"].detach().cpu()
            for trajectory in batch.layout.trajectories:
                values = full[
                    0,
                    trajectory.edges.start : trajectory.edges.stop,
                ].clone()
                setattr(batch.episodes[trajectory.episode_index], attribute, values)

    def _compute_actor_old_log_probs(self, episodes: Sequence[Any]) -> None:
        batches = self._prepare_batches(episodes, for_log_probs=True)
        self.model[0].eval()
        try:
            with torch.no_grad():
                outputs = self._run_schedule(
                    batches,
                    forward_step_func=self._log_prob_forward_step(),
                    forward_only=True,
                    collect_non_loss_data=True,
                )
            self._attach_log_probs(
                batches,
                outputs,
                attribute="_actor_old_log_probs",
            )
        finally:
            self.model[0].train()
            if self.router_replay is not None:
                self.router_replay.cleanup()

    def compute_log_probs(self, _rollout_id: int, rollout_data_ref: list) -> None:
        """Compute actor-old log probabilities and cache the local episode partition."""

        episodes = process_rollout_data(
            self.args,
            rollout_data_ref,
            self.dp_rank,
            self.dp_size,
        )
        if (
            not self.args.debug_rollout_only
            and self.args.old_logprob_source == "actor"
        ):
            self._compute_actor_old_log_probs(episodes)
        self._pending_episodes = episodes

    def _global_step_stats(
        self,
        episodes: Sequence[Any],
    ) -> tuple[int, int]:
        local_active = sum(
            int(torch.as_tensor(episode.loss_mask).sum().item()) for episode in episodes
        )
        stats = torch.tensor(
            [local_active, len(episodes)],
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(stats, group=self.pg_collection.dp)
        return int(stats[0].item()), int(stats[1].item())

    def _aggregate_metrics(
        self,
        schedule_output: Sequence[dict[str, torch.Tensor]],
        *,
        global_active_tokens: int,
    ) -> dict[str, float] | None:
        metrics: dict[str, float] | None = None
        if self.pg_collection.pp.rank() == self.pg_collection.pp.size() - 1:
            names = schedule_output[0].keys() if schedule_output else ()
            reduced = {
                name: torch.stack([item[name] for item in schedule_output]).sum()
                for name in names
            }
            for value in reduced.values():
                dist.all_reduce(value, group=self.pg_collection.dp_cp)
            denominator = max(global_active_tokens, 1)
            metrics = {
                name: float((value / denominator).item())
                for name, value in reduced.items()
            }

        pp_ranks = dist.get_process_group_ranks(self.pg_collection.pp)
        payload: list[Any] = [metrics]
        dist.broadcast_object_list(
            payload,
            src=pp_ranks[-1],
            group=self.pg_collection.pp,
        )
        return payload[0]

    def _optimizer_step(
        self,
        episodes: Sequence[Any],
        *,
        rollout_id: int,
    ) -> None:
        del rollout_id
        batches = self._prepare_batches(episodes, for_log_probs=False)
        global_active_tokens, global_num_sequences = self._global_step_stats(episodes)

        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
            model_chunk.train()
        self.optimizer.zero_grad()

        schedule_completed = False
        try:
            schedule_output = self._run_schedule(
                batches,
                forward_step_func=self._forward_step(
                    global_active_tokens=global_active_tokens,
                    global_num_sequences=global_num_sequences,
                ),
                forward_only=False,
            )
            schedule_completed = True
            update_successful, grad_norm, _num_zeros = self.optimizer.step()
            if _as_bool(update_successful):
                self.scheduler.step(increment=global_num_sequences)
                self.global_step += 1

            metrics = self._aggregate_metrics(
                schedule_output,
                global_active_tokens=global_active_tokens,
            )
            if dist.get_rank() == 0 and metrics is not None:
                log_dict = {
                    f"train/actor/{name}": value for name, value in metrics.items()
                }
                log_dict["train/actor/grad_norm"] = (
                    float(grad_norm) if grad_norm is not None else 0.0
                )
                log_dict["train/step"] = self.global_step
                logger.info("Megatron actor step %s: %s", self.global_step, log_dict)
                logging_utils.log(self.args, log_dict)
        finally:
            if self.router_replay is not None:
                if schedule_completed and self.args.gradient_checkpointing:
                    self.router_replay.assert_replay_consumed()
                self.router_replay.cleanup()

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: list,
        values_refs: list | None = None,
    ) -> None:
        """Run one or more actor optimizer steps over a rollout partition."""

        if values_refs is not None:
            raise ValueError("The slim Megatron backend does not support critic values.")
        if self._pending_episodes is not None:
            episodes = self._pending_episodes
            self._pending_episodes = None
        else:
            episodes = process_rollout_data(
                self.args,
                rollout_data_ref,
                self.dp_rank,
                self.dp_size,
            )
        if self.args.debug_rollout_only:
            return

        if self.args.advantage_estimator != "grpo":
            raise ValueError("The slim Megatron backend currently supports GRPO only.")
        for episode in episodes:
            episode._advantages = [episode.reward] * episode.num_edges
            episode._returns = list(episode._advantages)

        if self.args.old_logprob_source == "actor" and any(
            getattr(episode, "_actor_old_log_probs", None) is None
            for episode in episodes
        ):
            self._compute_actor_old_log_probs(episodes)

        if self.args.global_batch_size % self.dp_size:
            raise ValueError(
                "global_batch_size must be divisible by the Megatron data-parallel size."
            )
        local_batch_size = self.args.global_batch_size // self.dp_size
        if len(episodes) % local_batch_size:
            raise ValueError(
                f"Local episode count {len(episodes)} is not divisible by local batch size "
                f"{local_batch_size}."
            )
        for start in range(0, len(episodes), local_batch_size):
            self._optimizer_step(
                episodes[start : start + local_batch_size],
                rollout_id=rollout_id,
            )
        clear_memory()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Save one MCore distributed checkpoint for the completed rollout."""

        del force_sync
        if self.args.debug_rollout_only or self._checkpoint_save_dir is None:
            return
        checkpoint_path = (
            Path(self._checkpoint_save_dir) / f"rollout_{rollout_id:08d}"
        )
        self.checkpoint_manager.save(
            checkpoint_path,
            self._checkpoint_metadata(
                rollout_id=rollout_id,
                next_rollout_id=rollout_id + 1,
            ),
        )

    def update_weights(self) -> None:
        """Stream Bridge-converted actor weights to the active SGLang engines."""

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.rollout_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(
                    self.rollout_manager.recover_and_get_updatable_engines.remote()
                )
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
        ) = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(
                    self.rollout_manager.clear_updatable_num_new_engines.remote()
                )

        self.weight_updater.update_weights()
        if self.args.ci_test and rollout_engines:
            engine_version = ray.get(
                rollout_engines[0].get_weight_version.remote()
            )
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    "SGLang and Megatron weight versions differ: "
                    f"{engine_version} != {self.weight_updater.weight_version}."
                )
        clear_memory()


__all__ = ["MegatronTrainer"]
