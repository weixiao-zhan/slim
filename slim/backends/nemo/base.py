# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Ray lifecycle for NeMo AutoModel actor and critic trainers."""

from __future__ import annotations

import contextlib
import logging
import os
import random
from argparse import Namespace
from contextlib import ExitStack
from datetime import timedelta
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.training.utils import (
    get_expert_tp_replication_factor,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from transformers import AutoConfig

import slim.utils.eval_config
from slim.ray.ray_worker import RayWorker
from slim.utils import logging_utils, train_dump_utils, train_metric_utils
from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group, init_gloo_group
from slim.utils.logging_utils import configure_logger, init_tracking
from slim.utils.memory_utils import clear_memory
from slim.utils.processing_utils import load_processor, load_tokenizer
from slim.utils.profile_utils import TrainProfiler
from slim.utils.timer import Timer, inverse_timer, timer, with_defer
from slim.utils.trajectory_batch import TrajectoryBatch

from . import checkpoint
from .data_packing import (
    build_token_budget_partitions,
    pack_sequences,
    unpack_sequences,
    update_packed_targets,
)
from .loss import count_global_denominators
from .lr_scheduler import get_lr_scheduler
from .models import validate_config
from .topology import NeMoTopology

logger = logging.getLogger(__name__)


def _assigned_cuda_device(rank: int) -> int:
    device_count = torch.cuda.device_count()
    assigned_gpu_ids = ray.get_gpu_ids()
    if not assigned_gpu_ids:
        return rank % max(device_count, 1)
    if device_count == 1:
        return 0

    device = int(assigned_gpu_ids[0])
    if not 0 <= device < device_count:
        raise RuntimeError(
            f"Ray assigned GPU {device}, but this worker sees {device_count} CUDA devices."
        )
    return device


def _bind_default_process_group_device(device: torch.device) -> None:
    binding = torch.zeros(1, device=device)
    dist.all_reduce(binding)


def _clear_inactive_optimizer_grads(
    optimizer: torch.optim.Optimizer,
    lr_step: int,
) -> None:
    for group in optimizer.param_groups:
        if lr_step >= group.get("start_step", 0):
            continue
        for parameter in group["params"]:
            parameter.grad = None


def _move_optimizer(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def _move_module(module: torch.nn.Module, device: str | torch.device) -> None:
    for buffer in module.buffers():
        buffer.data = buffer.data.to(device=device, non_blocking=True)
    module.to(device=device, non_blocking=True)


class NeMoTrainer(RayWorker):
    """Compose Slim's rollout lifecycle with NeMo distributed components."""

    _train_log_prefix = "train"

    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(_assigned_cuda_device(rank))

    @property
    def model_parts(self) -> list[torch.nn.Module]:
        return [self.model]

    def _steps_per_rollout(self) -> int:
        return self.args.rollout_batch_size * self.args.n_samples_per_prompt // self.args.global_batch_size

    def _init_distributed(self, args: Namespace) -> None:
        self.args = args
        torch.serialization.add_safe_globals([slim.utils.eval_config.EvalDatasetConfig])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        _bind_default_process_group_device(torch.device("cuda", local_rank))
        init_gloo_group()
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

    def _setup_topology(self) -> None:
        self.topology = NeMoTopology.from_args(self.args, dist.get_world_size())
        self.distributed_setup = self.topology.build(self.args)
        self.mesh_context = self.distributed_setup.mesh_context
        self.mesh_context.process_group = get_gloo_group()
        self.device_mesh = self.mesh_context.device_mesh
        self.moe_mesh = self.mesh_context.moe_mesh
        self.dp_mesh = self.device_mesh["dp_shard"]
        self.cp_mesh = self.device_mesh["cp"]
        self.dp_group = self.dp_mesh.get_group()
        self.cp_group = self.cp_mesh.get_group()
        self.dp_size = self.dp_mesh.size()
        self.dp_rank = self.dp_mesh.get_local_rank()
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = self.cp_mesh.get_local_rank()
        self.backward_group_size = dist.get_world_size()
        logger.info(
            "NeMo topology rank=%d world=%d logical_dp=%d dp_rank=%d cp=%d cp_rank=%d ep=%d",
            dist.get_rank(),
            dist.get_world_size(),
            self.dp_size,
            self.dp_rank,
            self.cp_size,
            self.cp_rank,
            self.topology.expert_model_parallel_size,
        )

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace) -> int:  # type: ignore[override]
        self._init_distributed(args)
        self._setup_topology()
        torch.manual_seed(args.seed)
        self.train_parallel_config = {
            "dp_size": self.dp_size,
            "cp_size": self.cp_size,
            "ep_size": self.topology.expert_model_parallel_size,
        }
        if args.debug_rollout_only:
            return 0

        self._need_offload = args.rollout_colocate or args.critic_colocate
        if dist.get_rank() == 0:
            init_tracking(args, primary=False, disable_stats=True)
        if args.start_rollout_id is None:
            args.start_rollout_id = 0
        self.prof = TrainProfiler(args)

        model_checkpoint = self._resolve_checkpoint_paths()
        self.processor = None
        for loading_rank in range(dist.get_world_size()):
            if loading_rank == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(model_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(model_checkpoint, trust_remote_code=True)
                if getattr(self.hf_config, "vision_config", None) is not None:
                    self.processor = load_processor(model_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        validate_config(self.hf_config, self.topology)
        self.global_step = 0
        self._create_model_and_optimizer(model_checkpoint)
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.checkpointer = checkpoint.build_checkpointer(self)
        self._pending_checkpoint = None
        checkpoint_payload = checkpoint.load(self)
        self._post_model_setup()

        self._precomputed_packed_data: tuple[list[dict], list[int]] | None = None
        checkpoint.finalize_load(self, checkpoint_payload)
        self.sleep()
        self.prof.on_init_end()
        return int(args.start_rollout_id)

    def _resolve_checkpoint_paths(self) -> str:
        raise NotImplementedError

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        raise NotImplementedError

    def _post_model_setup(self) -> None:
        pass

    @property
    def checkpoint_model(self) -> torch.nn.Module:
        return getattr(self, "_checkpoint_model", self.model)

    def _run_train_loop(self, packed_batches: list, grad_accum: list[int]) -> None:
        raise NotImplementedError

    def _maybe_update_ref_model(self, rollout_id: int) -> None:
        pass

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and dist.get_rank() == 0:
            ray.get(rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))

    @timer
    def sleep(self) -> None:
        if not self._need_offload or self.args.debug_rollout_only:
            return
        for model_part in self.model_parts:
            _move_module(model_part, "cpu")
        _move_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())

    @timer
    def wake_up(self) -> None:
        if not self._need_offload or self.args.debug_rollout_only:
            return
        device = torch.device("cuda", torch.cuda.current_device())
        for model_part in self.model_parts:
            _move_module(model_part, device)
        _move_optimizer(self.optimizer, device)
        dist.barrier(group=get_gloo_group())

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self._checkpoint_save_dir is None:
            return
        checkpoint.save(self, rollout_id, force_sync=force_sync)

    def _packed_data(self, batch: TrajectoryBatch) -> tuple[list[dict], list[int]]:
        """Pack each optimizer step's trajectories into DP-synchronized microbatches.

        The flattener already gave every rank the same number of trajectories per
        step with balanced token sums, so the MAX all-reduce below finds a pack
        count every rank can reach.
        """
        packed_batches = []
        step_microbatch_counts = []
        for start, step_trajectories in batch.steps():
            lengths = [len(trajectory.token_ids) for trajectory in step_trajectories]
            if self.args.use_dynamic_batch_size:
                physical_pack_token_budget = self.args.max_tokens_per_gpu * self.cp_size
                partitions = build_token_budget_partitions(lengths, physical_pack_token_budget)
            else:
                partitions = [
                    list(range(offset, min(offset + self.args.micro_batch_size, len(step_trajectories))))
                    for offset in range(0, len(step_trajectories), self.args.micro_batch_size)
                ]

            count = torch.tensor(len(partitions), dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.dp_group)
            pack_count = int(count.item())
            if self.args.use_dynamic_batch_size:
                partitions = build_token_budget_partitions(
                    lengths,
                    physical_pack_token_budget,
                    num_packs=pack_count,
                )
            elif len(partitions) != pack_count:
                raise RuntimeError("fixed microbatch counts differ across data-parallel ranks")

            step_batches = pack_sequences(step_trajectories, partitions=partitions)
            if len(step_batches) != pack_count:
                raise RuntimeError(f"requested {pack_count} synchronized packs but built {len(step_batches)}")
            for pack in step_batches:
                pack["_document_indices"] = [index + start for index in pack["_document_indices"]]
            packed_batches.extend(step_batches)
            step_microbatch_counts.append(pack_count)
        return packed_batches, list(accumulate(step_microbatch_counts))

    def _cache_packed_data(self, packed_batches: list[dict], grad_accum: list[int]) -> None:
        if self._precomputed_packed_data is not None:
            raise RuntimeError("precomputed packed data is already cached")
        self._precomputed_packed_data = (packed_batches, grad_accum)

    def _take_packed_data(self, batch: TrajectoryBatch) -> tuple[list[dict], list[int]]:
        if self._precomputed_packed_data is None:
            return self._packed_data(batch)
        packed_batches, grad_accum = self._precomputed_packed_data
        self._precomputed_packed_data = None
        update_packed_targets(packed_batches, batch.trajectories)
        return packed_batches, grad_accum

    @staticmethod
    def _optimizer_step_batches(packed_batches: list[dict], boundaries: list[int]):
        start = 0
        for end in boundaries:
            yield packed_batches[start:end]
            start = end
        if start != len(packed_batches):
            raise ValueError("gradient accumulation boundaries do not cover all packed batches")

    def train(self, rollout_id: int, rollout_data_ref: list) -> None:
        if self.args.debug_rollout_only:
            return
        self.wake_up()
        with inverse_timer("train_wait"), timer("train"):
            batch = process_rollout_data(rollout_data_ref, self.dp_rank, self.dp_size)
            packed_batches, grad_accum = self._take_packed_data(batch)
            self._train_core(rollout_id, packed_batches, grad_accum)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )
        self.sleep()
        clear_memory()

    def _train_core(self, rollout_id, packed_batches: list[dict], grad_accum: list[int]) -> None:
        if not grad_accum:
            raise ValueError("training produced no microbatches")
        self._run_train_loop(packed_batches, grad_accum)
        self.prof.step(rollout_id=rollout_id)
        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=None)
        self._maybe_update_ref_model(rollout_id)

    def _begin_gradient_accumulation(self) -> None:
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=False)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            float(self.backward_group_size),
            device=torch.cuda.current_device(),
        )

    def _prepare_final_backward(self) -> None:
        prepare_for_final_backward(self.model_parts, pp_enabled=False)

    def _after_first_microbatch(self) -> None:
        prepare_after_first_microbatch()

    @contextlib.contextmanager
    def _sync_context(self, is_final_microbatch: bool):
        with ExitStack() as stack:
            for model_part in self.model_parts:
                stack.enter_context(
                    get_sync_ctx(
                        model_part,
                        is_final_microbatch,
                        defer_fsdp_grad_sync=self.args.defer_fsdp_grad_sync,
                    )
                )
            yield

    def _optimizer_step(self) -> float:
        self.checkpointer.maybe_wait_for_staging()
        self._last_step_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        _clear_inactive_optimizer_grads(self.optimizer, self.lr_scheduler.last_epoch)
        ep_axis_name = None
        if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names:
            ep_axis_name = "ep"
        grad_norm = scale_grads_and_clip_grad_norm(
            self.args.clip_grad,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name=ep_axis_name,
            foreach=True,
            num_label_tokens=1,
            dp_group_size=self.backward_group_size,
            expert_tp_replication_factor=get_expert_tp_replication_factor(self.model_parts, self.device_mesh),
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        for model_part in self.model_parts:
            if hasattr(model_part, "update_moe_gate_bias"):
                model_part.update_moe_gate_bias()
        self.global_step += 1
        return float(grad_norm)

    def _reduce_metrics(self, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        reduced = {}
        for key, value in metrics.items():
            tensor = value.detach().float().clone()
            dist.all_reduce(tensor)
            reduced[key] = tensor.item()
        return reduced

    def _log_step(self, metrics: dict[str, float], grad_norm: float) -> None:
        if dist.get_rank() != 0:
            return
        log_dict = {f"{self._train_log_prefix}/{key}": value for key, value in metrics.items()}
        log_dict[f"{self._train_log_prefix}/grad_norm"] = grad_norm
        for index, lr in enumerate(self._last_step_lrs):
            log_dict[f"{self._train_log_prefix}/lr-pg_{index}"] = lr
        log_dict["train/step"] = self.global_step
        logger.info("%s step %d: %s", self._train_log_prefix, self.global_step, log_dict)
        logging_utils.log(self.args, log_dict)

    def _log_packed_metrics(self, packed_batches, metric_keys):
        """Average each metric over documents, weighted by the loss weights."""
        log_dict = {}
        weight_sum, _ = count_global_denominators(
            packed_batches,
            self.dp_group,
            torch.cuda.current_device(),
        )
        for metric_key in metric_keys:
            if metric_key not in packed_batches[0]:
                continue
            value = torch.zeros((), device=torch.cuda.current_device())
            for packed_batch in packed_batches:
                for document in unpack_sequences(packed_batch):
                    metric = document.get(metric_key)
                    if isinstance(metric, torch.Tensor):
                        mask = document["loss_masks"].to(value.device)
                        value += (
                            document["loss_weights"]
                            * (metric.to(value.device) * mask).sum()
                            / mask.sum().clamp_min(1)
                        )
            dist.all_reduce(value, group=self.dp_group)
            log_dict[f"{self._train_log_prefix}/{metric_key}"] = (value / weight_sum).item()
        if dist.get_rank() == 0 and log_dict:
            log_dict["train/step"] = self.global_step
            logging_utils.log(self.args, log_dict)

    def update_weights(self) -> None:
        pass
