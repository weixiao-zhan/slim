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
from transformers import AutoConfig

import slim.utils.eval_config
from slim.ray.ray_worker import RayWorker
from slim.utils import logging_utils, train_dump_utils, train_metric_utils
from slim.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slim.utils.distributed_utils import get_gloo_group, init_gloo_group
from slim.utils.logging_utils import configure_logger, init_tracking
from slim.utils.memory_utils import clear_memory
from slim.utils.misc import load_function
from slim.utils.ppo_utils import vanilla_gae
from slim.utils.processing_utils import load_processor, load_tokenizer
from slim.utils.profile_utils import TrainProfiler
from slim.utils.timer import Timer, inverse_timer, timer, with_defer
from slim.utils.types import Episode

from . import checkpoint
from .data_packing import pack_sequences, unpack_sequences, update_packed_advantages
from .grad_clip import clip_cpu_offloaded_grad_norm
from .lr_scheduler import get_lr_scheduler
from .model import is_moe_config, validate_model_config
from .topology import NeMoTopology, flat_mesh, mesh_rank

logger = logging.getLogger(__name__)


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
        self._world_size = world_size
        self._rank = rank
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
        os.environ["LOCAL_RANK"] = str(rank % max(torch.cuda.device_count(), 1))

    @property
    def model_parts(self) -> list[torch.nn.Module]:
        return [self.model]

    def _init_distributed(self, args: Namespace, role: str, with_ref: bool) -> None:
        self.args = args
        self.role = role
        self.with_ref = with_ref
        torch.serialization.add_safe_globals([slim.utils.eval_config.EvalDatasetConfig])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
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
        self.dp_mesh = flat_mesh(self.device_mesh, "dp")
        self.dp_cp_mesh = flat_mesh(self.device_mesh, "dp_cp")
        self.cp_mesh = self.device_mesh["cp"]
        self.dp_group = self.dp_mesh.get_group()
        self.dp_cp_group = self.dp_cp_mesh.get_group()
        self.cp_group = self.cp_mesh.get_group()
        self.dp_size = self.dp_mesh.size()
        self.dp_rank = mesh_rank(self.dp_mesh)
        self.dp_cp_rank = mesh_rank(self.dp_cp_mesh)
        self.cp_size = self.cp_mesh.size()
        self.cp_rank = mesh_rank(self.cp_mesh)
        self.backward_group_size = self.dp_cp_mesh.size()
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
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        self._init_distributed(args, role, with_ref)
        self._setup_topology()
        torch.manual_seed(args.seed)
        self.train_parallel_config = {
            "dp_size": self.dp_size,
            "cp_size": self.cp_size,
            "ep_size": self.topology.expert_model_parallel_size,
        }
        if args.debug_rollout_only:
            return 0

        self._need_offload = (args.rollout_colocate or args.critic_colocate) and not args.nemo_cpu_offload
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

        validate_model_config(self.hf_config, self.topology)
        self.is_moe = is_moe_config(self.hf_config)
        self.global_step = 0
        self.micro_step = 0
        self._create_model_and_optimizer(model_checkpoint)
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.checkpointer = checkpoint.build_checkpointer(self)
        checkpoint_payload = checkpoint.load(self)
        self._post_model_setup()

        self.critic_handle = None
        self._pending_episodes = None
        self._pending_packed_batches = None
        self._pending_grad_accum = None
        self.rollout_data_postprocess = (
            load_function(args.rollout_data_postprocess_path)
            if args.rollout_data_postprocess_path
            else None
        )
        checkpoint.finalize_load(self, checkpoint_payload)
        self.max_tokens_per_gpu = args.max_tokens_per_gpu
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

    def _run_train_loop(self, rollout_id: int, packed_batches: list, grad_accum: list[int]) -> None:
        raise NotImplementedError

    def _maybe_update_ref_model(self, rollout_id: int) -> None:
        pass

    def connect_actor_critic(self, critic_handle) -> None:
        self.critic_handle = critic_handle

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and dist.get_rank() == 0:
            ray.get(rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))

    def clear_memory(self):
        if not self.args.debug_rollout_only:
            clear_memory()

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

    def _packed_data(self, episodes: list[Episode]) -> tuple[list[dict], list[int]]:
        packed_batches = []
        local_batch_size = self.args.global_batch_size // self.dp_size
        if self.args.global_batch_size % self.dp_size:
            raise ValueError(
                f"global_batch_size {self.args.global_batch_size} must be divisible by logical DP {self.dp_size}"
            )

        step_microbatch_counts = []
        for start in range(0, len(episodes), local_batch_size):
            chunk = episodes[start : start + local_batch_size]
            if self.args.use_dynamic_batch_size:
                physical_pack_token_budget = self.args.max_tokens_per_gpu * self.cp_size
                pack_count = get_minimum_num_micro_batch_size(
                    [len(episode.tokens) for episode in chunk],
                    physical_pack_token_budget,
                )
            else:
                pack_count = max(1, len(chunk) // self.args.micro_batch_size)

            count = torch.tensor(pack_count, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.dp_group)
            pack_count = int(count.item())
            chunk_batches = pack_sequences(chunk, num_packs=pack_count)
            if len(chunk_batches) != pack_count:
                raise RuntimeError(f"requested {pack_count} synchronized packs but built {len(chunk_batches)}")
            for batch in chunk_batches:
                batch["_episode_indices"] = [index + start for index in batch["_episode_indices"]]
            packed_batches.extend(chunk_batches)
            step_microbatch_counts.append(pack_count)
        return packed_batches, list(accumulate(step_microbatch_counts))

    @staticmethod
    def _optimizer_step_batches(packed_batches: list[dict], boundaries: list[int]):
        start = 0
        for end in boundaries:
            yield packed_batches[start:end]
            start = end
        if start != len(packed_batches):
            raise ValueError("gradient accumulation boundaries do not cover all packed batches")

    def train(self, rollout_id: int, rollout_data_ref: list, values_refs: list | None = None) -> None:
        if self.args.debug_rollout_only:
            return
        self.wake_up()
        with inverse_timer("train_wait"), timer("train"):
            if self._pending_episodes is not None:
                episodes = self._pending_episodes
                packed_batches = self._pending_packed_batches
                grad_accum = self._pending_grad_accum
                self._pending_episodes = None
                self._pending_packed_batches = None
                self._pending_grad_accum = None
            else:
                if self.rollout_data_postprocess is not None:
                    self.rollout_data_postprocess(self.args)
                episodes = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
                packed_batches = None
                grad_accum = None
            values = ray.get(values_refs[self.dp_rank]) if values_refs is not None else None
            self._train_core(rollout_id, episodes, values, packed_batches, grad_accum)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )
        self.sleep()
        clear_memory()

    def _train_core(self, rollout_id, episodes, values=None, packed_batches=None, grad_accum=None) -> None:
        if self.args.advantage_estimator in ("grpo", "gspo"):
            for episode in episodes:
                episode._advantages = [episode.reward] * episode.num_edges
                episode._returns = episode._advantages
        elif self.args.advantage_estimator == "ppo_gae":
            if values is None:
                raise ValueError("PPO requires critic values")
            self._compute_ppo_advantages(episodes, values)
        else:
            raise NotImplementedError(self.args.advantage_estimator)

        if packed_batches is None:
            packed_batches, grad_accum = self._packed_data(episodes)
        else:
            update_packed_advantages(packed_batches, episodes)
        if not grad_accum:
            raise ValueError("training produced no microbatches")
        self._run_train_loop(rollout_id, packed_batches, grad_accum)
        self.prof.step(rollout_id=rollout_id)
        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=None)
        self._maybe_update_ref_model(rollout_id)

    def _compute_ppo_advantages(self, episodes: list[Episode], values: list[torch.Tensor]) -> None:
        if len(episodes) != len(values):
            raise ValueError("episode and value counts differ")
        max_edges = max(episode.num_edges for episode in episodes)
        rewards = torch.zeros(len(episodes), max_edges)
        value_tensor = torch.zeros_like(rewards)
        masks = torch.zeros_like(rewards, dtype=torch.bool)
        for index, (episode, value) in enumerate(zip(episodes, values, strict=True)):
            edge_count = episode.num_edges
            if len(value) != edge_count:
                raise ValueError(f"episode {index} has {edge_count} edges but {len(value)} values")
            mask = torch.as_tensor(episode.loss_mask, dtype=torch.bool)
            active = mask.nonzero().flatten()
            if active.numel() == 0:
                raise ValueError(f"episode {index} has no policy-controlled edges")
            rewards[index, active[-1]] = episode.reward
            value_tensor[index, :edge_count] = value.float()
            masks[index, :edge_count] = mask
        advantages, returns = vanilla_gae(
            rewards,
            value_tensor,
            masks,
            self.args.gamma,
            self.args.lambd,
        )
        if self.args.normalize_advantages:
            selected = torch.cat(
                [advantages[i, : episode.num_edges][masks[i, : episode.num_edges]] for i, episode in enumerate(episodes)]
            )
            stats = torch.tensor(
                [selected.sum(), (selected**2).sum(), selected.numel()],
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(stats, group=self.dp_group)
            mean = stats[0] / stats[2]
            std = (stats[1] / stats[2] - mean**2).clamp_min(0).sqrt().clamp_min(1e-8)
            advantages = torch.where(masks, (advantages - mean.cpu()) / std.cpu(), 0)
        for index, episode in enumerate(episodes):
            edge_count = episode.num_edges
            episode._advantages = advantages[index, :edge_count].tolist()
            episode._returns = returns[index, :edge_count].tolist()
            episode._values = values[index].tolist()

    def _steps_per_rollout(self) -> int:
        return self.args.rollout_batch_size * self.args.n_samples_per_prompt // self.args.global_batch_size

    def _begin_gradient_accumulation(self) -> None:
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
        from nemo_automodel.components.training.utils import prepare_for_grad_accumulation

        prepare_for_grad_accumulation(self.model_parts, pp_enabled=False)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            float(self.backward_group_size),
            device=torch.cuda.current_device(),
        )

    def _prepare_final_backward(self) -> None:
        from nemo_automodel.components.training.utils import prepare_for_final_backward

        prepare_for_final_backward(self.model_parts, pp_enabled=False)

    def _after_first_microbatch(self) -> None:
        from nemo_automodel.components.training.utils import prepare_after_first_microbatch

        prepare_after_first_microbatch()

    @contextlib.contextmanager
    def _sync_context(self, is_final_microbatch: bool):
        from nemo_automodel.components.distributed.utils import get_sync_ctx

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
        from nemo_automodel.components.training.utils import (
            get_expert_tp_replication_factor,
            scale_grads_and_clip_grad_norm,
        )

        self.checkpointer.maybe_wait_for_staging()
        ep_axis_name = None
        if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names:
            ep_axis_name = "ep"
        max_grad_norm = None if self.args.nemo_cpu_offload else self.args.clip_grad
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
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
        if self.args.nemo_cpu_offload:
            grad_norm = clip_cpu_offloaded_grad_norm(
                (parameter for model_part in self.model_parts for parameter in model_part.parameters()),
                self.args.clip_grad,
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
            dist.all_reduce(tensor, group=self.dp_cp_group)
            reduced[key] = tensor.item()
        return reduced

    def _log_step(self, metrics: dict[str, float], grad_norm: float) -> None:
        if dist.get_rank() != 0:
            return
        log_dict = {f"{self._train_log_prefix}/{key}": value for key, value in metrics.items()}
        log_dict[f"{self._train_log_prefix}/grad_norm"] = grad_norm
        for index, lr in enumerate(self.lr_scheduler.get_last_lr()):
            log_dict[f"{self._train_log_prefix}/lr-pg_{index}"] = lr
        log_dict["train/step"] = self.global_step
        logger.info("%s step %d: %s", self._train_log_prefix, self.global_step, log_dict)
        logging_utils.log(self.args, log_dict)

    def _log_packed_metrics(self, packed_batches, metric_keys):
        log_dict = {}
        for metric_key in metric_keys:
            if metric_key not in packed_batches[0]:
                continue
            value = torch.zeros((), device=torch.cuda.current_device())
            for packed_batch in packed_batches:
                for batch in unpack_sequences(packed_batch):
                    metric = batch.get(metric_key)
                    if isinstance(metric, torch.Tensor):
                        mask = batch["loss_masks"].to(value.device)
                        value += (metric.to(value.device) * mask).sum() / mask.sum().clamp_min(1)
            dist.all_reduce(value, group=self.dp_group)
            log_dict[f"{self._train_log_prefix}/{metric_key}"] = (
                value / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0 and log_dict:
            log_dict["train/step"] = self.global_step
            logging_utils.log(self.args, log_dict)

    def update_weights(self) -> None:
        pass
