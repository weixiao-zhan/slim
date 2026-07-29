# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed actor, critic, routing-replay, and checkpoint qualification."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import MethodType, SimpleNamespace

import pynvml
import torch
import torch.distributed as dist
from nemo_automodel.components.moe.router_replay import RouterReplay
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from slim.backends.nemo import checkpoint
from slim.backends.nemo import actor as actor_module
from slim.backends.nemo import critic as critic_module
from slim.backends.nemo.actor import ActorNeMoTrainer
from slim.backends.nemo.base import NeMoTrainer
from slim.backends.nemo.critic import CriticNeMoTrainer
from slim.backends.nemo.data_packing import (
    build_token_budget_partitions,
    pack_sequences,
    token_slots_to_edges,
)
from slim.backends.nemo.forward import model_forward, prepare_forward
from slim.backends.nemo.loss import count_global_denominators, selective_log_probs
from slim.backends.nemo.lr_scheduler import get_lr_scheduler
from slim.backends.nemo.models import validate_config
from slim.backends.nemo.update_weight_utils import UpdateWeightFromTensor
from slim.utils.distributed_utils import get_gloo_group, init_gloo_group
from slim.utils.types import Episode


def _nvml_process_used_memory() -> int:
    pynvml.nvmlInit()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{device.uuid}")
    process_id = os.getpid()
    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if process.pid == process_id:
            return int(process.usedGpuMemory)
    raise RuntimeError(f"NVML did not report CUDA process {process_id}")


def _memory_snapshot(stage: str) -> dict[str, object] | None:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    local = torch.tensor(
        [
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            total - free,
            _nvml_process_used_memory(),
        ],
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local, group=get_gloo_group())
    if dist.get_rank() != 0:
        return None
    snapshot = {
        "stage": stage,
        "ranks": [
            {
                "rank": rank,
                "allocated_mib": round(values[0].item() / 2**20, 2),
                "reserved_mib": round(values[1].item() / 2**20, 2),
                "cuda_context_used_mib": round(values[2].item() / 2**20, 2),
                "nvml_process_used_mib": round(values[3].item() / 2**20, 2),
            }
            for rank, values in enumerate(gathered)
        ],
    }
    print(json.dumps({"memory": snapshot}, sort_keys=True), flush=True)
    return snapshot


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--role", choices=("actor", "critic"), required=True)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--routing-replay", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--heterogeneous-multimodal", action="store_true")
    parser.add_argument("--rollout-data", type=Path)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=2048)
    parser.add_argument(
        "--freeze-vision-tower",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--weight-conversion-only", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trainer_args(cli: argparse.Namespace) -> SimpleNamespace:
    checkpoint_dir = str(cli.checkpoint_dir) if cli.checkpoint_dir is not None else None
    return SimpleNamespace(
        activation_checkpointing=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        advantage_estimator="grpo",
        async_save=False,
        calculate_per_token_loss=False,
        checkpoint_cpu_offload=False,
        checkpoint_save_consolidated="every",
        ckpt_step=None,
        clip_grad=1.0,
        context_parallel_size=cli.context_parallel_size,
        custom_loss_function_path=None,
        defer_fsdp_grad_sync=False,
        distributed_timeout_minutes=30,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_c=None,
        eps_clip_high=0.2,
        expert_model_parallel_size=cli.expert_parallel_size,
        freeze_audio_tower=True,
        freeze_language_model=False,
        freeze_vision_tower=cli.freeze_vision_tower,
        get_mismatch_metrics=False,
        global_batch_size=8,
        hf_checkpoint=cli.checkpoint,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        load=checkpoint_dir,
        loss_type="policy_loss",
        lr=1e-2,
        lr_actor=1e-2,
        lr_critic=1e-2,
        lr_critic_value_head=1e-2,
        lr_decay_iters=None,
        lr_decay_style="constant",
        lr_min=0.0,
        lr_scheduler_start_step=None,
        lr_warmup_fraction=None,
        lr_warmup_init=0.0,
        lr_warmup_iters=0,
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
        mismatch_correction="none",
        nemo_dispatcher="torch",
        nemo_experts_backend="torch_mm",
        nemo_linear_backend="torch",
        nemo_rms_norm_backend="torch_fp32",
        no_load_lr_scheduler=False,
        no_load_optim=False,
        no_load_rng=False,
        no_save_optim=False,
        n_samples_per_prompt=1,
        num_rollout=4,
        old_logprob_source="rollout",
        policy_surrogate="ppo_clip",
        ref_load=cli.checkpoint,
        rollout_batch_size=8,
        rollout_temperature=1.0,
        save=checkpoint_dir,
        start_rollout_id=0,
        use_rollout_routing_replay=cli.routing_replay,
        use_unbiased_kl=False,
        value_clip=0.2,
        weight_decay=0.0,
    )


def _pack(processor=None) -> dict:
    multimodal_inputs = None
    multimodal_num_items = None
    if processor is None:
        first_document = [11, 12, 13, 14, 15]
    else:
        from PIL import Image

        image = Image.new("RGB", (64, 64), (30, 120, 200))
        rendered = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Name the dominant color.<|im_end|>\n<|im_start|>assistant\n"
        encoded = processor(text=[rendered], images=[image], return_tensors="pt")
        first_document = encoded["input_ids"].squeeze(0).tolist()
        multimodal_inputs = {name: encoded[name].detach().cpu() for name in ("pixel_values", "image_grid_thw") if name in encoded}
        multimodal_num_items = {name: [tensor.shape[0], 0] for name, tensor in multimodal_inputs.items()}
    second_document = [101, 102, 103, 104, 105, 106]
    tokens = torch.tensor(first_document + second_document, dtype=torch.long)
    first_end = len(first_document)
    edge_count = tokens.numel() - 2
    pack = {
        "tokens": tokens,
        "position_ids": torch.cat(
            (
                torch.arange(first_end, dtype=torch.long),
                torch.arange(len(second_document), dtype=torch.long),
            )
        ),
        "loss_masks": torch.ones(edge_count, dtype=torch.int32),
        "advantages": torch.linspace(0.5, 1.5, edge_count),
        "returns": torch.linspace(0.25, 1.25, edge_count),
        "old_values": torch.zeros(edge_count),
        "cu_seqlens": torch.tensor([0, first_end, tokens.numel()], dtype=torch.int32),
        "response_lengths": [first_end - 1, len(second_document) - 1],
        "reward": [1.0, 1.0],
        "_episode_indices": [0, 1],
    }
    if multimodal_inputs:
        pack["multimodal_inputs"] = multimodal_inputs
        pack["multimodal_num_items"] = multimodal_num_items
    return pack


def _uses_local_vision(cli: argparse.Namespace, dp_rank: int) -> bool:
    if cli.heterogeneous_multimodal:
        return dp_rank == 1
    return cli.multimodal


def _training_pack(
    cli: argparse.Namespace,
    trainer: ActorNeMoTrainer | CriticNeMoTrainer,
) -> tuple[dict, bool]:
    uses_vision = _uses_local_vision(cli, trainer.dp_rank)
    return _pack(trainer.processor if uses_vision else None), uses_vision


def _load_rollout_episodes(path: Path, samples_per_prompt: int = 8) -> list[Episode]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} does not contain an episodes list")
    if len(records) % samples_per_prompt:
        raise ValueError(f"{len(records)} episodes is not divisible by group size {samples_per_prompt}")

    episodes = [Episode(**record) for record in records]
    for start in range(0, len(episodes), samples_per_prompt):
        group = episodes[start : start + samples_per_prompt]
        mean_reward = sum(float(episode.reward) for episode in group) / samples_per_prompt
        for episode in group:
            episode.ensure_edge_alignment()
            episode.raw_reward = episode.reward
            advantage = float(episode.reward) - mean_reward
            episode.reward = advantage
            episode._advantages = [advantage] * episode.num_edges
            episode._returns = list(episode._advantages)
    return episodes


def _heterogeneous_rollout_partition(
    episodes: list[Episode],
    *,
    dp_rank: int,
    dp_size: int,
) -> tuple[list[Episode], bool]:
    if dp_size < 2 or dp_size % 2:
        raise ValueError(f"heterogeneous rollout partition requires an even DP size of at least 2, got {dp_size}")
    if len(episodes) % dp_size:
        raise ValueError(f"fixed batch size {len(episodes)} is not divisible by DP size {dp_size}")

    text_count = sum(not episode.multimodal_inputs for episode in episodes)
    vision = [episode for episode in episodes if episode.multimodal_inputs]
    if not text_count or not vision:
        raise ValueError("heterogeneous rollout qualification requires both text and vision episodes")
    inactive_count = sum(not any(float(advantage) != 0 for advantage in getattr(episode, "_advantages", ())) for episode in episodes)
    if inactive_count:
        raise ValueError(f"heterogeneous rollout qualification has {inactive_count} episodes with zero advantage")
    active_vision = [episode for episode in vision if any(float(advantage) != 0 for advantage in getattr(episode, "_advantages", ()))]
    if len(vision) != 4 or len(active_vision) != len(vision):
        raise ValueError("heterogeneous rollout qualification requires four vision episodes with nonzero advantages")

    vision_dp_ranks = {index % dp_size for index, episode in enumerate(episodes) if episode.multimodal_inputs}
    if vision_dp_ranks != {1}:
        raise ValueError(f"fixed heterogeneous rollout assigns vision to DP ranks {sorted(vision_dp_ranks)}, expected [1]")

    local = episodes[dp_rank::dp_size]
    uses_vision = dp_rank == 1
    return local, uses_vision


def _rollout_training_packs(
    cli: argparse.Namespace,
    trainer: ActorNeMoTrainer,
) -> tuple[list[dict], bool]:
    if cli.rollout_data is None:
        raise ValueError("--rollout-data is required for rollout-backed qualification")
    episodes = _load_rollout_episodes(cli.rollout_data)
    local_episodes, uses_vision = _heterogeneous_rollout_partition(
        episodes,
        dp_rank=trainer.dp_rank,
        dp_size=trainer.dp_size,
    )
    token_budget = cli.max_tokens_per_gpu * trainer.cp_size
    partitions = build_token_budget_partitions(
        [len(episode.tokens) for episode in local_episodes],
        token_budget,
    )
    synchronized_count = torch.tensor(len(partitions), dtype=torch.int32, device=torch.cuda.current_device())
    dist.all_reduce(synchronized_count, op=dist.ReduceOp.MAX, group=trainer.dp_group)
    pack_count = int(synchronized_count.item())
    partitions = build_token_budget_partitions(
        [len(episode.tokens) for episode in local_episodes],
        token_budget,
        num_packs=pack_count,
    )
    return pack_sequences(local_episodes, partitions=partitions), uses_vision


def _validate_modality_layout(
    trainer: ActorNeMoTrainer | CriticNeMoTrainer,
    uses_vision: bool,
    packs: list[dict],
) -> list[dict[str, object]]:
    episode_count = sum(pack["cu_seqlens"].numel() - 1 for pack in packs)
    vision_episode_count = 0
    for pack in packs:
        media_counts = pack.get("multimodal_num_items", {})
        for episode_offset in range(pack["cu_seqlens"].numel() - 1):
            if any(counts[episode_offset] > 0 for counts in media_counts.values()):
                vision_episode_count += 1
    if uses_vision != (vision_episode_count > 0):
        raise RuntimeError(f"DP rank {trainer.dp_rank} vision assignment disagrees with its packs: uses_vision={uses_vision}, vision_episodes={vision_episode_count}")

    local = {
        "rank": dist.get_rank(),
        "dp_rank": trainer.dp_rank,
        "cp_rank": trainer.cp_rank,
        "pack": "vision_and_text" if uses_vision else "text_only",
        "physical_packs": len(packs),
        "episodes": episode_count,
        "text_episodes": episode_count - vision_episode_count,
        "vision_episodes": vision_episode_count,
    }
    layout = [None] * dist.get_world_size()
    dist.all_gather_object(layout, local, group=get_gloo_group())

    by_dp_rank: dict[int, set[tuple]] = {}
    for item in layout:
        signature = (
            item["pack"],
            item["physical_packs"],
            item["episodes"],
            item["text_episodes"],
            item["vision_episodes"],
        )
        by_dp_rank.setdefault(item["dp_rank"], set()).add(signature)
    inconsistent_cp_groups = {dp_rank: sorted(signatures) for dp_rank, signatures in by_dp_rank.items() if len(signatures) != 1}
    if inconsistent_cp_groups:
        raise RuntimeError(f"CP peers received different modality packs: {inconsistent_cp_groups}")

    vision_dp_ranks = {item["dp_rank"] for item in layout if item["vision_episodes"] > 0}
    if vision_dp_ranks != {1}:
        raise RuntimeError(f"heterogeneous multimodal qualification put vision on DP ranks {sorted(vision_dp_ranks)}")
    dp1 = next(item for item in layout if item["dp_rank"] == 1)
    if dp1["text_episodes"] == 0:
        raise RuntimeError("DP rank 1 must contain both text and vision episodes")
    return layout


def _install_vision_forward_counter(trainer: ActorNeMoTrainer | CriticNeMoTrainer) -> list[int]:
    counts = [0]
    original = trainer.model._encode_vision_for_cp

    def counted(self, *args, **kwargs):
        counts[0] += 1
        return original(*args, **kwargs)

    trainer.model._encode_vision_for_cp = MethodType(counted, trainer.model)
    return counts


def _local_parameter_snapshot(module: torch.nn.Module, suffix: str) -> tuple[str, bool, torch.Tensor]:
    for name, parameter in module.named_parameters():
        canonical = _canonical_name(name)
        if not canonical.endswith(suffix):
            continue
        tensor = parameter.detach()
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return canonical, parameter.requires_grad, tensor.float().cpu().clone()
    raise RuntimeError(f"parameter ending in {suffix!r} was not found")


def _vision_update_metrics(
    trainer: ActorNeMoTrainer | CriticNeMoTrainer,
    before: tuple[str, bool, torch.Tensor],
    *,
    expect_frozen: bool,
) -> dict[str, object]:
    name, was_trainable, before_tensor = before
    after_name, is_trainable, after_tensor = _local_parameter_snapshot(trainer.model, name)
    if after_name != name or is_trainable != was_trainable:
        raise RuntimeError(f"vision parameter state changed unexpectedly for {name}")
    if is_trainable == expect_frozen:
        raise RuntimeError(f"vision parameter {name} requires_grad={is_trainable}, expected freeze={expect_frozen}")

    delta_sq = torch.tensor(
        (after_tensor - before_tensor).square().sum().item(),
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(delta_sq, op=dist.ReduceOp.SUM)
    update_l2 = delta_sq.sqrt().item()
    if expect_frozen and update_l2 != 0:
        raise RuntimeError(f"frozen vision parameter {name} changed by L2={update_l2}")
    if not expect_frozen and update_l2 == 0:
        raise RuntimeError(f"trainable vision parameter {name} did not update")
    return {
        "vision_parameter": name,
        "vision_parameter_trainable": is_trainable,
        "vision_parameter_update_l2": update_l2,
    }


def _build_trainer(
    cli: argparse.Namespace,
    args: SimpleNamespace,
    record_memory: Callable[[str], None] | None = None,
):
    def record(stage: str) -> None:
        if record_memory is not None:
            record_memory(stage)

    trainer_type = ActorNeMoTrainer if cli.role == "actor" else CriticNeMoTrainer
    trainer = trainer_type.__new__(trainer_type)
    trainer.args = args
    trainer.global_step = 0
    trainer.hf_config = AutoConfig.from_pretrained(cli.checkpoint, trust_remote_code=True)
    record("config_loaded")
    NeMoTrainer._setup_topology(trainer)
    validate_config(trainer.hf_config, trainer.topology)
    record("topology_built")

    backend_module = actor_module if cli.role == "actor" else critic_module
    original_build_optimizer = backend_module.build_optimizer

    def measured_build_optimizer(*optimizer_args, **optimizer_kwargs):
        record("model_sharded")
        optimizer = original_build_optimizer(*optimizer_args, **optimizer_kwargs)
        record("optimizer_built")
        return optimizer

    backend_module.build_optimizer = measured_build_optimizer
    try:
        trainer._create_model_and_optimizer(cli.checkpoint)
    finally:
        backend_module.build_optimizer = original_build_optimizer
    trainer.lr_scheduler = get_lr_scheduler(args, trainer.optimizer)
    record("scheduler_built")
    trainer._checkpoint_load_dir = str(cli.checkpoint_dir) if cli.checkpoint_dir is not None else None
    trainer._checkpoint_save_dir = str(cli.checkpoint_dir) if cli.checkpoint_dir is not None else None
    trainer.tokenizer = AutoTokenizer.from_pretrained(cli.checkpoint, trust_remote_code=True)
    record("tokenizer_loaded")
    trainer.processor = AutoProcessor.from_pretrained(cli.checkpoint, trust_remote_code=True)
    record("processor_loaded")
    trainer.checkpointer = checkpoint.build_checkpointer(trainer)
    record("checkpointer_built")
    return trainer


def _canonical_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _full_parameter(module: torch.nn.Module, suffix: str) -> tuple[str, torch.Tensor]:
    for name, parameter in module.named_parameters():
        canonical = _canonical_name(name)
        if not canonical.endswith(suffix):
            continue
        tensor = parameter.detach()
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        return canonical, tensor.float().cpu()
    raise RuntimeError(f"parameter ending in {suffix!r} was not found")


def _local_parameter(module: torch.nn.Module, suffix: str) -> tuple[str, torch.Tensor]:
    for name, parameter in module.named_parameters():
        canonical = _canonical_name(name)
        if not canonical.endswith(suffix):
            continue
        tensor = parameter.detach()
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        return canonical, tensor.float().cpu()
    raise RuntimeError(f"parameter ending in {suffix!r} was not found")


def _optimizer_digest(optimizer: torch.optim.Optimizer) -> list[tuple[int, str, tuple[int, ...], float, float]]:
    digest = []
    parameter_index = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for name in sorted(state):
                value = state[name]
                if not isinstance(value, torch.Tensor):
                    continue
                if isinstance(value, DTensor):
                    value = value.to_local()
                value = value.detach().double()
                digest.append(
                    (
                        parameter_index,
                        name,
                        tuple(value.shape),
                        value.sum().item(),
                        value.square().sum().item(),
                    )
                )
            parameter_index += 1
    return digest


def _record_actor_inputs(trainer: ActorNeMoTrainer, pack: dict) -> tuple[torch.Tensor, int]:
    prepared = prepare_forward(trainer.model, trainer.device_mesh, pack, padding_token_id=0)
    replay_count = len(RouterReplay.instances())
    if replay_count:
        with torch.no_grad(), RouterReplay.record(), prepared.context_factory():
            output = model_forward(trainer.model, prepared.model_batch)
            local_log_probs = selective_log_probs(output.logits, prepared.fields["labels"])
        recorded = RouterReplay.collect()
        local_routes = torch.stack(recorded, dim=1).unsqueeze(0)
        full_routes = prepared.gather(local_routes, fill=0)
        pack["rollout_routed_experts"] = token_slots_to_edges(
            full_routes.squeeze(0),
            pack["cu_seqlens"],
        ).to(device="cpu", dtype=torch.int32)
    else:
        with torch.no_grad(), prepared.context_factory():
            output = model_forward(trainer.model, prepared.model_batch)
            local_log_probs = selective_log_probs(output.logits, prepared.fields["labels"])
    full_log_probs = prepared.gather(local_log_probs, fill=0)
    pack["rollout_log_probs"] = (
        token_slots_to_edges(
            full_log_probs.squeeze(0),
            pack["cu_seqlens"],
        )
        .detach()
        .cpu()
    )
    return full_log_probs.detach().cpu(), replay_count


def _install_replay_counters() -> list[int]:
    instances = RouterReplay.instances()
    counts = [0] * len(instances)
    for index, handle in enumerate(instances):
        original = handle.apply

        def counted(self, indices, *, _index=index, _original=original):
            from nemo_automodel.components.moe.router_replay import RouterReplayMode

            if self.mode == RouterReplayMode.REPLAY:
                counts[_index] += 1
            return _original(indices)

        handle.apply = MethodType(counted, handle)
    return counts


def _train_actor(trainer: ActorNeMoTrainer, packs: list[dict]) -> tuple[dict, float, list[int]]:
    for pack in packs:
        _record_actor_inputs(trainer, pack)
    replay_counts = _install_replay_counters()
    trainer.optimizer.zero_grad(set_to_none=True)
    global_sequences, global_tokens = count_global_denominators(
        packs,
        trainer.dp_group,
        torch.device("cuda", torch.cuda.current_device()),
    )
    trainer._begin_gradient_accumulation()
    metric_sums: dict[str, torch.Tensor] = {}
    for index, pack in enumerate(packs):
        metrics = trainer._train_microbatch(
            pack,
            is_final=index == len(packs) - 1,
            global_sequences=global_sequences,
            global_tokens=global_tokens,
        )
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
        if index == 0:
            trainer._after_first_microbatch()
    grad_norm = trainer._optimizer_step()
    return metric_sums, grad_norm, replay_counts


def _train_critic(trainer: CriticNeMoTrainer, pack: dict) -> tuple[dict, float]:
    trainer.optimizer.zero_grad(set_to_none=True)
    global_sequences, _ = count_global_denominators(
        [pack],
        trainer.dp_group,
        torch.device("cuda", torch.cuda.current_device()),
    )
    trainer._begin_gradient_accumulation()
    metrics = trainer._train_microbatch(pack, is_final=True, global_sequences=global_sequences)
    trainer._after_first_microbatch()
    grad_norm = trainer._optimizer_step()
    return metrics, grad_norm


def _assert_finite_metrics(metrics: dict[str, torch.Tensor], grad_norm: float) -> None:
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise RuntimeError(f"non-finite gradient norm: {grad_norm}")
    non_finite = [name for name, value in metrics.items() if not torch.isfinite(value).all()]
    if non_finite:
        raise RuntimeError(f"non-finite training metrics: {non_finite}")


def _qualify_weight_conversion(trainer: ActorNeMoTrainer) -> dict[str, int]:
    updater = UpdateWeightFromTensor(trainer.args, trainer.model)
    converted = {}
    converted_bytes = 0
    for name, tensor in updater._hf_tensors():
        if name in converted:
            raise RuntimeError(f"single-tensor conversion produced duplicate HF key {name!r}")
        if tensor.device.type != "cuda" or not tensor.is_contiguous():
            raise RuntimeError(f"converted HF tensor {name!r} is not contiguous CUDA storage")
        converted[name] = (tuple(tensor.shape), str(tensor.dtype))
        converted_bytes += tensor.numel() * tensor.element_size()

    expected = {name for name in getattr(trainer.model, "_pre_shard_hf_state_dict_keys", ()) if not name.endswith("_extra_state")}
    actual = set(converted)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"single-tensor HF conversion disagrees with the model state contract: missing={missing[:5]}, unexpected={unexpected[:5]}")

    signature = sorted((name, *metadata) for name, metadata in converted.items())
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, signature, group=get_gloo_group())
    if any(candidate != signature for candidate in gathered):
        raise RuntimeError("single-tensor HF conversion metadata differs across ranks")
    return {
        "converted_tensors": len(converted),
        "converted_bytes": converted_bytes,
    }


def _prepare_reference(trainer: ActorNeMoTrainer) -> None:
    trainer.ref_model = trainer._create_ref_model(trainer.args.ref_load)
    with torch.no_grad():
        for parameter in trainer.ref_model.parameters():
            if parameter.is_floating_point():
                parameter.add_(0.125)
                break


def _save_and_reload(
    cli: argparse.Namespace,
    trainer,
    tracked_suffix: str,
    memory_snapshots: list[dict[str, object]],
) -> tuple[object, dict[str, object]]:
    if cli.checkpoint_dir is None:
        raise ValueError("checkpoint qualification requires --checkpoint-dir")

    if cli.role == "actor":
        _prepare_reference(trainer)
        reference_name, reference_before = _local_parameter(trainer.ref_model, "embed_tokens.weight")
    else:
        reference_name = None
        reference_before = None

    tracked_name, tracked_before = _full_parameter(trainer.checkpoint_model, tracked_suffix)
    scheduler_before = trainer.lr_scheduler.state_dict()
    saved_rollout_id = 3
    checkpoint.save(trainer, rollout_id=saved_rollout_id, force_sync=True)
    snapshot = _memory_snapshot("checkpoint_saved")
    if snapshot is not None:
        memory_snapshots.append(snapshot)
    optimizer_before = _optimizer_digest(trainer.optimizer)

    expected_cpu_random = torch.rand(8)
    expected_cuda_random = torch.rand(8, device=torch.cuda.current_device()).cpu()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    snapshot = _memory_snapshot("original_trainer_released")
    if snapshot is not None:
        memory_snapshots.append(snapshot)

    args = _trainer_args(cli)
    resumed = _build_trainer(cli, args)
    snapshot = _memory_snapshot("resume_model_built")
    if snapshot is not None:
        memory_snapshots.append(snapshot)
    payload = checkpoint.load(resumed)
    snapshot = _memory_snapshot("model_optimizer_loaded")
    if snapshot is not None:
        memory_snapshots.append(snapshot)
    if cli.role == "actor":
        resumed.ref_model = resumed._create_ref_model(resumed.args.ref_load)
    checkpoint.finalize_load(resumed, payload)
    snapshot = _memory_snapshot("resume_finalized")
    if snapshot is not None:
        memory_snapshots.append(snapshot)

    loaded_name, tracked_after = _full_parameter(resumed.checkpoint_model, tracked_suffix)
    if loaded_name != tracked_name:
        raise RuntimeError(f"resumed a different tracked parameter: {tracked_name} versus {loaded_name}")
    torch.testing.assert_close(tracked_after, tracked_before, rtol=0, atol=0)
    if _optimizer_digest(resumed.optimizer) != optimizer_before:
        raise RuntimeError("optimizer state changed across checkpoint resume")
    if resumed.lr_scheduler.state_dict() != scheduler_before:
        raise RuntimeError("scheduler state changed across checkpoint resume")
    expected_start_rollout_id = saved_rollout_id + 1
    if resumed.global_step != 1 or resumed.args.start_rollout_id != expected_start_rollout_id:
        raise RuntimeError(
            "checkpoint metadata did not resume exactly: "
            f"global_step={resumed.global_step}, start_rollout_id={resumed.args.start_rollout_id}"
        )
    torch.testing.assert_close(torch.rand(8), expected_cpu_random, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.rand(8, device=torch.cuda.current_device()).cpu(),
        expected_cuda_random,
        rtol=0,
        atol=0,
    )

    if cli.role == "actor":
        loaded_reference_name, reference_after = _local_parameter(resumed.ref_model, "embed_tokens.weight")
        if loaded_reference_name != reference_name:
            raise RuntimeError("resumed a different reference parameter")
        torch.testing.assert_close(reference_after, reference_before, rtol=0, atol=0)

    return resumed, {
        "checkpoint_iteration": payload["iteration"],
        "optimizer_state_tensors": len(optimizer_before),
        "scheduler_last_epoch": scheduler_before["last_epoch"],
    }


def _write_actor_score_artifact(cli: argparse.Namespace, trainer: ActorNeMoTrainer, output: Path) -> None:
    pack = _pack(trainer.processor if cli.multimodal or cli.heterogeneous_multimodal else None)
    prepared = prepare_forward(trainer.model, trainer.device_mesh, pack, padding_token_id=0)
    trainer.model.eval()
    with torch.no_grad(), prepared.context_factory():
        model_output = model_forward(trainer.model, prepared.model_batch)
        local_log_probs = selective_log_probs(model_output.logits, prepared.fields["labels"])
    log_probs = prepared.gather(local_log_probs, fill=0).cpu()
    if dist.get_rank() == 0:
        torch.save(
            {
                "documents": [
                    pack["tokens"][start:end].tolist()
                    for start, end in zip(
                        pack["cu_seqlens"].tolist()[:-1],
                        pack["cu_seqlens"].tolist()[1:],
                        strict=True,
                    )
                ],
                "log_probs": token_slots_to_edges(log_probs.squeeze(0), pack["cu_seqlens"]),
                "consolidated_checkpoint": str(cli.checkpoint_dir / "iter_0000004/model/consolidated"),
            },
            output.with_suffix(".score.pt"),
        )


def main() -> None:
    cli = _arguments()
    if cli.multimodal and cli.heterogeneous_multimodal:
        raise ValueError("--multimodal and --heterogeneous-multimodal are mutually exclusive")
    if cli.routing_replay and cli.role != "actor":
        raise ValueError("routing replay qualification applies only to the actor")
    if cli.weight_conversion_only and cli.role != "actor":
        raise ValueError("weight conversion qualification applies only to the actor")
    if cli.heterogeneous_multimodal and cli.role != "actor":
        raise ValueError("heterogeneous multimodal qualification applies only to the actor")
    if cli.checkpoint_dir is not None and cli.checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint output already exists: {cli.checkpoint_dir}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    init_gloo_group()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    memory_snapshots = []
    snapshot = _memory_snapshot("distributed_initialized")
    if snapshot is not None:
        memory_snapshots.append(snapshot)
    args = _trainer_args(cli)

    def record_memory(stage: str) -> None:
        snapshot = _memory_snapshot(stage)
        if snapshot is not None:
            memory_snapshots.append(snapshot)

    trainer = _build_trainer(cli, args, record_memory=record_memory)
    if cli.weight_conversion_only:
        conversion_metrics = _qualify_weight_conversion(trainer)
        record_memory("weight_conversion_completed")
        if dist.get_rank() == 0:
            cli.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "role": cli.role,
                "context_parallel_size": cli.context_parallel_size,
                "expert_parallel_size": cli.expert_parallel_size,
                "memory": memory_snapshots,
                **conversion_metrics,
            }
            cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(json.dumps(payload, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    if cli.heterogeneous_multimodal and trainer.dp_size < 2:
        raise ValueError("heterogeneous multimodal qualification requires at least two logical DP ranks")
    if cli.rollout_data is not None:
        if not isinstance(trainer, ActorNeMoTrainer):
            raise ValueError("rollout-backed qualification applies only to the actor")
        packs, uses_vision = _rollout_training_packs(cli, trainer)
    else:
        pack, uses_vision = _training_pack(cli, trainer)
        packs = [pack]
    modality_layout = _validate_modality_layout(trainer, uses_vision, packs) if cli.heterogeneous_multimodal else []
    vision_forward_counter = _install_vision_forward_counter(trainer) if cli.heterogeneous_multimodal else None
    vision_before = _local_parameter_snapshot(trainer.model, "visual.patch_embed.proj.weight") if cli.heterogeneous_multimodal else None
    tracked_parameter = _full_parameter
    if cli.role == "actor":
        tracked_suffix = "linear_attn.norm.weight"
        tracked_name, tracked_before = tracked_parameter(trainer.model, tracked_suffix)
        metrics, grad_norm, replay_counts = _train_actor(trainer, packs)
        tracked_after_name, tracked_after = tracked_parameter(trainer.model, tracked_suffix)
        if tracked_after_name != tracked_name or torch.equal(tracked_after, tracked_before):
            raise RuntimeError(f"actor optimizer did not update {tracked_name}")
        if cli.routing_replay:
            if not replay_counts or min(replay_counts) < 2:
                raise RuntimeError(f"routing replay did not remain active through activation-checkpoint recomputation: minimum calls per layer={min(replay_counts, default=0)}")
            if any(handle.target_indices is not None for handle in RouterReplay.instances()):
                raise RuntimeError("routing replay targets were not cleared after backward")
    else:
        if len(packs) != 1:
            raise ValueError("critic qualification expects one synthetic pack")
        pack = packs[0]
        tracked_suffix = "value_head.proj.weight"
        tracked_name, tracked_before = tracked_parameter(trainer.checkpoint_model, tracked_suffix)
        metrics, grad_norm = _train_critic(trainer, pack)
        replay_counts = []
        tracked_after_name, tracked_after = tracked_parameter(trainer.checkpoint_model, tracked_suffix)
        if tracked_after_name != tracked_name or torch.equal(tracked_after, tracked_before):
            raise RuntimeError("critic optimizer did not update the value head")

    _assert_finite_metrics(metrics, grad_norm)
    vision_metrics = {}
    vision_forward_calls = []
    if cli.heterogeneous_multimodal:
        if vision_forward_counter is None or vision_before is None:
            raise RuntimeError("heterogeneous multimodal instrumentation was not initialized")
        vision_forward_calls = [None] * dist.get_world_size()
        dist.all_gather_object(
            vision_forward_calls,
            vision_forward_counter[0],
            group=get_gloo_group(),
        )
        if min(vision_forward_calls) < 2:
            raise RuntimeError(f"not every rank entered the synchronized vision path for actor-old and training forwards: {vision_forward_calls}")
        vision_metrics = _vision_update_metrics(
            trainer,
            vision_before,
            expect_frozen=cli.freeze_vision_tower,
        )
    snapshot = _memory_snapshot("optimizer_stepped")
    if snapshot is not None:
        memory_snapshots.append(snapshot)
    checkpoint_metrics = {}
    if cli.checkpoint_dir is not None:
        trainer, checkpoint_metrics = _save_and_reload(cli, trainer, tracked_suffix, memory_snapshots)
        if cli.role == "actor":
            _write_actor_score_artifact(cli, trainer, cli.output)

    if dist.get_rank() == 0:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "role": cli.role,
            "context_parallel_size": cli.context_parallel_size,
            "expert_parallel_size": cli.expert_parallel_size,
            "routing_replay": cli.routing_replay,
            "heterogeneous_multimodal": cli.heterogeneous_multimodal,
            "freeze_vision_tower": cli.freeze_vision_tower,
            "modality_layout": modality_layout,
            "vision_forward_calls": vision_forward_calls,
            "routing_layers": len(replay_counts),
            "minimum_replay_calls_per_layer": min(replay_counts, default=0),
            "grad_norm": grad_norm,
            "metrics": {name: value.item() for name, value in metrics.items()},
            "memory": memory_snapshots,
            **vision_metrics,
            **checkpoint_metrics,
        }
        cli.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
