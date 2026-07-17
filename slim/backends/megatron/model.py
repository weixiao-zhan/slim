"""Megatron Bridge model, DDP, and optimizer construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .topology import ParallelTopology, require_single_model_chunk, validate_provider_topology

_SCHEDULER_RUNTIME_FIELDS = frozenset(
    {
        "lr_decay_steps",
        "lr_warmup_steps",
        "wd_incr_steps",
        "wsd_decay_steps",
    }
)


@dataclass(frozen=True, slots=True)
class MegatronModelAPI:
    """Lazy references to the Bridge and MCore model-construction API."""

    AutoBridge: Any
    DistributedDataParallelConfig: Any
    OptimizerConfig: Any
    SchedulerConfig: Any
    ProcessGroupCollection: Any
    setup_optimizer: Any


@dataclass(frozen=True, slots=True)
class MegatronModelBundle:
    """Objects owned by one Slim Megatron training worker."""

    bridge: Any
    provider: Any
    model: list[Any]
    optimizer: Any
    scheduler: Any
    pg_collection: Any
    topology: ParallelTopology


def load_megatron_model_api() -> MegatronModelAPI:
    """Import the optional Megatron dependencies only when this backend is selected."""

    try:
        bridge_module = importlib.import_module("megatron.bridge")
        config_module = importlib.import_module("megatron.bridge.training.config")
        process_groups_module = importlib.import_module("megatron.core.process_groups_config")
        optim_module = importlib.import_module("megatron.bridge.training.optim")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Megatron backend requires Megatron Bridge, Megatron Core, and their runtime dependencies."
        ) from exc

    return MegatronModelAPI(
        AutoBridge=bridge_module.AutoBridge,
        DistributedDataParallelConfig=config_module.DistributedDataParallelConfig,
        OptimizerConfig=config_module.OptimizerConfig,
        SchedulerConfig=config_module.SchedulerConfig,
        ProcessGroupCollection=process_groups_module.ProcessGroupCollection,
        setup_optimizer=optim_module.setup_optimizer,
    )


def _finalize(config: Any, *, name: str) -> None:
    finalize = getattr(config, "finalize", None)
    if not callable(finalize):
        raise TypeError(f"Bridge {name} must provide finalize().")
    finalize()


def _build_training_configs(
    *,
    provider: Any,
    use_distributed_optimizer: bool,
    ddp_config_kwargs: Mapping[str, Any] | None,
    optimizer_config_kwargs: Mapping[str, Any] | None,
    scheduler_config_kwargs: Mapping[str, Any] | None,
    api: MegatronModelAPI,
) -> tuple[Any, Any, Any]:
    ddp_values = dict(ddp_config_kwargs or {})
    ddp_values["use_distributed_optimizer"] = use_distributed_optimizer
    ddp_config = api.DistributedDataParallelConfig(**ddp_values)
    _finalize(ddp_config, name="DistributedDataParallelConfig")

    optimizer_values: dict[str, Any] = {
        "optimizer": "adam",
        "lr": 1e-6,
        "min_lr": 0.0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
    }
    for field_name in ("bf16", "fp16", "params_dtype"):
        if hasattr(provider, field_name):
            optimizer_values[field_name] = getattr(provider, field_name)
    optimizer_values.update(optimizer_config_kwargs or {})
    optimizer_values["use_distributed_optimizer"] = use_distributed_optimizer
    optimizer_config = api.OptimizerConfig(**optimizer_values)
    _finalize(optimizer_config, name="OptimizerConfig")

    scheduler_values: dict[str, Any] = {
        "lr_decay_style": "constant",
        "lr_warmup_init": 0.0,
        "start_weight_decay": optimizer_config.weight_decay,
        "end_weight_decay": optimizer_config.weight_decay,
        "weight_decay_incr_style": "constant",
    }
    scheduler_values.update(scheduler_config_kwargs or {})
    runtime_values = {
        name: scheduler_values.pop(name) for name in tuple(scheduler_values) if name in _SCHEDULER_RUNTIME_FIELDS
    }
    scheduler_config = api.SchedulerConfig(**scheduler_values)
    scheduler_config.lr_decay_steps = runtime_values.get(
        "lr_decay_steps",
        getattr(scheduler_config, "lr_decay_iters", None) or 1,
    )
    scheduler_config.lr_warmup_steps = runtime_values.get(
        "lr_warmup_steps",
        getattr(scheduler_config, "lr_warmup_iters", 0),
    )
    scheduler_config.wd_incr_steps = runtime_values.get(
        "wd_incr_steps",
        scheduler_config.lr_decay_steps,
    )
    scheduler_config.wsd_decay_steps = runtime_values.get(
        "wsd_decay_steps",
        getattr(scheduler_config, "lr_wsd_decay_iters", None),
    )
    _finalize(scheduler_config, name="SchedulerConfig")
    return ddp_config, optimizer_config, scheduler_config


def build_megatron_model(
    checkpoint: str | Path,
    *,
    world_size: int,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    expert_model_parallel_size: int = 1,
    use_distributed_optimizer: bool = True,
    use_rollout_routing_replay: bool = False,
    seed: int = 1234,
    trust_remote_code: bool = True,
    bridge_load_kwargs: Mapping[str, Any] | None = None,
    provider_config_kwargs: Mapping[str, Any] | None = None,
    ddp_config_kwargs: Mapping[str, Any] | None = None,
    optimizer_config_kwargs: Mapping[str, Any] | None = None,
    scheduler_config_kwargs: Mapping[str, Any] | None = None,
    api: MegatronModelAPI | None = None,
) -> MegatronModelBundle:
    """Create one non-interleaved Bridge model and its optimizer state."""

    api = api or load_megatron_model_api()
    load_kwargs = dict(bridge_load_kwargs or {})
    load_kwargs.setdefault("trust_remote_code", trust_remote_code)
    bridge = api.AutoBridge.from_hf_pretrained(checkpoint, **load_kwargs)
    provider = bridge.to_megatron_provider(load_weights=True)

    provider.tensor_model_parallel_size = tensor_model_parallel_size
    provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    provider.context_parallel_size = context_parallel_size
    provider.sequence_parallel = sequence_parallel
    provider.expert_model_parallel_size = expert_model_parallel_size
    provider.virtual_pipeline_model_parallel_size = None
    provider.moe_enable_routing_replay = use_rollout_routing_replay
    provider._enable_in_batch_packing = True
    for field_name, value in (provider_config_kwargs or {}).items():
        if not hasattr(provider, field_name):
            raise AttributeError(
                f"Bridge provider {type(provider).__name__} has no attribute {field_name!r}."
            )
        setattr(provider, field_name, value)
    _finalize(provider, name="model provider")

    topology = validate_provider_topology(provider, world_size=world_size)
    if use_rollout_routing_replay and getattr(provider, "num_moe_experts", None) is None:
        raise ValueError("Rollout routing replay requires a Bridge MoE provider.")

    provider.initialize_model_parallel(seed=seed)
    pg_collection = api.ProcessGroupCollection.use_mpu_process_groups()
    ddp_config, optimizer_config, scheduler_config = _build_training_configs(
        provider=provider,
        use_distributed_optimizer=use_distributed_optimizer,
        ddp_config_kwargs=ddp_config_kwargs,
        optimizer_config_kwargs=optimizer_config_kwargs,
        scheduler_config_kwargs=scheduler_config_kwargs,
        api=api,
    )
    model = provider.provide_distributed_model(
        ddp_config=ddp_config,
        pg_collection=pg_collection,
    )
    require_single_model_chunk(model)
    optimizer, scheduler = api.setup_optimizer(
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        model=model,
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )
    return MegatronModelBundle(
        bridge=bridge,
        provider=provider,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pg_collection=pg_collection,
        topology=topology,
    )


__all__ = [
    "MegatronModelAPI",
    "MegatronModelBundle",
    "build_megatron_model",
    "load_megatron_model_api",
]
