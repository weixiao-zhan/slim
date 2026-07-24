# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line arguments for the NeMo AutoModel training backend."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import yaml


@dataclass
class NeMoArgs:
    optimizer: str = "adam"
    lr: float = 2e-5
    lr_warmup_init: float = 0.0
    lr_min: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: float | None = None
    lr_wsd_decay_iters: int | None = None
    lr_wsd_decay_style: str | None = None
    no_load_lr_scheduler: bool = False
    lr_scheduler_start_step: int | None = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    dp_replicate_size: int = 1
    sequence_parallel: bool = False

    nemo_linear_backend: str = "torch"
    nemo_rms_norm_backend: str = "torch_fp32"
    nemo_experts_backend: str = "torch_mm"
    nemo_dispatcher: str = "torch"
    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True
    freeze_language_model: bool = False
    activation_checkpointing: bool = False
    gradient_checkpointing: bool = False
    defer_fsdp_grad_sync: bool = False
    nemo_cpu_offload: bool = False

    checkpoint_save_consolidated: str = "final"
    checkpoint_cpu_offload: bool = False
    no_load_optim: bool = False
    no_load_rng: bool = False

    wandb_project: str = "slim-nemo"
    wandb_run_name: str | None = None
    deterministic_mode: bool = False
    config: str | None = None


def _field_type(field: dataclasses.Field, type_hints: dict[str, object]):
    field_type = type_hints[field.name]
    origin = get_origin(field_type)
    if origin in (UnionType,):
        members = [member for member in get_args(field_type) if member is not type(None)]
        return members[0] if members else str
    return field_type


def _parse_nemo_cli(extra_args_provider=None, ignore_unknown_args=False):
    parser = argparse.ArgumentParser("NeMo AutoModel Training", allow_abbrev=False)
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    type_hints = get_type_hints(NeMoArgs)
    for field in dataclasses.fields(NeMoArgs):
        if field.name == "config":
            continue
        flag = f"--{field.name.replace('_', '-')}"
        field_type = _field_type(field, type_hints)
        if field_type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(flag, type=field_type, default=field.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args()


def nemo_parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args = _parse_nemo_cli(extra_args_provider, ignore_unknown_args=ignore_unknown_args)
    if args.config:
        with open(args.config) as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("NeMo config must be a mapping")
        unknown = sorted(set(config) - vars(args).keys())
        if unknown:
            raise ValueError(f"unknown NeMo config fields: {', '.join(unknown)}")
        for key, value in config.items():
            setattr(args, key, value)

    args.rank = 0
    args.world_size = args.actor_num_gpus
    validate_args(args)
    return args


def validate_args(args) -> None:
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    cp = args.context_parallel_size
    ep = args.expert_model_parallel_size
    dp_replicate = args.dp_replicate_size
    world_size = getattr(args, "actor_num_gpus", None) or getattr(args, "world_size", 0)

    if tp != 1:
        raise ValueError("tensor_model_parallel_size must be 1")
    if pp != 1:
        raise ValueError("pipeline_model_parallel_size must be 1")
    if args.sequence_parallel:
        raise ValueError("sequence_parallel is not supported; use context_parallel_size")
    for name, value in (
        ("context_parallel_size", cp),
        ("expert_model_parallel_size", ep),
        ("dp_replicate_size", dp_replicate),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if world_size and world_size % (cp * dp_replicate):
        raise ValueError(
            f"world size {world_size} must be divisible by context_parallel_size * dp_replicate_size "
            f"({cp * dp_replicate})"
        )
    if args.nemo_linear_backend != "torch":
        raise ValueError("nemo_linear_backend must be torch")
    if args.nemo_rms_norm_backend != "torch_fp32":
        raise ValueError("nemo_rms_norm_backend must be torch_fp32")
    if args.nemo_experts_backend != "torch_mm":
        raise ValueError("nemo_experts_backend must be torch_mm")
    if args.nemo_dispatcher != "torch":
        raise ValueError("nemo_dispatcher must be torch")
    if args.optimizer != "adam":
        raise ValueError("optimizer must be adam")
    if args.checkpoint_save_consolidated not in ("false", "final", "every"):
        raise ValueError("checkpoint_save_consolidated must be false, final, or every")
