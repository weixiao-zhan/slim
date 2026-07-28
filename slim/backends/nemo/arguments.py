# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line arguments for the NeMo AutoModel training backend."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from types import UnionType
from typing import Literal, get_args, get_origin, get_type_hints

import yaml


@dataclass
class NeMoArgs:
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

    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    nemo_linear_backend: Literal["torch", "te"] = "torch"
    nemo_rms_norm_backend: Literal["torch", "torch_fp32", "te"] = "torch_fp32"
    nemo_experts_backend: Literal["torch", "te", "gmm", "torch_mm"] = "torch_mm"
    nemo_dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep"] = "torch"
    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True
    freeze_language_model: bool = False
    activation_checkpointing: bool = False
    defer_fsdp_grad_sync: bool = False

    checkpoint_save_consolidated: str = "final"
    checkpoint_cpu_offload: bool = False
    no_load_optim: bool = False
    no_load_rng: bool = False

    wandb_project: str = "slim-nemo"
    config: str | None = None


def _field_type(field: dataclasses.Field, type_hints: dict[str, object]):
    field_type = type_hints[field.name]
    origin = get_origin(field_type)
    if origin is UnionType:
        members = [member for member in get_args(field_type) if member is not type(None)]
        return members[0] if members else str
    if origin is Literal:
        choices = get_args(field_type)
        return type(choices[0]) if choices else str
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
            annotation = type_hints[field.name]
            choices = get_args(annotation) if get_origin(annotation) is Literal else None
            parser.add_argument(flag, type=field_type, choices=choices, default=field.default)

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
    type_hints = get_type_hints(NeMoArgs)
    for field in dataclasses.fields(NeMoArgs):
        annotation = type_hints[field.name]
        if get_origin(annotation) is not Literal:
            continue
        choices = get_args(annotation)
        value = getattr(args, field.name)
        if value not in choices:
            raise ValueError(f"{field.name} must be one of {choices}, got {value!r}")

    cp = args.context_parallel_size
    ep = args.expert_model_parallel_size
    world_size = getattr(args, "actor_num_gpus", None) or getattr(args, "world_size", 0)

    for name, value in (
        ("context_parallel_size", cp),
        ("expert_model_parallel_size", ep),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if world_size and world_size % cp:
        raise ValueError(
            f"world size {world_size} must be divisible by context_parallel_size ({cp})"
        )
    if world_size and world_size % ep:
        raise ValueError(f"world size {world_size} must be divisible by expert_model_parallel_size ({ep})")
    if args.checkpoint_save_consolidated not in ("false", "final", "every"):
        raise ValueError("checkpoint_save_consolidated must be false, final, or every")
