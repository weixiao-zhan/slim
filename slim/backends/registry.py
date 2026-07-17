"""Lazy training-backend entry points."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Any


TRAINING_BACKENDS = ("fsdp", "megatron")


@dataclass(frozen=True)
class BackendSpec:
    argument_parser: str
    validator: str | None
    trainer_classes: dict[str, str]


_BACKEND_SPECS = {
    "fsdp": BackendSpec(
        argument_parser="slim.backends.fsdp_utils.arguments:fsdp_parse_args",
        validator=None,
        trainer_classes={
            "actor": "slim.backends.fsdp_utils.actor:ActorFSDPTrainer",
            "critic": "slim.backends.fsdp_utils.critic:CriticFSDPTrainer",
        },
    ),
    "megatron": BackendSpec(
        argument_parser="slim.backends.megatron.arguments:megatron_parse_args",
        validator="slim.backends.megatron.arguments:validate_args",
        trainer_classes={
            "actor": "slim.backends.megatron.trainer:MegatronTrainer",
        },
    ),
}


def _load_symbol(target: str) -> Any:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"Invalid backend entry point {target!r}.")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def get_training_backend(argv: list[str] | None = None) -> str:
    """Read only the backend selector without importing a backend package."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--training-backend", choices=TRAINING_BACKENDS, default="fsdp")
    args, _ = parser.parse_known_args(argv)
    return args.training_backend


def parse_backend_args(extra_args_provider=None, ignore_unknown_args: bool = False):
    """Parse CLI arguments with the selected backend's lazy parser."""

    backend = get_training_backend()
    parser = _load_symbol(_BACKEND_SPECS[backend].argument_parser)
    args = parser(
        extra_args_provider=extra_args_provider,
        ignore_unknown_args=ignore_unknown_args,
    )
    parsed_backend = getattr(args, "training_backend", backend)
    if parsed_backend != backend:
        raise ValueError(
            f"Backend parser mismatch: selected {backend!r}, parsed {parsed_backend!r}."
        )
    args.training_backend = backend
    return args


def validate_backend_args(args) -> None:
    """Run validation owned by the selected backend."""

    backend = getattr(args, "training_backend", "fsdp")
    try:
        spec = _BACKEND_SPECS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown training backend {backend!r}.") from exc
    if spec.validator is not None:
        _load_symbol(spec.validator)(args)


def get_trainer_class(backend: str, role: str):
    """Resolve the Ray trainer class for a backend and role on demand."""

    try:
        spec = _BACKEND_SPECS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown training backend {backend!r}.") from exc
    try:
        target = spec.trainer_classes[role]
    except KeyError as exc:
        supported = ", ".join(sorted(spec.trainer_classes))
        raise ValueError(
            f"Training backend {backend!r} does not support role {role!r}; "
            f"supported roles: {supported}."
        ) from exc
    return _load_symbol(target)


__all__ = [
    "TRAINING_BACKENDS",
    "get_trainer_class",
    "get_training_backend",
    "parse_backend_args",
    "validate_backend_args",
]
