"""Optional Megatron training backend."""

from __future__ import annotations

from .arguments import megatron_parse_args, validate_args


def __getattr__(name: str):
    if name == "MegatronTrainer":
        from .trainer import MegatronTrainer

        return MegatronTrainer
    raise AttributeError(name)


__all__ = ["MegatronTrainer", "megatron_parse_args", "validate_args"]
