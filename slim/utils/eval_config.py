# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_overrides must be a mapping.")
    return value


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: str | None = None

    eval_n_samples_per_prompt: int | None = None

    max_context_len: int | None = None

    # per-dataset custom generate function (e.g., for tool calling)
    custom_generate_function_path: str | None = None

    metadata_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata_overrides = _ensure_metadata_overrides(self.metadata_overrides)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        return (self.name, self.path)

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return updated metadata merging overrides."""
        if not isinstance(sample_metadata, dict):
            metadata = {}
        else:
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            metadata[key] = value

        return metadata


def ensure_dataset_list(config: Any) -> list[dict[str, Any]]:
    """
    Normalize OmegaConf containers into a list of dicts.
    Accepts either a list or dictionary keyed by dataset name.
    """
    if config is None:
        return []

    if isinstance(config, dict):
        datasets = []
        for name, cfg in config.items():
            dataset = dict(cfg or {})
            dataset.setdefault("name", name)
            datasets.append(dataset)
        return datasets

    if isinstance(config, (list, tuple)):
        datasets = []
        for item in config:
            dataset = dict(item or {})
            if "name" not in dataset:
                raise ValueError("Each evaluation dataset entry must include a `name` field.")
            datasets.append(dataset)
        return datasets

    raise TypeError("eval.datasets must be either a list or a mapping.")


def build_eval_dataset_configs(
    args: Any,
    raw_config: Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[EvalDatasetConfig]:
    defaults = defaults or {}
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        cfg_dict = dict(cfg or {})
        cfg_dict.setdefault(
            "eval_n_samples_per_prompt", defaults.get("eval_n_samples_per_prompt", args.eval_n_samples_per_prompt)
        )
        cfg_dict.setdefault("max_context_len", defaults.get("max_context_len", args.max_context_len))
        dataset = EvalDatasetConfig(**cfg_dict)
        datasets.append(dataset)
    return datasets
