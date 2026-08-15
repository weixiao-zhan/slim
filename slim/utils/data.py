# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re

try:
    import ray
except ModuleNotFoundError:  # pragma: no cover
    ray = None

from .timer import Timer

__all__ = ["load_hf_dataset", "process_rollout_data"]

logger = logging.getLogger(__name__)


def _parse_generalized_path(path: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", path)) is None:
        return path, None

    start = int(x) if (x := m.group("start")) != "" else None
    end = int(x) if (x := m.group("end")) != "" else None
    return m.group("real_path"), slice(start, end)


def load_hf_dataset(path: str):
    import datasets as hf_datasets

    real_path, row_slice = _parse_generalized_path(path)

    # Directory-based HF dataset: require explicit split via "path:split" syntax
    split_name = None
    if ":" in real_path:
        real_path, split_name = real_path.rsplit(":", 1)

    if os.path.isdir(real_path):
        loaded = hf_datasets.load_from_disk(real_path)
        if isinstance(loaded, hf_datasets.DatasetDict):
            if split_name is None:
                raise ValueError(
                    f"Path {real_path} is a DatasetDict with splits {list(loaded.keys())}. "
                    f"Specify a split with 'path:split_name' syntax (e.g., '{real_path}:train')."
                )
            if split_name not in loaded:
                raise ValueError(f"Split '{split_name}' not found in {real_path}. Available: {list(loaded.keys())}")
            dataset = loaded[split_name]
        else:
            dataset = loaded
    elif real_path.endswith(".jsonl"):
        dataset = hf_datasets.load_dataset("json", data_files=real_path, split="train")
    elif real_path.endswith(".parquet"):
        dataset = hf_datasets.load_dataset("parquet", data_files=real_path, split="train")
    else:
        raise ValueError(f"Unsupported file format: {real_path}. Supported formats are .jsonl, .parquet, or a directory.")

    if row_slice is not None:
        logger.info("load_hf_dataset path=%s applying slice row_slice=%s", real_path, row_slice)
        dataset = dataset.select(range(*row_slice.indices(len(dataset))))
    return dataset


def process_rollout_data(rollout_data_refs, dp_rank, dp_size):
    """Fetch this DP rank's trajectory batch."""
    assert len(rollout_data_refs) == dp_size
    if ray is None:
        raise ModuleNotFoundError("ray is required to process rollout data")
    batch = ray.get(rollout_data_refs[dp_rank])
    Timer().seq_lens = [len(trajectory.token_ids) for trajectory in batch.trajectories]
    return batch
