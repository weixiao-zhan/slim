# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import ray


def get_nvidia_ld_library_path() -> dict[str, str]:
    """Auto-detect nvidia lib paths from pip-installed packages for LD_LIBRARY_PATH.

    Returns a dict suitable for unpacking into Ray runtime_env env_vars.
    """
    # `nvidia` is a namespace package (no __init__.py), so __file__ is None;
    # resolve its location via __path__ instead.
    try:
        nvidia_bases = list(__import__("nvidia").__path__)
    except (ImportError, AttributeError):
        return {}

    lib_dirs = [d for base in nvidia_bases for d in glob.glob(os.path.join(base, "*/lib"))]
    if not lib_dirs:
        return {}

    return {"LD_LIBRARY_PATH": ":".join(lib_dirs)}


# slim's own defaults for every process of the job; a value already in the driver's environment wins.
# Settings that depend on the cluster or the machine belong in the launch script.
_JOB_ENV_DEFAULTS = {
    # Ray then pickles a CPU tensor as an out-of-band buffer: `ray.get` maps a fetched tensor from the
    # object store without a copy, and a tensor view sends only its own bytes. A fetched tensor shares
    # the memory of its object, so the processes that fetch rollout data never write one in place.
    "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
}


def init_ray() -> None:
    """Connect to Ray with the job-level environment of every slim process."""
    env_vars = {key: os.environ.get(key, value) for key, value in _JOB_ENV_DEFAULTS.items()}
    env_vars |= get_nvidia_ld_library_path()
    # The driver's environment already holds what `ray job submit` passes to every process, and Ray
    # rejects a key that both it and `ray.init` set, so only the values that differ go in.
    ray.init(runtime_env={"env_vars": {key: value for key, value in env_vars.items() if os.environ.get(key) != value}})
