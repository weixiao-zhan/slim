# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FSDP wrapping and device/offload infrastructure for the FSDP backend."""

import logging
import os

import ray
import torch

logger = logging.getLogger(__name__)

# For Vision/audio encoder module paths.
# These are replicated (not FSDP-sharded) and frozen for mixed modality batches
# Matched as substrings of module names.
_REPLICATED_FROZEN_MODULE_KEYWORDS = ("visual", "vision_tower", "vision_model", "audio", "speech")


def _get_replicated_frozen_module_roots(model):
    """Top-level replicated+frozen modules.

    Returns a name -> module dict of the outermost matches.
    """
    roots = {}
    for name, module in model.named_modules():
        if any(kw in name for kw in _REPLICATED_FROZEN_MODULE_KEYWORDS):
            if not any(name.startswith(f"{r}.") for r in roots):
                roots[name] = module
    return roots


def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (compute_dtype)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # PeftModel doesn't expose _no_split_modules, so unwrap to find which layer classes FSDP should shard
    base_hf = getattr(model, "base_model", model)
    base_hf = getattr(base_hf, "model", base_hf)
    layer_cls_to_wrap = getattr(base_hf, "_no_split_modules", None)
    assert layer_cls_to_wrap and next(iter(layer_cls_to_wrap)) is not None
    model_config = getattr(base_hf, "config", model.config)

    modules = []
    ignored_params = set()
    for name, m in model.named_modules():
        replicated = any(kw in name for kw in _REPLICATED_FROZEN_MODULE_KEYWORDS)
        if replicated:
            for p in m.parameters():
                p.requires_grad_(False)
                ignored_params.add(p)
        elif type(m).__name__ in layer_cls_to_wrap:
            modules.append(m)
        elif isinstance(m, torch.nn.Embedding) and not model_config.tie_word_embeddings:
            modules.append(m)

    # Determine precision policy. None lets compute follow storage dtype
    # (per-tensor; FSDP2 keeps the param dtype when param_dtype=None).
    _COMPUTE_DTYPE_MAP = {None: None, "bf16": torch.bfloat16, "fp16": torch.float16}
    param_dtype = _COMPUTE_DTYPE_MAP[args.compute_dtype]
    reduce_dtype = torch.float32

    mesh_desc = f"{mesh.ndim}D mesh={mesh.shape}" if mesh is not None else "default"
    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, {mesh_desc}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
        "ignored_params": ignored_params if ignored_params else None,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model
