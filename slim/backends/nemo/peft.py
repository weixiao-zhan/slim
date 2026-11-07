# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent LoRA support over AutoModel's PEFT modules."""

from __future__ import annotations

import contextlib

import torch
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA

# AutoModel's LoRA modules; the Triton, patched, and MXFP4 variants subclass them.
# Every forward path, the fused LoRA MLP included, reads the module's `scale` at call time.
LORA_MODULES = (LinearLoRA, GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA)


@contextlib.contextmanager
def lora_disabled(model: torch.nn.Module):
    """Run `model` as its base for the block by zeroing the scale of every LoRA module."""
    modules = [module for module in model.modules() if isinstance(module, LORA_MODULES)]
    scales = [module.scale for module in modules]
    for module in modules:
        module.scale = 0.0
    try:
        yield
    finally:
        for module, scale in zip(modules, scales, strict=True):
            module.scale = scale


def merged_lora_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the effective weights $W + s B A$ of one LoRA module by parameter name.

    AutoModel's `materialize_effective_weight(s)` computes them from the sharded tensors in the adapters' dtype.
    """
    if isinstance(module, LinearLoRA):
        return {"weight": module.materialize_effective_weight()}
    # The DeepEP grouped experts hold the same adapter tensors as `GroupedExpertsLoRA`.
    merged = GroupedExpertsLoRA.materialize_effective_weights(module)
    return dict(zip(("gate_and_up_projs", "down_projs"), merged, strict=True))
