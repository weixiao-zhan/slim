# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-architecture Hugging Face model patches owned by the FSDP backend.

Two concerns live here, dispatched by Hugging Face ``model_type``:

1. Pre-build class patches replace model methods before ``from_pretrained``
   constructs the modules.
2. Routing-replay adapters stamp runtime metadata on built model instances.

New architectures register an adapter in ``ROUTING_REPLAY_REGISTRY`` and can
expose a class patch through ``apply_hf_model_patches``.
"""

from typing import Protocol


class RoutingReplayAdapter(Protocol):
    """Per-arch hook for MoE routing replay.

    ``apply_patch`` runs once before any model is built (it monkey-patches
    HF classes). ``register_layer_indices`` runs once per built model and
    stamps each MoE router instance with its decoder-layer index, returning
    the count of tagged routers for logging.
    """

    def apply_patch(self) -> None: ...

    def register_layer_indices(self, model) -> int: ...


ROUTING_REPLAY_REGISTRY: dict[str, RoutingReplayAdapter] = {}


# Trigger registration of each arch's adapter.
from . import qwen3_5  # noqa: E402, F401


def apply_hf_model_patches(hf_config, args) -> "RoutingReplayAdapter | None":
    """Apply class-level patches selected by ``hf_config.model_type``.

    This must run before ``from_pretrained`` constructs the model.

    Returns the routing-replay adapter (or ``None``) so the trainer can
    later call ``adapter.register_layer_indices(model)`` once the model is
    built. Keeping the adapter handle in the trainer avoids a second
    registry lookup at instance-stamp time.
    """
    model_type = getattr(hf_config, "model_type", None)

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        qwen3_5.apply_qwen3_5_vision_packing_patch()

    # Routing replay is currently implemented for Qwen3.5 MoE.
    if not args.use_rollout_routing_replay:
        return None

    adapter = ROUTING_REPLAY_REGISTRY.get(model_type)
    if adapter is None:
        raise ValueError(
            f"--use-rollout-routing-replay set but no adapter registered for "
            f"model_type={model_type!r}. Supported: {sorted(ROUTING_REPLAY_REGISTRY)}"
        )
    adapter.apply_patch()
    return adapter
