"""Per-architecture routing replay support for the FSDP backend.

Adapters are selected by the Hugging Face ``model_type`` and can patch model
classes before construction or register metadata on model instances.
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
    """Apply HF class-level patches selected by ``hf_config.model_type``.

    Must run *before* ``from_pretrained`` instantiates any module: HF binds
    method references at construction time, so replacing classes after the
    fact only affects future instances.

    Returns the routing-replay adapter (or ``None``) so the trainer can
    later call ``adapter.register_layer_indices(model)`` once the model is
    built. Keeping the adapter handle in the trainer avoids a second
    registry lookup at instance-stamp time.
    """
    model_type = getattr(hf_config, "model_type", None)

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
