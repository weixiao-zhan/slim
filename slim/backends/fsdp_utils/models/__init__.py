"""Per-arch HF model patches owned by the FSDP backend.

Two orthogonal concerns live here, dispatched by HF ``model_type``:

1. **Pre-build class patches** (``apply_hf_model_patches``): repo-owned monkey
   patches that must replace HF class methods *before* ``from_pretrained``
   instantiates any module — varlen DeltaNet for Qwen3.5, MoE router replay
   for ``qwen3_5_moe``, etc. These are short-lived; once HF upstream fixes
   the underlying gaps (e.g. issue #42638 for routing replay) the patch can
   be retired by deleting the sibling file.

2. **Routing-replay adapters** (``ROUTING_REPLAY_REGISTRY``): per-arch hooks
   used by the trainer at runtime to stamp layer indices on built routers.
   See ``RoutingReplayAdapter``.

Adding a new arch: drop a new file under this package, register an adapter
in ``ROUTING_REPLAY_REGISTRY`` and/or extend ``apply_hf_model_patches`` to
call its class-level patcher, then add a top-level import below so
registration runs.
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

    # Qwen3.5 / Qwen3.6 (dense + MoE) share Mamba-style linear_attention layers.
    # Stock HF doesn't honor packed-sequence boundaries; patch before any
    # decoder layer is built. The MoE variant uses model_type "qwen3_5_moe".
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        from .qwen3_5 import apply_qwen_deltanet_varlen_patch

        apply_qwen_deltanet_varlen_patch()

    # MoE routing replay: look up a per-arch adapter and apply its router
    # patch now. Currently only Qwen3.5-MoE is wired; other archs return
    # None and the trainer treats replay as off.
    if not getattr(args, "use_rollout_routing_replay", False):
        return None

    adapter = ROUTING_REPLAY_REGISTRY.get(model_type)
    if adapter is None:
        raise ValueError(
            f"--use-rollout-routing-replay set but no adapter registered for "
            f"model_type={model_type!r}. Supported: {sorted(ROUTING_REPLAY_REGISTRY)}"
        )
    adapter.apply_patch()
    return adapter
