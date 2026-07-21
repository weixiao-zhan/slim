# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE routing-replay buffer (FSDP backend, model-agnostic).

Two layers live here:

1. ``RoutingReplay``: a thread-safe class-level slot holding the active
   token-aligned ``[N_tokens, num_layers, top_k]`` int tensor. Per-model
   router patches read it via ``gather_replayed_topk``; the trainer
   activates/deactivates it around forward+backward.

2. ``gather_replayed_topk``: the gather primitive used by per-arch
   router patches. Returns ``None`` when the buffer is inactive so
   patches can fall back to stock ``torch.topk``. This is the only
   symbol model files need to import from this module.

Lifecycle, per micro-batch forward driven by the trainer:
  1. Trainer calls ``RoutingReplay.activate(routed_experts)``.
  2. Each MoE router's patched forward calls ``gather_replayed_topk``,
     keyed by the per-instance ``_routing_replay_layer_idx`` attribute
     (set by the model's adapter ``register_layer_indices`` walker).
  3. The buffer **stays active across the original forward AND the
     gradient-checkpointing recomputation** that runs during backward.
     Deactivating between forward and backward causes the recompute to
     take a different code path, save a different number of tensors,
     and trip ``CheckpointError``.
  4. Trainer calls ``RoutingReplay.deactivate()`` after the full
     forward+backward (e.g. end of ``train()`` /
     ``compute_log_probs()``) to clear stale state.

Stored as a class attribute (not a thread-local): autograd's recompute
can run on a different thread than the original forward, and we need
both passes to see the same indices.
"""

import threading

import torch


class RoutingReplay:
    _routed_experts: torch.Tensor | None = None
    _lock = threading.Lock()

    @classmethod
    def activate(cls, routed_experts: torch.Tensor) -> None:
        """``routed_experts``: int tensor of shape [N_tokens, num_layers, top_k]."""
        with cls._lock:
            cls._routed_experts = routed_experts

    @classmethod
    def deactivate(cls) -> None:
        with cls._lock:
            cls._routed_experts = None

    @classmethod
    def get(cls) -> torch.Tensor | None:
        return cls._routed_experts


def gather_replayed_topk(
    router_probs: torch.Tensor,
    layer_idx: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Replay primitive for per-arch MoE router patches.

    Returns ``(frozen_indices, normalized_top_value)`` when the replay
    buffer is active, else ``None`` so the caller can fall back to
    stock ``torch.topk``. ``router_probs`` is ``[N_tokens, num_experts]``
    (post-softmax), and the returned ``top_value`` is normalized to sum
    to 1 along the last dim — same convention as HF's stock router.

    The gather is differentiable in ``router_probs``, so the router
    weight still receives gradient; only the *choice* of experts is
    frozen to the rollout-recorded indices.
    """
    routed = RoutingReplay.get()
    if routed is None:
        return None
    frozen = routed[:, layer_idx, :].to(device=router_probs.device, dtype=torch.long)
    assert frozen.shape == (router_probs.shape[0], top_k), (
        f"routing replay expected {(router_probs.shape[0], top_k)}, got {frozen.shape}"
    )
    top_value = router_probs.gather(1, frozen)
    top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return frozen, top_value
