"""MoE routing-replay buffer (FSDP backend, model-agnostic).

This module owns the cross-model state for replaying rollout-time MoE
expert selections during training. Per-model router patches (e.g. the
Qwen3.5-MoE patch under ``models/qwen3_5.py``) read the buffer via
``RoutingReplay.get()`` and gather scores at the rollout-recorded
indices instead of running their own ``torch.topk``.

Lifecycle, per micro-batch forward driven by the trainer:
  1. Trainer calls ``RoutingReplay.activate(routed_experts)`` with a
     token-aligned ``[N_tokens, num_layers, top_k]`` int tensor.
  2. Each MoE router's patched forward looks up its decoder-layer index
     via the per-instance ``_routing_replay_layer_idx`` attribute (set
     by the model's ``register_routing_replay_layer_indices`` walker)
     and ``gather``s scores at those indices.
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
