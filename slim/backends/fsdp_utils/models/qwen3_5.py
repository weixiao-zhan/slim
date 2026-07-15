"""Routing replay support for Hugging Face Qwen3.5 MoE models."""

import torch
import torch.nn.functional as F

from ..routing_replay import gather_replayed_topk
from . import ROUTING_REPLAY_REGISTRY

_ROUTER_REPLAY_PATCHED = False


def _patched_qwen3_5_moe_router_forward(self, hidden_states):
    """Replay-aware replacement for ``Qwen3_5MoeTopKRouter.forward``.

    Stock forward:
        logits = F.linear(hidden_states, self.weight)
        probs = softmax(logits, fp32)
        top_value, indices = topk(probs, top_k)
        top_value /= top_value.sum(-1, keepdim=True)
        return logits, top_value.to(dtype), indices

    Replay forward: identical *autograd shape* (same number/types/shapes of
    saved tensors) so gradient checkpointing's recomputation matches. We
    still call ``torch.topk`` to keep its saved-tensor footprint, then
    delegate the actual gather to ``gather_replayed_topk``.
    """
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)

    # Always run topk so the saved-tensor pattern matches stock. Without this,
    # gradient checkpointing recompute can drift
    # against the original forward and trip CheckpointError.
    stock_top_value, stock_indices = torch.topk(router_probs, self.top_k, dim=-1)

    layer_idx = getattr(self, "_routing_replay_layer_idx", None)
    replayed = (
        gather_replayed_topk(router_probs, layer_idx, self.top_k)
        if layer_idx is not None
        else None
    )
    if replayed is None:
        top_value = stock_top_value / stock_top_value.sum(dim=-1, keepdim=True)
        return router_logits, top_value.to(router_logits.dtype), stock_indices

    frozen, top_value = replayed
    return router_logits, top_value.to(router_logits.dtype), frozen


def _apply_qwen3_5_moe_router_replay_patch() -> None:
    global _ROUTER_REPLAY_PATCHED
    if _ROUTER_REPLAY_PATCHED:
        return

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

    Qwen3_5MoeTopKRouter.forward = _patched_qwen3_5_moe_router_forward
    _ROUTER_REPLAY_PATCHED = True


def _register_qwen3_5_moe_layer_indices(model) -> int:
    """Stamp each ``Qwen3_5MoeTopKRouter`` instance with its decoder-layer index.

    Returns the number of routers tagged. The replay tensor is laid out as
    ``[N_tokens, num_hidden_layers, top_k]`` regardless of which layers are
    MoE. Non-MoE layer slices are not read.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

    count = 0
    text_model = getattr(model, "model", model)
    text_model = getattr(text_model, "language_model", text_model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        text_model = getattr(text_model, "model", text_model)
        layers = getattr(text_model, "layers", [])
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None) if mlp is not None else None
        if isinstance(gate, Qwen3_5MoeTopKRouter):
            gate._routing_replay_layer_idx = layer_idx
            count += 1
    return count


class Qwen3_5MoeRoutingReplayAdapter:
    def apply_patch(self) -> None:
        _apply_qwen3_5_moe_router_replay_patch()

    def register_layer_indices(self, model) -> int:
        return _register_qwen3_5_moe_layer_indices(model)


ROUTING_REPLAY_REGISTRY["qwen3_5_moe"] = Qwen3_5MoeRoutingReplayAdapter()
