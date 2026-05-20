"""Varlen/packing patches and routing-replay support for HF Qwen3.5 family.

Qwen3.5 / Qwen3.6 share HF's `qwen3_5` implementation, and the MoE variants
share HF's `qwen3_5_moe`. Both are hybrid: most text layers are Mamba-style
gated delta-net, not attention. Stock HF calls `causal_conv1d_fn(...,
seq_idx=None)` and `chunk_gated_delta_rule(..., cu_seqlens=None)` inside those
layers, so recurrence flows across episode boundaries when we pack multiple
episodes into one [1, N] row. State can compound and produce NaN/Inf within a
few layers, crashing the next full-attention layer or checkpoint recompute.

Patches in this module:

1. `apply_qwen_deltanet_varlen_patch()` — patches `*GatedDeltaNet.forward`
   and `*DecoderLayer.forward` (for both `qwen3_5` and `qwen3_5_moe`) so the
   standard HF `cu_seq_lens_q` kwarg flows through to fla's stateless
   `causal_conv1d` / `chunk_gated_delta_rule`, which reset state at each
   boundary. Forward becomes bit-identical to the slime/SGLang rollout.

2. ``Qwen3_5MoeRoutingReplayAdapter`` (registered in
   ``ROUTING_REPLAY_REGISTRY["qwen3_5_moe"]``) — replaces
   ``Qwen3_5MoeTopKRouter.forward`` with a replay-aware variant that
   delegates the gather to ``gather_replayed_topk``. Eliminates
   train/inference expert-selection mismatch. The router weight still
   receives gradient; only the *choice* of experts is frozen.

Ref: upstream slime ``slime_plugins/models/qwen3_5.py`` and
``slime/utils/routing_replay.py`` do the same under Megatron; we replicate
under stock HF for FSDP.
"""

import torch
import torch.nn.functional as F
from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

from ..routing_replay import gather_replayed_topk
from . import ROUTING_REPLAY_REGISTRY

_PATCHED = False
_ROUTER_REPLAY_PATCHED = False


def _patched_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Drop-in replacement for Qwen3_5GatedDeltaNet.forward with varlen support."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    mixed_qkv, _ = fla_causal_conv1d(
        x=mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        cu_seqlens=cu_seqlens,
    )
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.float().sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = fla_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


def _patched_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Thread `cu_seq_lens_q` into Qwen DeltaNet layers."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


def _patched_moe_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Thread `cu_seq_lens_q` into Qwen3.5-MoE DeltaNet layers; preserves the
    MoE block's tuple-output unpacking that stock Qwen3_5MoeDecoderLayer does."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    return residual + hidden_states


def apply_qwen_deltanet_varlen_patch() -> None:
    """Monkey-patch HF Qwen3.5/3.6 (dense + MoE) DeltaNet to honor packed sequences."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
    )

    Qwen3_5GatedDeltaNet.forward = _patched_gated_delta_net_forward
    Qwen3_5DecoderLayer.forward = _patched_decoder_layer_forward

    # MoE variant: same DeltaNet kernels, different decoder shell (MoE block
    # returns tuples). Patch the moe-namespaced classes with the moe-aware
    # decoder forward.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeGatedDeltaNet,
    )

    Qwen3_5MoeGatedDeltaNet.forward = _patched_gated_delta_net_forward
    Qwen3_5MoeDecoderLayer.forward = _patched_moe_decoder_layer_forward

    _PATCHED = True


# ----------------------------------------------------------------------------
# MoE routing replay (Qwen3.5-MoE specific bits; the cross-model
# RoutingReplay buffer lives in ``slim.backends.fsdp_utils.routing_replay``).
# ----------------------------------------------------------------------------


def _patched_qwen3_5_moe_router_forward(self, hidden_states):
    """Replay-aware replacement for ``Qwen3_5MoeTopKRouter.forward``.

    Stock forward (qwen3_5_moe v5.6.0):
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

    # Always run topk so the saved-tensor pattern matches stock; cheap relative
    # to softmax. Without this, gradient checkpointing recompute can drift
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
    actually MoE — non-MoE layers get a slice of the buffer that's never read.
    The sglang capturer follows the same convention.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

    count = 0
    # Walk the decoder layers of the underlying text model.
    text_model = getattr(model, "model", model)
    text_model = getattr(text_model, "language_model", text_model)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        # Try one more level: HF wraps things differently across vlm/causal-lm.
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
