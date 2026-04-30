"""Varlen/packing patch for HuggingFace Qwen DeltaNet models.

Qwen3.5 and Qwen3.6 currently share HF's `qwen3_5` implementation. They are
hybrid models: most text layers are Mamba-style gated delta-net
(`Qwen3_5GatedDeltaNet`), not attention. Stock HF calls
`causal_conv1d_fn(..., seq_idx=None)` and `chunk_gated_delta_rule(...,
cu_seqlens=None)` inside those layers, which means recurrence flows across
episode boundaries when we pack multiple episodes into one [1, N] row. State
can compound across boundaries and produce NaN/Inf within a few layers,
crashing the next full-attention layer or checkpoint recompute.

This module monkey-patches `Qwen3_5DecoderLayer.forward` and
`Qwen3_5GatedDeltaNet.forward` so that the standard HF `cu_seq_lens_q` kwarg
flows through to the recurrent kernels, which reset state at each boundary.

Ref: upstream slime `slime_plugins/models/qwen3_5.py` does the same under
Megatron; we replicate the effect under stock HF for FSDP.
"""

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

_PATCHED = False


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

    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state
        and seq_len == 1
        and cache_position is not None
    )
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state

        if cu_seqlens is not None and batch_size == 1:
            # Per-episode conv so kernel taps do not cross packed boundaries.
            k = self.conv_kernel_size
            weight = self.conv1d.weight
            pieces = []
            for i in range(cu_seqlens.numel() - 1):
                s = int(cu_seqlens[i].item())
                e = int(cu_seqlens[i + 1].item())
                n = e - s
                if n == 0:
                    continue
                chunk = mixed_qkv[:, :, s:e]
                y = F.conv1d(chunk, weight, bias=self.conv1d.bias, padding=k - 1, groups=weight.shape[0])
                pieces.append(y[:, :, :n])
            mixed_qkv = torch.cat(pieces, dim=-1) if pieces else mixed_qkv[:, :, :0]
            mixed_qkv = F.silu(mixed_qkv)
        elif self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        if cu_seqlens is not None:
            core_attn_out, last_recurrent_state = fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        if hasattr(cache_params, "update_recurrent_state"):
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        else:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

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


def apply_qwen_deltanet_varlen_patch() -> None:
    """Monkey-patch HF Qwen3.5/Qwen3.6 DeltaNet classes to honor packed sequences."""
    global _PATCHED
    if _PATCHED:
        return

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
    )

    Qwen3_5GatedDeltaNet.forward = _patched_gated_delta_net_forward
    Qwen3_5DecoderLayer.forward = _patched_decoder_layer_forward
    _PATCHED = True


def apply_qwen3_5_varlen_patch() -> None:
    """Backward-compatible alias for older imports."""
    apply_qwen_deltanet_varlen_patch()
