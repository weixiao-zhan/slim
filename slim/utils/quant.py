# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-only quantization shared by the offline converter
(tools/convert_hf_to_fp8.py) and the online rollout weight-sync path
(backends/nemo/update_weight_utils.py).

Both paths quantize through the same `block_fp8`, so the weights the rollout engine
serves during training match what a deployed quantized checkpoint would serve.

`Quantizer.maybe_from_checkpoint` reads a checkpoint's `quantization_config` and
returns the matching quantizer, or None for an unquantized checkpoint. The only
format is block-FP8 (`QuantizerFP8`): e4m3 weights with 128x128 blocks and a
per-block scale named `<weight>.weight_scale_inv`, multiplicative (dequant = q*scale),
matching sglang's block-FP8 GEMM. Per-tensor / per-channel FP8 are intentionally
not supported (sglang applies special load-time transforms for those that the
online sync path can't replicate).

Two scale formats:
  * fp32  (scale_fmt=None)   -- a full float32 per-block scale. The default; consumed
                               by sglang's Triton block-FP8 GEMM on any GPU.
  * ue8m0 (scale_fmt="ue8m0")-- DeepSeek-V3.1 / DeepGEMM-on-Blackwell style: each block
                               scale is rounded UP to a power of two. On disk the scale
                               is still a float32 power-of-two (sglang re-packs it at
                               load). The ONLINE sync path additionally packs it into the
                               int32 MN-major TMA-aligned layout DeepGEMM consumes at
                               runtime, because the sync path does not re-run sglang's
                               `process_weights_after_loading` (see QuantizerFP8.quantize).
"""

import torch
import torch.nn.functional as F

FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min


def ceildiv(a, b):
    return -(-a // b)


def module_name_of(key):
    """Strip a trailing `.weight` to recover the owning module name."""
    suffix = ".weight"
    return key[: -len(suffix)] if key.endswith(suffix) else key


def module_path_match(pattern, module_name):
    """Whether `pattern` matches `module_name` on dotted-path boundaries.

    Identical to sglang's `is_layer_skipped` matching (srt/layers/quantization/
    utils.py), so slim and the rollout engine agree on which modules are kept:
    a pattern matches as a full segment-run anywhere in the path, which lets short
    patterns (`linear_attn`, `visual`) and fully-qualified ones both work.
    """
    return (
        pattern == module_name
        or module_name.startswith(pattern + ".")
        or ("." + pattern + ".") in ("." + module_name + ".")
    )


def make_keep_predicate(keep_patterns):
    """Build an `is_kept(module_name)` predicate from a `modules_to_not_convert` list."""

    def is_kept(module_name):
        return any(module_path_match(p, module_name) for p in keep_patterns)

    return is_kept


def parse_quant_recipe(quantization_config):
    """Extract the block-FP8 quantization recipe from a HF `quantization_config` dict.

    Returns a dict with: block_size, keep_patterns, activation_scheme, fmt, scale_fmt.
    Only `quant_method == "fp8"` with a `weight_block_size` is supported.
    """
    qc = quantization_config
    method = qc.get("quant_method")
    if method != "fp8":
        raise ValueError(f"Unsupported quant_method: {method!r} (only block-FP8 is supported)")

    weight_block_size = qc.get("weight_block_size")
    if not weight_block_size:
        raise ValueError("Block-FP8 requires `weight_block_size` in quantization_config.")

    return {
        "block_size": list(weight_block_size),
        "keep_patterns": list(qc.get("modules_to_not_convert") or []),
        "activation_scheme": qc.get("activation_scheme", "dynamic"),
        "fmt": qc.get("fmt", "e4m3"),
        "scale_fmt": qc.get("scale_fmt"),
    }


def _block_fp8_fp32(weight, block_size):
    """128x128 block e4m3 quant with full float32 per-block scales (slim's own recipe).

    Validated bit-for-bit against the official Qwen3.6-27B-FP8 checkpoint.
    """
    block_n, block_k = block_size
    shape_0, shape_1 = weight.shape
    n_tiles, k_tiles = ceildiv(shape_0, block_n), ceildiv(shape_1, block_k)

    padded = F.pad(weight, (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0))
    tiled = padded.reshape(n_tiles, block_n, k_tiles, block_k)
    scale = tiled.abs().amax(dim=(1, 3), keepdim=True).to(torch.float32).clamp(min=1e-12) / FP8_MAX
    qweight = (
        (tiled / scale)
        .clamp(FP8_MIN, FP8_MAX)
        .reshape(n_tiles * block_n, k_tiles * block_k)
        .to(torch.float8_e4m3fn)[:shape_0, :shape_1]
        .contiguous()
    )
    return qweight, scale.reshape(n_tiles, k_tiles)


def _block_fp8_ue8m0(weight, block_size):
    """128x128 block e4m3 quant with power-of-two (UE8M0) per-block scales.

    Delegates to sglang's `per_block_cast_to_fp8` so the e4m3 weights and the
    fp32 power-of-two scales are bit-identical to what the rollout engine itself
    produces (the engine requants its own checkpoints through the same function).
    The returned scale is a float32 power of two with shape (n_tiles, k_tiles);
    the online path packs it into DeepGEMM's int32 layout (see QuantizerFP8).
    """
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    if list(block_size) != [128, 128]:
        raise ValueError(f"ue8m0 scales require 128x128 blocks, got {block_size}")
    # per_block_cast_to_fp8 expects a 2D tensor; it pads to 128-multiples internally
    # and crops the weight back to the original shape. Scale shape: (n/128, k/128).
    qweight, scale = per_block_cast_to_fp8(weight)
    return qweight.contiguous(), scale


def block_fp8(weight, block_size, scale_fmt=None):
    """Quantize a 2D weight to block-FP8. Returns (e4m3 weight, float32 scale).

    scale_fmt=None  -> full float32 scales.
    scale_fmt="ue8m0" -> power-of-two (float32) scales, matching DeepGEMM on Blackwell.
    """
    if scale_fmt == "ue8m0":
        return _block_fp8_ue8m0(weight, block_size)
    if scale_fmt is not None:
        raise ValueError(f"Unsupported scale_fmt: {scale_fmt!r}")
    return _block_fp8_fp32(weight, block_size)


def pack_ue8m0_scale_for_engine(scale, mn):
    """Pack a float32 power-of-two block scale into DeepGEMM's runtime layout
    (int32, MN-major, TMA-aligned, 4 UE8M0 bytes per int32).

    Reuses sglang's `transform_scale_ue8m0` so the packed bytes match the engine
    exactly. Only needed on the online sync path on Blackwell, where the engine
    consumes scales without re-running its load-time requant.
    """
    from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

    return transform_scale_ue8m0(scale, mn=mn)


def _engine_expects_packed_ue8m0():
    """True iff the local rollout engine consumes int32-packed UE8M0 scales at runtime
    (DeepGEMM on Blackwell). On Hopper/Ada the engine keeps fp32 power-of-two scales.
    """
    try:
        from sglang.srt.layers.deep_gemm_wrapper.configurer import DEEPGEMM_SCALE_UE8M0

        return bool(DEEPGEMM_SCALE_UE8M0)
    except ImportError:
        return False


class Quantizer:
    """Quantizes one BF16 training weight into the tensors to push to the rollout engine.

    Subclasses implement `quantize`; `maybe_from_checkpoint` is the factory that
    selects the subclass from a checkpoint's `quantization_config`.
    """

    @classmethod
    def maybe_from_checkpoint(cls, hf_checkpoint):
        """Build the quantizer matching `hf_checkpoint`'s config, or None if unquantized."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
        qc = config.to_dict().get("quantization_config")
        if not qc:
            return None
        method = qc.get("quant_method")
        if method == "fp8":
            return QuantizerFP8.from_quant_config(qc)
        return None

    def quantize(self, name, param):
        """Yield (name, tensor) pairs to push for one parameter."""
        raise NotImplementedError


class QuantizerFP8(Quantizer):
    """Block-FP8 quantizer, configured from the rollout checkpoint's own config so the
    engine and the synced weights share one recipe and keep-list by construction.
    """

    def __init__(self, block_size, keep_patterns, scale_fmt=None):
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.is_kept = make_keep_predicate(keep_patterns)
        # Packing the UE8M0 scale into DeepGEMM's int32 layout is required ONLY when the
        # local engine consumes that layout at runtime (Blackwell). On Hopper/Ada the
        # engine keeps fp32 power-of-two scales, so we leave the scale unpacked.
        self._pack_ue8m0 = scale_fmt == "ue8m0" and _engine_expects_packed_ue8m0()

    @classmethod
    def from_quant_config(cls, qc):
        assert qc.get("weight_block_size"), (
            "FP8 weight sync only supports per-block. sglang applies special "
            "per-tensor/per-channel transforms at load that the sync path can't replicate."
        )
        assert qc.get("activation_scheme", "dynamic") == "dynamic", (
            "FP8 weight sync only supports dynamic activation quantization."
        )
        scale_fmt = qc.get("scale_fmt")
        assert scale_fmt in (None, "ue8m0"), f"Unsupported scale_fmt: {scale_fmt!r}"
        if scale_fmt == "ue8m0":
            assert list(qc["weight_block_size"]) == [128, 128], (
                "ue8m0 scales require 128x128 blocks (DeepGEMM constraint)."
            )
        return cls(
            list(qc["weight_block_size"]),
            list(qc.get("modules_to_not_convert") or []),
            scale_fmt=scale_fmt,
        )

    def quantize(self, name, param):
        """Quantizable 2D weights become `(name, fp8_weight)` plus
        `(name.weight_scale_inv, scale)`; everything else passes through unchanged.

        For ue8m0 on Blackwell the scale is packed into DeepGEMM's int32 layout, because
        the engine's `update_weights_from_tensor` -> `load_weights` path does NOT re-run
        `process_weights_after_loading` (which is where it would otherwise pack scales).
        """
        if not (name.endswith(".weight") and param.dim() == 2 and not self.is_kept(module_name_of(name))):
            return ((name, param),)
        if hasattr(param, "wait"):
            param = param.wait()
        block_n = self.block_size[0]
        assert param.shape[0] % block_n == 0, (
            f"{name}: output dim {param.shape[0]} is not divisible by block size {block_n}"
        )
        qweight, scale = block_fp8(param, self.block_size, scale_fmt=self.scale_fmt)
        if self._pack_ue8m0:
            scale = pack_ue8m0_scale_for_engine(scale, mn=qweight.shape[-2])
        return ((name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale))
