"""Weight-only quantization shared by the offline converter
(tools/convert_hf_to_fp8.py) and the online rollout weight-sync path
(backends/fsdp_utils/update_weight_utils.py).

Both paths quantize through the same code, so the weights the rollout engine
serves during training match what a deployed quantized checkpoint would serve.

`Quantizer.maybe_from_checkpoint` reads a checkpoint's `quantization_config` and
returns the matching quantizer, or None for an unquantized checkpoint. The only
format today is block-FP8 (`QuantizerFP8`, e4m3 with NxK blocks and a per-block
float32 scale named `<weight>.weight_scale_inv`, multiplicative: dequant = q*scale,
matching sglang's block-FP8 GEMM). To add another format (e.g. NVFP4), add a sibling
`Quantizer` subclass and one dispatch branch in `maybe_from_checkpoint`.
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
    """Extract the quantization recipe from a HF `quantization_config` dict.

    Returns a dict with: strategy, block_size, keep_patterns, activation_scheme,
    fmt, scale_fmt.
    """
    qc = quantization_config
    method = qc.get("quant_method")
    recipe = {
        "activation_scheme": qc.get("activation_scheme", "dynamic"),
        "fmt": qc.get("fmt", "e4m3"),
        "scale_fmt": qc.get("scale_fmt"),
    }

    if method == "fp8":
        weight_block_size = qc.get("weight_block_size")
        recipe["strategy"] = "block" if weight_block_size else "tensor"
        recipe["block_size"] = list(weight_block_size) if weight_block_size else None
        recipe["keep_patterns"] = list(qc.get("modules_to_not_convert") or [])
    elif method == "compressed-tensors":
        weights = next(iter(qc["config_groups"].values()))["weights"]
        recipe["strategy"] = weights.get("strategy") or "channel"
        recipe["block_size"] = None
        recipe["keep_patterns"] = list(qc.get("ignore") or [])
    else:
        raise ValueError(f"Unsupported quant_method: {method!r}")

    return recipe


def block_fp8(weight, block_size):
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


def channel_fp8(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
    return qweight, scale


def tensor_fp8(weight):
    scale = weight.abs().amax().clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
    return qweight, scale.view(1)


def quant_fp8(weight, strategy, block_size=None):
    if strategy == "tensor":
        return tensor_fp8(weight)
    if strategy == "channel":
        return channel_fp8(weight)
    return block_fp8(weight, block_size)


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
        # NVFP4: elif method in ("nvfp4", "modelopt_fp4"): return QuantizerNVFP4.from_quant_config(qc)
        return None

    def quantize(self, name, param):
        """Yield (name, tensor) pairs to push for one parameter."""
        raise NotImplementedError


class QuantizerFP8(Quantizer):
    """Block-FP8 quantizer, configured from the rollout checkpoint's own config so the
    engine and the synced weights share one recipe and keep-list by construction.
    """

    def __init__(self, block_size, keep_patterns):
        self.block_size = block_size
        self.is_kept = make_keep_predicate(keep_patterns)

    @classmethod
    def from_quant_config(cls, qc):
        assert qc.get("weight_block_size"), (
            "FP8 weight sync only support per-block. Sglang has special per-tensor/per-channel transformation."
        )
        assert qc.get("activation_scheme", "dynamic") == "dynamic", (
            "FP8 weight sync only support dynamic activation quantization."
        )
        # TODO(blackwell)
        # sglang on Blackwell has to DeepGEMM and Trition backend
        # DeepGEMM only supports 128x128 + UE8M0 scales
        # Triton supports other block size + fp32 scale
        return cls(list(qc["weight_block_size"]), list(qc.get("modules_to_not_convert") or []))

    def quantize(self, name, param):
        """Quantizable 2D weights become `(name, fp8_weight)` plus
        `(name.weight_scale_inv, scale)`; everything else passes through unchanged.
        """
        if not (name.endswith(".weight") and param.dim() == 2 and not self.is_kept(module_name_of(name))):
            return ((name, param),)
        if hasattr(param, "wait"):
            param = param.wait()
        block_n = self.block_size[0]
        assert param.shape[0] % block_n == 0, (
            f"{name}: output dim {param.shape[0]} is not divisible by block size {block_n}"
        )
        qweight, scale = block_fp8(param, self.block_size)
        return ((name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale))
