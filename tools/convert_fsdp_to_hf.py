import argparse
import json
import os
import pickle
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    predefined = [
        "model_state.model.base_model.model.",  # FSDP + PEFT
        "model_state.model.",
        "model_state.",
        "base_model.model.",  # PEFT only
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {k.removeprefix(prefix) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _is_lora_checkpoint(keys: list[str]) -> bool:
    return any(".lora_A." in k for k in keys)


def _merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    lora_alpha: float,
    lora_r: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Merge LoRA adapter weights into base weights and return a clean HF state dict."""
    scaling = lora_alpha / lora_r

    lora_a_suffix = ".lora_A.default.weight"
    lora_modules = {k.removesuffix(lora_a_suffix) for k in state_dict if k.endswith(lora_a_suffix)}

    merged: dict[str, torch.Tensor] = {}
    handled: set[str] = set()

    for module in sorted(lora_modules):
        a_key = f"{module}.lora_A.default.weight"
        b_key = f"{module}.lora_B.default.weight"
        base_key = f"{module}.base_layer.weight"

        lora_a = state_dict[a_key].to(device)
        lora_b = state_dict[b_key].to(device)
        base_w = state_dict[base_key].to(device)

        merged[f"{module}.weight"] = (base_w + (lora_b @ lora_a) * scaling).cpu()
        handled.update([a_key, b_key, base_key])

        # Bias (not affected by LoRA, just rename)
        bias_key = f"{module}.base_layer.bias"
        if bias_key in state_dict:
            merged[f"{module}.bias"] = state_dict[bias_key]
            handled.add(bias_key)

    # Pass through remaining keys (layernorm, embeddings, etc.)
    for key, value in state_dict.items():
        if key not in handled:
            merged[key] = value

    print(f"Merged {len(lora_modules)} LoRA modules (scaling={scaling}).")
    return merged


def _infer_dtype_map(origin_hf_dir: str) -> dict[str, torch.dtype]:
    """Map each parameter name to its dtype in the base model's safetensors files."""
    from safetensors import safe_open

    dtype_map: dict[str, torch.dtype] = {}
    for f in sorted(os.listdir(origin_hf_dir)):
        if not f.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(origin_hf_dir, f), framework="pt") as sf:
            for k in sf.keys():
                dtype_map[k] = sf.get_slice(k).get_dtype()
    # safetensors returns dtype as string (e.g. "BF16"); convert to torch.dtype
    str_to_torch = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }
    return {k: str_to_torch.get(v, v) if isinstance(v, str) else v for k, v in dtype_map.items()}


def _build_hf_model(config: AutoConfig) -> torch.nn.Module:
    print(f"Detected model type: {config.model_type}")
    model_cls = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    print(f"Loaded with {model_cls.__name__}")
    return model_cls.from_config(config, trust_remote_code=True)


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
    peft_config: dict | None = None,
    device: str = "cpu",
) -> None:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    tensor_items = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    del state_dict

    # Use meta device to get target keys without allocating real memory
    config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    with torch.device("meta"):
        hf_model = _build_hf_model(config)
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    total_keys = len(tensor_items)

    print(f"Using prefix '{best_prefix}' for key mapping. " f"Matched {best_match}/{total_keys} parameter keys.")

    model_state = {k.removeprefix(best_prefix): v for k, v in tensor_items.items()}
    del tensor_items

    if not model_state:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )

    # Merge LoRA adapters if this is a PEFT checkpoint
    if _is_lora_checkpoint(list(model_state.keys())):
        if peft_config is None:
            raise ValueError(
                "Detected LoRA checkpoint but --peft-config not provided. "
                "Pass the same --peft-config used during training, e.g. "
                """--peft-config '{"r": 64, "lora_alpha": 128}'"""
            )
        lora_r = peft_config["r"]
        lora_alpha = peft_config.get("lora_alpha", lora_r * 2)
        model_state = _merge_lora_state_dict(model_state, lora_alpha, lora_r, device=device)

    # Cast each tensor to its native dtype in the base model (FSDP stores fp32 masters,
    # but some params like SSM A_log / norm weights must stay fp32).
    dtype_map = _infer_dtype_map(origin_hf_dir)
    fallback = next(iter(dtype_map.values()), torch.bfloat16)
    dtype_counts: dict[torch.dtype, int] = {}
    casted: dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        if v.is_floating_point():
            target = dtype_map.get(k, fallback)
            v = v.to(target)
            dtype_counts[target] = dtype_counts.get(target, 0) + 1
        casted[k] = v
    model_state = casted
    print(f"Per-tensor dtype cast: { {str(d): n for d, n in dtype_counts.items()} }")

    # Validate keys
    merged_keys = set(model_state.keys())
    missing = sorted(target_keys - merged_keys)
    unexpected = sorted(merged_keys - target_keys)
    print(f"Missing keys ({len(missing)}): {missing}")
    print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

    # Load into meta model (assign=True replaces meta tensors, no extra copy)
    hf_model.load_state_dict(model_state, strict=False, assign=True)
    del model_state

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir}")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        src = os.path.join(origin_hf_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isdir(src):
            print(f"copy tree {src} -> {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            print(f"copy {src} -> {dst}")
            shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--peft-config",
        type=str,
        default=None,
        help='JSON string of PEFT/LoRA config used during training, e.g. \'{"r": 64, "lora_alpha": 128}\'',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for LoRA merge matmul (default: cuda if available).",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    peft_config = json.loads(args.peft_config) if args.peft_config else None

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(
        args.origin_hf_dir, model_dir, args.output_dir,
        peft_config=peft_config, device=args.device,
    )
    copy_assets(args.origin_hf_dir, args.output_dir)
