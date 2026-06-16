"""Quantize a BF16 HF safetensors model to block-FP8, mimicking a reference recipe.

The recipe (block size, activation_scheme, scale_fmt, and the exact
`modules_to_not_convert` keep-list) is always read from a reference FP8 model's
`quantization_config` via --ref-config, so the produced checkpoint matches the
target FP8 model's layer selection exactly. Explicit --block-size / --scale-fmt
override the corresponding values derived from the reference config.

Only block-FP8 is supported (e4m3, NxK blocks, per-block `*.weight_scale_inv`),
matching slim's online rollout weight-sync. Per-tensor / per-channel FP8 are not
supported.

python tools/convert_hf_to_fp8.py --model-dir MODEL_DIR --save-dir SAVE_DIR --ref-config REF_CONFIG
                           [--block-size [BLOCK_SIZE ...]] [--scale-fmt {ue8m0}] [--max-workers MAX_WORKERS]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to the directory of the HF safetensors model (BF16 source).
  --save-dir SAVE_DIR   Path to the directory to save the converted model.
  --ref-config REF_CONFIG
                        Reference FP8 model dir (or its config.json) whose
                        `quantization_config` defines the recipe and keep-list. Required.
  --block-size [BLOCK_SIZE ...]
                        eg. --block-size 128 128
  --scale-fmt {ue8m0}   Round per-block scales up to a power of two (DeepGEMM style).
  --max-workers MAX_WORKERS
                        Number of worker threads for parallel processing
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm

from slim.utils.quant import block_fp8, make_keep_predicate, module_name_of, parse_quant_recipe


def load_quant_recipe(ref_config):
    """Read `ref_config` (a model dir or config.json) and parse its quant recipe."""
    path = ref_config
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("quantization_config") is None:
        raise ValueError(f"No `quantization_config` found in reference config: {path}")
    return parse_quant_recipe(cfg["quantization_config"])


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []

    def add_result(self, filename, q_weights, module_names):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += len(v)
            self.modules_to_not_convert.extend(module_names)


def process_file(input_path, output_path, filename, block_size, scale_fmt, is_kept, result_collector):
    if not filename.endswith(".safetensors"):
        return

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cuda") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    modules_to_not_convert = []
    for key in weights.keys():
        weight = weights[key]
        module_name = module_name_of(key)

        do_quant = key.endswith(".weight") and weight.dim() == 2 and not is_kept(module_name)

        if do_quant:
            qw, s = block_fp8(weight, block_size, scale_fmt=scale_fmt)
            q_weights[key] = qw
            q_weights[key.replace(".weight", ".weight_scale_inv")] = s
        else:
            if key.endswith(".weight"):
                modules_to_not_convert.append(module_name)
            q_weights[key] = weight

    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_fp8(
    input_path,
    output_path,
    block_size,
    max_workers=4,
    scale_fmt=None,
    keep_patterns=(),
    activation_scheme="dynamic",
    fmt="e4m3",
):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    is_kept = make_keep_predicate(keep_patterns)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            shutil.copyfile(os.path.join(input_path, filename), os.path.join(output_path, filename))

    safetensors_files = [f for f in os.listdir(input_path) if f.endswith(".safetensors")]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file, input_path, output_path, filename, block_size, scale_fmt, is_kept, result_collector
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    # Emit the concrete kept modules plus the requested keep-patterns, so a pattern
    # that matched nothing in the source (e.g. `lm_head` under tied embeddings) is
    # still honored downstream — the training model may have that module untied.
    kept_modules = sorted(set(result_collector.modules_to_not_convert) | set(keep_patterns))
    print(f"  Kept {len(result_collector.modules_to_not_convert)} source modules in source dtype; quantized the rest.")

    quantization_config = {
        "activation_scheme": activation_scheme,
        "fmt": fmt,
        "quant_method": "fp8",
        "weight_block_size": block_size,
    }
    if scale_fmt is not None:
        quantization_config["scale_fmt"] = scale_fmt
    if len(kept_modules) > 0:
        quantization_config["modules_to_not_convert"] = kept_modules

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {"weight_map": result_collector.weight_map, "metadata": {"total_size": result_collector.param_count}}
    json.dump(index_dict, open(os.path.join(output_path, "model.safetensors.index.json"), "w"), indent=2)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="Path to the directory of the HF safetensors model.")
    parser.add_argument("--save-dir", type=str, help="Path to the directory to save the converted model.")
    parser.add_argument("--ref-config", type=str, required=True,
                        help="Reference FP8 model dir or config.json whose quantization_config defines the recipe and keep-list.")
    parser.add_argument("--block-size", type=int, nargs="*", default=None, help="eg. --block-size 128 128")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--scale-fmt", type=str, default=None, choices=["ue8m0"])
    args = parser.parse_args()

    recipe = load_quant_recipe(args.ref_config)

    # Resolve recipe parameters: explicit CLI flags override the reference config.
    block_size = args.block_size if args.block_size is not None else recipe["block_size"]
    scale_fmt = args.scale_fmt or recipe.get("scale_fmt")
    keep_patterns = recipe["keep_patterns"]
    activation_scheme = recipe["activation_scheme"]
    fmt = recipe["fmt"]

    if not block_size:
        raise ValueError("Block-FP8 requires --block-size (or a reference config with weight_block_size).")
    if scale_fmt == "ue8m0" and list(block_size) != [128, 128]:
        raise ValueError("ue8m0 scales require 128x128 blocks (DeepGEMM constraint).")

    print(
        f"Recipe from {args.ref_config}: block_size={block_size} "
        f"scale_fmt={scale_fmt} keep={len(keep_patterns)} modules"
    )

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_fp8(
        args.model_dir,
        args.save_dir,
        block_size,
        max_workers=args.max_workers,
        scale_fmt=scale_fmt,
        keep_patterns=keep_patterns,
        activation_scheme=activation_scheme,
        fmt=fmt,
    )
