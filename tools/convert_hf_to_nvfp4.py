# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate a BF16 HF model with NVIDIA ModelOpt and export an NVFP4 checkpoint.

The recipe (`--ref-config`, e.g. tools/quant_recipes/qwen35_nvfp4.json) must set
`quantization_config.quant_algo == "NVFP4"`. Its keep-list `modules_to_not_convert` uses
the same dotted-path segment patterns as the FP8 recipes and replaces ModelOpt's built-in
name-based exclusions, so every recipe in tools/quant_recipes/ keeps the same layers in BF16.

Calibration is ModelOpt `max` over `--num-prompts` prompts sampled from `--calib-data`
(slim dataset format: a chat `prompt` column and an optional `images` column). Each prompt goes
through the same processor path as slim's rollout (chat template + image processor) and runs as
its own forward, so text and image-token activations both reach the quantizers. It only sets the
per-tensor activation amax, which becomes the static `input_scale` of sglang's modelopt_fp4 path;
weight scales are data-free.

The export (`quant_method: modelopt`, `quant_algo: NVFP4`) is auto-detected by sglang as
`modelopt_fp4`. Needs the optional `quant` extra: `uv sync --extra quant`.

python tools/convert_hf_to_nvfp4.py --model-dir MODEL_DIR --save-dir SAVE_DIR --ref-config REF_CONFIG
                                [--calib-data PARQUET] [--num-prompts N]
"""

import argparse
import copy
import json
import os
import shutil

import torch
from tqdm import tqdm

# Entries of ModelOpt's NVFP4 config that define the format or exclude by module type; its
# name-based exclusions are dropped in favor of the recipe's keep-list.
_FORMAT_QUANTIZER_NAMES = ("*", "*weight_quantizer", "*input_quantizer")


def load_nvfp4_recipe(ref_config):
    path = os.path.join(ref_config, "config.json") if os.path.isdir(ref_config) else ref_config
    with open(path) as f:
        qc = json.load(f).get("quantization_config") or {}
    if qc.get("quant_method") != "modelopt" or qc.get("quant_algo") != "NVFP4":
        raise ValueError(f"{path}: expected quant_method=modelopt, quant_algo=NVFP4, got {qc}")
    return qc


def build_quant_cfg(group_size, keep_patterns):
    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    entries = [e for e in cfg["quant_cfg"] if e["quantizer_name"] in _FORMAT_QUANTIZER_NAMES]
    for entry in entries:
        block_sizes = entry.get("cfg", {}).get("block_sizes")
        if block_sizes is not None and block_sizes[-1] != group_size:
            raise ValueError(f"Recipe group_size={group_size}, but ModelOpt NVFP4 uses {block_sizes[-1]}")
    # Segment-boundary match, as slim.utils.quant.module_path_match: anywhere in the path, or top level.
    for pattern in keep_patterns:
        entries.append({"quantizer_name": f"*.{pattern}.*", "enable": False})
        entries.append({"quantizer_name": f"{pattern}.*", "enable": False})
    cfg["quant_cfg"] = entries
    return cfg


def build_calib_inputs(processor, calib_data, num_prompts, seed=0):
    """Processor outputs (input_ids + vision tensors) for `num_prompts` shuffled dataset prompts."""
    from datasets import Dataset, Image, Sequence

    ds = Dataset.from_parquet(calib_data)
    if "images" in ds.column_names:
        ds = ds.cast_column("images", Sequence(Image()))
    ds = ds.shuffle(seed=seed).select(range(min(num_prompts, len(ds))))
    inputs = []
    for example in ds:
        text = processor.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        images = example.get("images") or None
        out = processor(text=text, images=images, return_tensors="pt")
        out.pop("attention_mask", None)
        inputs.append(dict(out))
    num_vision = sum("pixel_values" in x for x in inputs)
    num_tokens = sum(x["input_ids"].numel() for x in inputs)
    print(f"Calibration: {len(inputs)} prompts ({num_vision} with images), {num_tokens} tokens")
    return inputs


def copy_missing_assets(model_dir, save_dir):
    """Copy tokenizer / processor / template files the export did not write."""
    for filename in os.listdir(model_dir):
        source = os.path.join(model_dir, filename)
        target = os.path.join(save_dir, filename)
        if os.path.isdir(source) or os.path.exists(target) or filename.endswith((".safetensors", ".bin")):
            continue
        if filename.endswith(".index.json"):
            continue
        shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="BF16 HF model dir.")
    parser.add_argument("--save-dir", required=True, help="Output dir for the NVFP4 checkpoint.")
    parser.add_argument("--ref-config", required=True, help="NVFP4 recipe JSON (or a dir with config.json).")
    parser.add_argument("--calib-data", default="datasets/mixed/train.parquet",
                        help="Parquet with a chat `prompt` column (slim dataset format).")
    parser.add_argument("--num-prompts", type=int, default=2048, help="Calibration prompts.")
    args = parser.parse_args()

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForImageTextToText, AutoProcessor

    qc = load_nvfp4_recipe(args.ref_config)
    keep_patterns = list(qc.get("modules_to_not_convert") or [])
    quant_cfg = build_quant_cfg(qc.get("group_size", 16), keep_patterns)
    print(f"Recipe from {args.ref_config}: NVFP4 group_size={qc.get('group_size', 16)} keep={keep_patterns}")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    calib_inputs = build_calib_inputs(processor, args.calib_data, args.num_prompts)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    device = model.get_input_embeddings().weight.device

    def forward_loop(m):
        with torch.no_grad():
            for inputs in tqdm(calib_inputs, desc="Calibrating"):
                m(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False)

    model = mtq.quantize(model, quant_cfg, forward_loop)
    mtq.print_quant_summary(model)

    export_hf_checkpoint(model, export_dir=args.save_dir)
    copy_missing_assets(args.model_dir, args.save_dir)
    print(f"Exported NVFP4 checkpoint to {args.save_dir}")


if __name__ == "__main__":
    main()
