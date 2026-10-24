"""Shared registries + within-run-dir record I/O for the low-precision rollout study (see README.md).

Defines only the conventions INSIDE a run dir (the record schema below); the run-dir layout itself
is owned by run_matrix.sh, which hands each script an explicit dir to read/write.

Record schema (one JSON line per sample in records.jsonl), mirroring slim's finalized source-token-aligned
Episode (slim/utils/types.py) so 3_kl.py can rebuild an Episode directly; slot i predicts tokens[i+1]:
  tokens                : list[int]          full [prompt..., generated...], length N
  loss_mask             : list[int]          length N; 0 on prompt/terminal slots, 1 on generated predictions
  rollout_log_probs     : list[float]        length N; logprob of tokens[i+1], with a zero terminal slot
  rollout_routed_experts: int32 [N, L, K]    source-slot top-k expert ids (R3 only, in experts/<i>.npy)
  reward                : float              rule-based reward (accuracy) for this sample
  label                 : str                ground-truth answer
  num_prompt_tokens     : int                len(prompt_ids); slots < this-1 are prompt predictions
  has_mm                : bool               True if multimodal tensors saved in mm/<i>.npz

Multimodal inputs (VLM only): the processor-output tensors (pixel_values, image_grid_thw, ...) are
saved per sample in mm/<i>.npz and restored onto episode.multimodal_inputs in 3_kl.py, so the
training forward runs slim's VLM branch (image embeddings + MRoPE).
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Model registry: logical name -> BF16 source dir basename under the repo's models/ (symlinks).
MODELS = {
    "9b": "Qwen3.5-9B",
    "27b": "Qwen3.5-27B",
    "35b": "Qwen3.5-35B-A3B",
}
MOE_MODELS = {"35b"}

# Generated quantized checkpoints (0_prepare_models.sh writes them): <QUANT_DIR>/<base>-<suffix>.
QUANT_DIR = Path(os.environ.get("QUANT_DIR", "/mnt/sagemaker-nvme/experiments/low_precision/models"))


@dataclass(frozen=True)
class Precision:
    suffix: str | None  # checkpoint dir suffix; None = the BF16 source itself
    recipe: str | None  # tools/quant_recipes/ file that produces the checkpoint
    engine_kwargs: dict = field(default_factory=dict)  # sglang.Engine kwargs ...
    cli_flags: tuple = ()  # ... and the same settings as sglang CLI flags (bench_one_batch)
    moe: bool = True  # whether sglang can serve this format's MoE experts on SM120


PRECISIONS = {
    "bf16": Precision(None, None),
    # 128 Block FP8: 128x128 blocks, fp32 scales (the official Qwen3.5 FP8 recipe); sglang's SM120 CUTLASS
    # block-FP8 GEMM. MoE experts run sglang's default Triton fused-MoE (default kernel config):
    # sglang has no other block-FP8 MoE runner on SM120.
    "fp8": Precision("FP8", "qwen35_128blockfp8.json",
                     {"fp8_gemm_runner_backend": "cutlass"}, ("--fp8-gemm-backend", "cutlass")),
    # OCP MXFP8 (1x32, UE8M0); sglang auto-picks FlashInfer CUTLASS mm_mxfp8 on SM120.
    # sglang's MXFP8 MoE is SM100-only.
    "mxfp8": Precision("MXFP8", "qwen35_mxfp8.json", moe=False),
    # NVFP4 W4A4 (ModelOpt export, auto-detected as modelopt_fp4); FlashInfer CUTLASS GEMM + MoE.
    # Pin the MoE runner: sglang's SM120 override only fires for an explicit --quantization flag,
    # so an auto-detected checkpoint would otherwise pick the SM100-only flashinfer_trtllm.
    "nvfp4": Precision("NVFP4", "qwen35_nvfp4.json",
                       {"moe_runner_backend": "flashinfer_cutlass"}, ("--moe-runner-backend", "flashinfer_cutlass")),
}


def is_supported(model: str, precision: str) -> bool:
    return model not in MOE_MODELS or PRECISIONS[precision].moe


# Hybrid GDN state pool sizing, shared by every run: the lazy extra-buffer radix-cache strategy
# needs 4 mamba state slots per request (vs 5 for `extra_buffer`), so size the pool to exactly
# 4 x concurrency and leave the rest of the static pool to KV (tiny at these context lengths).
MAMBA_SLOTS_PER_REQUEST = 4
MAMBA_RADIX_CACHE_STRATEGY = "extra_buffer_lazy"


def mamba_engine_kwargs(concurrency: int) -> dict:
    return {
        "max_mamba_cache_size": MAMBA_SLOTS_PER_REQUEST * concurrency,
        "mamba_radix_cache_strategy": MAMBA_RADIX_CACHE_STRATEGY,
    }


def mamba_cli_flags(concurrency: int) -> tuple:
    return ("--max-mamba-cache-size", str(MAMBA_SLOTS_PER_REQUEST * concurrency),
            "--mamba-radix-cache-strategy", MAMBA_RADIX_CACHE_STRATEGY)

# Eval splits -> (parquet path, rm_type). 100-example sets from tests/prepare_mixed.py.
# The study uses 100 examples x 4 samples/prompt per split.
SPLITS = {
    "math": ("datasets/mixed/test_math.parquet", "math"),
    "vision": ("datasets/mixed/test_vision.parquet", "math"),  # geo3k graded with the math RM (boxed)
}


def model_dir(model: str, precision: str) -> str:
    suffix = PRECISIONS[precision].suffix
    if suffix is None:
        return f"models/{MODELS[model]}"
    return str(QUANT_DIR / f"{MODELS[model]}-{suffix}")


def write_meta(d: Path, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text())


def append_record(fh, rec: dict) -> None:
    fh.write(json.dumps(rec) + "\n")


def iter_records(d: Path):
    with open(d / "records.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_experts(d: Path, idx: int, arr: np.ndarray) -> None:
    ed = d / "experts"
    ed.mkdir(parents=True, exist_ok=True)
    np.save(ed / f"{idx}.npy", arr.astype(np.int32))


def load_experts(d: Path, idx: int):
    p = d / "experts" / f"{idx}.npy"
    return np.load(p) if p.exists() else None


def save_mm_inputs(d: Path, idx: int, mm: dict) -> None:
    """Persist a sample's processor-output tensors (VLM) to mm/<idx>.npz. `mm` maps key ->
    torch.Tensor; each tensor's torch dtype is recorded so load_mm_inputs restores it exactly."""
    md = d / "mm"
    md.mkdir(parents=True, exist_ok=True)
    arrays, dtypes = {}, {}
    for k, v in mm.items():
        arrays[k] = v.detach().cpu().to(torch.float32).numpy() if v.dtype.is_floating_point else v.detach().cpu().numpy()
        dtypes[k] = str(v.dtype).removeprefix("torch.")
    np.savez(md / f"{idx}.npz", __dtypes__=np.array(json.dumps(dtypes)), **arrays)


def load_mm_inputs(d: Path, idx: int):
    """Restore the saved processor tensors as a dict[str, torch.Tensor] with original dtypes, or None."""
    p = d / "mm" / f"{idx}.npz"
    if not p.exists():
        return None
    npz = np.load(p, allow_pickle=False)
    dtypes = json.loads(str(npz["__dtypes__"]))
    out = {}
    for k in npz.files:
        if k == "__dtypes__":
            continue
        out[k] = torch.from_numpy(npz[k]).to(getattr(torch, dtypes[k]))
    return out or None


if __name__ == "__main__":
    # Shell access to the registries, so the .sh drivers share one source of truth:
    #   python common.py models | precisions | moe M | supported M P | dir M P | recipe P | flags P
    #                    | mamba-flags CONCURRENCY
    cmd, *rest = sys.argv[1:]
    if cmd == "models":
        print(" ".join(MODELS))
    elif cmd == "precisions":
        print(" ".join(PRECISIONS))
    elif cmd == "moe":
        sys.exit(0 if rest[0] in MOE_MODELS else 1)
    elif cmd == "supported":
        sys.exit(0 if is_supported(*rest) else 1)
    elif cmd == "dir":
        print(model_dir(*rest))
    elif cmd == "recipe":
        print(PRECISIONS[rest[0]].recipe or "")
    elif cmd == "flags":
        print(" ".join(PRECISIONS[rest[0]].cli_flags))
    elif cmd == "mamba-flags":
        print(" ".join(mamba_cli_flags(int(rest[0]))))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
