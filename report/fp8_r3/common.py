"""Shared registries + within-run-dir record I/O for the FP8/R3 study (see fp8_r3_throughput_kl.md).

Defines only the conventions INSIDE a run dir (the record schema below); the run-dir layout itself
is owned by run_matrix.sh, which hands each script an explicit dir to read/write.

Record schema (one JSON line per sample in records.jsonl), mirroring slim's edge-aligned Episode
(slim/utils/types.py) so stage-2 can rebuild an Episode directly; edge i predicts tokens[i+1]:
  tokens                : list[int]          full [prompt..., generated...], length N
  loss_mask             : list[int]          length N-1; 0 on prompt edges, 1 on generated edges
  rollout_log_probs     : list[float]        length N-1; logprob of tokens[i+1] under rollout policy
  rollout_routed_experts: int32 [N-1, L, K]  per-edge top-k expert ids (R3 only, in experts/<i>.npy)
  reward                : float              rule-based reward (accuracy) for this sample
  label                 : str                ground-truth answer
  num_prompt_tokens     : int                len(prompt_ids); edges < this-1 are prompt (mask 0)
  has_mm                : bool               True if multimodal tensors saved in mm/<i>.npz

Multimodal inputs (VLM only): the processor-output tensors (pixel_values, image_grid_thw, ...) are
saved per sample in mm/<i>.npz and restored onto episode.multimodal_inputs in stage-2, so the
training forward runs slim's VLM branch (image embeddings + MRoPE).
"""

import json
from pathlib import Path

import numpy as np
import torch

# Model registry: logical name -> (bf16 dir, fp8 dir). Paths are the repo symlinks.
MODELS = {
    "2b": ("models/Qwen3.5-2B", "models/Qwen3.5-2B-FP8"),
    "9b": ("models/Qwen3.5-9B", "models/Qwen3.5-9B-FP8"),
    "35b": ("models/Qwen3.5-35B-A3B", "models/Qwen3.5-35B-A3B-FP8"),
}

# Eval splits -> (parquet path, rm_type). 100-example sets (sliced from the larger pool built by
# prepare_eval.py); the study uses 100 examples x 4 samples/prompt per split.
SPLITS = {
    "math": ("datasets/eval100/test_math.parquet", "math"),
    "vision": ("datasets/eval100/test_vision.parquet", "math"),  # geo3k graded with the math RM (boxed)
}


def model_dir(model: str, precision: str) -> str:
    bf16, fp8 = MODELS[model]
    return fp8 if precision == "fp8" else bf16


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
