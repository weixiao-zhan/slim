#!/usr/bin/env python3
"""Patch installed transformers + sglang for slim's rollout/training needs.

Run via: `uv run python patch_sglang.py` (uses the active venv's interpreter).

Idempotent: each replacement is skipped if its `new` text is already present.
"""
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path


def patch_file(
    path: Path,
    replacements: list[tuple[str, str]],
    log_reason: str = "",
) -> bool:
    """Apply (old, new) replacements to `path` and print a one-line status.

    For each pair: if `new` is already present, skip; else `old` must be
    found exactly once and is replaced. Missing `old` raises — that means
    the upstream file shape changed and we want to fail loudly.
    """
    text = path.read_text()
    original = text
    label = log_reason or path.name
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            print(f"  No match: {label} — anchor not found: {old[:80]!r}")
            raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text)
        print(f"  Applied: {label}")
    else:
        print(f"  Skipped (already applied): {label}")
    return True


def patch_sglang_base_processor(base_processor: Path) -> bool:
    """Decode our base64-enveloped tensors back into real tensors.

    slim's VLM rollout ships token-in/token-out processor outputs to the engine
    via `image_data = [{"format": "processor_output", ...}]`, with each tensor
    base64-enveloped (`encode_tensor_to_b64_envelope`) to survive the Rust
    router's JSON layer. Upstream SGLang has no base64 transport concept, so the
    decode is still our job: it must happen *before*
    `collect_mm_items_from_processor_output`, which assumes real tensors.

    As of SGLang 0.5.13 the multimodal path was refactored: token-in/token-out
    is native (`SGLANG_MM_AVOID_RETOKENIZE` + fast-path early return for
    preprocessed data), `image_grid_thw`/MRoPE are extracted natively, and the
    spurious mismatch warnings no longer fire (the preprocessed payload skips
    `submit_data_loading_tasks`). So the old qwen_vl legacy/grid_thw patches and
    the warning-suppression patches are gone; only this transport decode remains.

    The dispatch branch now keys on the `MultimodalInputFormat.PROCESSOR_OUTPUT`
    enum (was the `"processor_output"` string in 0.5.12).
    """
    return patch_file(
        base_processor,
        [
            (
                "import concurrent\n",
                "import concurrent\nimport pybase64\n",
            ),
            (
         '''if input_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)''',
         '''if input_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                for key, value in list(dict_item.items()):
                    if key == "format":
                        continue
                    if isinstance(value, dict) and value.get("__tensor__"):
                        dtype = getattr(torch, value["dtype"])
                        storage = pybase64.b64decode(value["data"].encode("ascii"))
                        dict_item[key] = torch.frombuffer(
                            bytearray(storage), dtype=dtype
                        ).reshape(value["shape"])
                    elif isinstance(value, list):
                        try:
                            dict_item[key] = torch.as_tensor(value)
                        except (TypeError, ValueError):
                            pass
                if input_ids is None and "input_ids" in dict_item:
                    input_ids = torch.as_tensor(dict_item["input_ids"]).flatten()
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)''',
            ),
        ],
        log_reason="base_processor.py pybase64 + processor_output b64 decode",
    )


def _install_configs(src_root: Path, dst_root: Path, label: str) -> int:
    """Mirror every *.json under `src_root` into `dst_root`, preserving subdirs.

    Idempotent: a destination file is written only if missing or differing. Returns
    the number of files copied.
    """
    import shutil

    if not src_root.is_dir():
        return 0
    copied = 0
    for src in src_root.rglob("*.json"):
        dst = dst_root / src.relative_to(src_root)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied += 1
    if copied:
        print(f"  Applied: installed {copied} {label} config(s) into sglang")
    else:
        print(f"  Skipped: {label} configs already installed")
    return copied


def install_triton_configs(sglang_dir: Path) -> None:
    """Copy in-repo tuned Triton configs into sglang's kernel config dirs.

    Source of truth is two in-repo dirs (version controlled, survive sglang upgrades):
      - tools/triton_moe_configs/triton_<ver>/E=*.json -> moe_runner triton_utils configs
        (fused-MoE kernel, keyed by triton version dir)
      - tools/triton_fp8_configs/N=*.json              -> quantization configs (flat)
        (block-FP8 W8A8 GEMM; loader has no env override, so the file must live there)
    """
    repo_root = Path(__file__).resolve().parent
    _install_configs(
        repo_root / "tools" / "triton_moe_configs",
        sglang_dir / "srt" / "layers" / "moe" / "moe_runner" / "triton_utils" / "configs",
        "Triton MoE",
    )
    _install_configs(
        repo_root / "tools" / "triton_fp8_configs",
        sglang_dir / "srt" / "layers" / "quantization" / "configs",
        "Triton FP8 GEMM",
    )


def relax_ptrace_scope() -> None:
    """Set kernel.yama.ptrace_scope=0 (needed for Torch CUDA IPC weight sync).

    Skipped with a warning if sudo is unavailable or requires a password.
    """
    if os.environ.get("SKIP_PTRACE_SYSCTL"):
        print("  Skipped: kernel.yama.ptrace_scope (SKIP_PTRACE_SYSCTL set)")
        return
    try:
        subprocess.run(
            ["sudo", "-n", "sysctl", "-w", "kernel.yama.ptrace_scope=0"],
            check=True,
            capture_output=True,
        )
        print("  Applied: kernel.yama.ptrace_scope=0")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"  Warning: could not set kernel.yama.ptrace_scope=0 ({e}). "
            "Torch CUDA IPC weight sync (pidfd_getfd) may fail. "
            "Set it manually: sudo sysctl -w kernel.yama.ptrace_scope=0"
        )


def main() -> int:
    import sglang

    sglang_dir = Path(sglang.__file__).resolve().parent
    base_processor = sglang_dir / "srt" / "multimodal" / "processors" / "base_processor.py"

    patch_sglang_base_processor(base_processor)
    install_triton_configs(sglang_dir)

    py_compile.compile(str(base_processor), doraise=True)

    relax_ptrace_scope()
    return 0


if __name__ == "__main__":
    sys.exit(main())
