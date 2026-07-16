#!/usr/bin/env python3
"""Patch installed SGLang for slim rollout and training.

Run via: `uv run python patch_sglang.py` (uses the active venv's interpreter).

Idempotent: each replacement is skipped if its new text is already present.
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
    """Apply exact text replacements and fail if an upstream anchor changed."""
    text = path.read_text()
    original = text
    label = log_reason or path.name
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            print(f"  No match: {label}; anchor not found: {old[:80]!r}")
            raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text)
        print(f"  Applied: {label}")
    else:
        print(f"  Skipped (already applied): {label}")
    return True


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
        print(f"  Skipped (already installed): {label} configs")
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


def patch_sglang_base_processor(base_processor: Path) -> bool:
    """Decode JSON tensor envelopes before SGLang consumes processor output."""
    constants_anchor = (
        "_IPC_POOL_HANDLE_CACHE = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()\n"
    )
    transport_helper = '''

def _decode_slim_tensor_transport(value):
    """Restore tensors encoded for JSON transport by slim."""
    if isinstance(value, dict) and value.get("__tensor__") is True:
        import base64

        dtype_name = value.get("dtype")
        dtype = getattr(torch, str(dtype_name), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported tensor transport dtype: {dtype_name!r}")

        shape = value.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(size, int) and size >= 0 for size in shape
        ):
            raise ValueError(f"Invalid tensor transport shape: {shape!r}")

        raw = base64.b64decode(value.get("data", ""), validate=True)
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        expected_elements = 1
        for size in shape:
            expected_elements *= size
        if tensor.numel() != expected_elements:
            raise ValueError(
                "Tensor transport size mismatch: "
                f"decoded={tensor.numel()}, expected={expected_elements}"
            )
        return tensor.reshape(shape)
    if isinstance(value, dict):
        return {
            key: _decode_slim_tensor_transport(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_slim_tensor_transport(item) for item in value]
    return value
'''
    data_anchor = "        all_loaded_data = base_output.organize_results()\n"
    decoded_data = (
        "        all_loaded_data = [\n"
        "            (modality, _decode_slim_tensor_transport(item))\n"
        "            for modality, item in base_output.organize_results()\n"
        "        ]\n"
    )
    return patch_file(
        base_processor,
        [
            (constants_anchor, constants_anchor + transport_helper),
            (data_anchor, decoded_data),
        ],
        log_reason="base_processor.py processor-output tensor transport",
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
