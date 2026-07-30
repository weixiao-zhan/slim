#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch installed runtime dependencies for slim rollout and training.

Run via: `uv run python patch_dependencies.py` (uses the active venv's interpreter).

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


def patch_automodel_optional_transformer_engine(parallelizer: Path) -> bool:
    """Keep AutoModel's native CP path usable without TE attention."""
    import_anchor = "    from transformer_engine.pytorch.attention import DotProductAttention\n"
    optional_import = """    try:
        from transformer_engine.pytorch.attention import DotProductAttention
    except ModuleNotFoundError as error:
        if not error.name or not error.name.startswith("transformer_engine"):
            raise
        DotProductAttention = ()
"""
    return patch_file(
        parallelizer,
        [(import_anchor, optional_import)],
        log_reason="AutoModel optional Transformer Engine attention",
    )


def patch_automodel_blockdiag_cp1(batch: Path, exchange: Path) -> None:
    """Use AutoModel's block-diagonal batch context at every CP degree."""
    cp1_noop = """    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend, sdpa_kernel

    world = cp_mesh.size()
    if world <= 1:
        primary = batch.get("inputs_embeds", batch.get("input_ids"))
        layout = None
        if primary is not None:
            layout = ShardLayout(original_seq_len=primary.shape[1], padded_seq_len=primary.shape[1])
        return nullcontext, batch, layout

    rank = cp_mesh.get_local_rank()
"""
    unified_context = """    from torch.nn.attention import SDPBackend, sdpa_kernel

    world = cp_mesh.size()
    rank = cp_mesh.get_local_rank()
"""
    group_world = "    _group_world = torch.distributed.get_world_size(group)\n"
    singleton_group_world = (
        "    _group_world = world if world == 1 else torch.distributed.get_world_size(group)\n"
    )
    patch_file(
        batch,
        [
            (cp1_noop, unified_context),
            (group_world, singleton_group_world),
        ],
        log_reason="AutoModel block-diagonal CP1 batch context",
    )

    gather_forward = """        ctx.world = world
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)
"""
    singleton_gather_forward = """        ctx.world = world
        x = x.contiguous()
        if world == 1:
            return x
        gathered = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)
"""
    gather_backward = """        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(local, chunks, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return local, None, None
"""
    singleton_gather_backward = """        if ctx.world == 1:
            return grad_out, None, None
        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(local, chunks, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return local, None, None
"""
    patch_file(
        exchange,
        [
            (gather_forward, singleton_gather_forward),
            (gather_backward, singleton_gather_backward),
        ],
        log_reason="AutoModel singleton block-diagonal K/V gather",
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
    import nemo_automodel
    import sglang

    automodel_dir = Path(nemo_automodel.__file__).resolve().parent
    parallelizer = automodel_dir / "components" / "moe" / "parallelizer.py"
    blockdiag_dir = automodel_dir / "components" / "distributed" / "blockdiag_cp"
    blockdiag_batch = blockdiag_dir / "batch.py"
    blockdiag_exchange = blockdiag_dir / "exchange.py"
    sglang_dir = Path(sglang.__file__).resolve().parent
    base_processor = sglang_dir / "srt" / "multimodal" / "processors" / "base_processor.py"

    patch_automodel_optional_transformer_engine(parallelizer)
    patch_automodel_blockdiag_cp1(blockdiag_batch, blockdiag_exchange)
    patch_sglang_base_processor(base_processor)
    install_triton_configs(sglang_dir)
    py_compile.compile(str(parallelizer), doraise=True)
    py_compile.compile(str(blockdiag_batch), doraise=True)
    py_compile.compile(str(blockdiag_exchange), doraise=True)
    py_compile.compile(str(base_processor), doraise=True)

    relax_ptrace_scope()
    return 0


if __name__ == "__main__":
    sys.exit(main())
