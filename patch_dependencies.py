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
    import_anchor = "from sglang.srt.server_args import get_global_server_args\n"
    slim_import = "from slim.utils.processing_utils import decode_tensor_envelopes\n"
    data_anchor = "        all_loaded_data = base_output.organize_results()\n"
    decoded_data = (
        "        all_loaded_data = [\n"
        "            (modality, decode_tensor_envelopes(item))\n"
        "            for modality, item in base_output.organize_results()\n"
        "        ]\n"
    )
    return patch_file(
        base_processor,
        [
            (import_anchor, import_anchor + slim_import),
            (data_anchor, decoded_data),
        ],
        log_reason="base_processor.py processor-output tensor transport",
    )


def patch_sglang_return_processor_outputs(sglang_dir: Path) -> tuple[Path, ...]:
    """Add opt-in multimodal processor outputs to SGLang's per-request meta_info.

    ``meta_info`` is where SGLang already reports ``routed_experts``, and it reaches
    the OpenAI chat response per choice through ``return_meta_info``. Publishing the
    processor tensors there gives both endpoints one key path and leaves the response
    schema untouched.
    """
    protocol = sglang_dir / "srt" / "entrypoints" / "openai" / "protocol.py"
    io_struct = sglang_dir / "srt" / "managers" / "io_struct.py"
    tokenizer_manager = sglang_dir / "srt" / "managers" / "tokenizer_manager.py"
    serving_chat = sglang_dir / "srt" / "entrypoints" / "openai" / "serving_chat.py"

    patch_file(
        protocol,
        [
            (
                """    return_prompt_token_ids: bool = False
    return_meta_info: bool = False
""",
                """    return_prompt_token_ids: bool = False
    return_processor_outputs: bool = False
    return_meta_info: bool = False
""",
            ),
        ],
        log_reason="SGLang OpenAI processor-output request flag",
    )

    patch_file(
        io_struct,
        [
            (
                """    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False

    # Propagates trace context via Engine.generate/async_generate
""",
                """    # Whether to return prompt token IDs without computing logprobs
    return_prompt_token_ids: bool = False
    # Whether to return multimodal processor tensors
    return_processor_outputs: bool = False

    # Propagates trace context via Engine.generate/async_generate
""",
            ),
            (
                """            return_prompt_token_ids=self.return_prompt_token_ids,
            external_trace_header=self.external_trace_header,
""",
                """            return_prompt_token_ids=self.return_prompt_token_ids,
            return_processor_outputs=self.return_processor_outputs,
            external_trace_header=self.external_trace_header,
""",
            ),
        ],
        log_reason="SGLang internal processor-output request flag",
    )

    import_anchor = "from sglang.utils import TypeBasedDispatcher, get_exception_traceback\n"
    slim_import = "from slim.utils.processing_utils import encode_processor_outputs\n"
    patch_file(
        tokenizer_manager,
        [
            (import_anchor, import_anchor + slim_import),
            (
                """    # For return_prompt_token_ids: stores prompt token IDs captured after tokenization
    prompt_token_ids: Optional[List[int]] = None
""",
                """    # For return_prompt_token_ids: stores prompt token IDs captured after tokenization
    prompt_token_ids: Optional[List[int]] = None
    processor_outputs: Optional[Dict[str, Any]] = None
""",
            ),
            (
                """        self._validate_one_request(obj, input_ids)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
        )
""",
                """        self._validate_one_request(obj, input_ids)
        if isinstance(obj, GenerateReqInput) and obj.return_processor_outputs:
            state = self.rid_to_state[obj.rid]
            state.processor_outputs = encode_processor_outputs(mm_inputs)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
        )
""",
            ),
            (
                """            if getattr(recv_obj, "dp_ranks", None):
""",
                """            if (
                state.processor_outputs is not None
                and recv_obj.finished_reasons[i] is not None
            ):
                meta_info["processor_outputs"] = state.processor_outputs
            if getattr(recv_obj, "dp_ranks", None):
""",
            ),
        ],
        log_reason="SGLang processor-output capture and transport",
    )

    patch_file(
        serving_chat,
        [
            (
                """            if request.return_meta_info:
                raise ValueError(
                    "return_meta_info is not supported with streaming. "
                    "Please set stream=false when using return_meta_info=true."
                )
""",
                """            if request.return_meta_info:
                raise ValueError(
                    "return_meta_info is not supported with streaming. "
                    "Please set stream=false when using return_meta_info=true."
                )
            if request.return_processor_outputs:
                raise ValueError(
                    "return_processor_outputs is not supported with streaming. "
                    "Please set stream=false when using return_processor_outputs=true."
                )
""",
            ),
            (
                """            return_prompt_token_ids=request.return_prompt_token_ids,
        )
""",
                """            return_prompt_token_ids=request.return_prompt_token_ids,
            return_processor_outputs=request.return_processor_outputs,
        )
""",
            ),
        ],
        log_reason="SGLang OpenAI processor-output request forwarding",
    )

    return protocol, io_struct, tokenizer_manager, serving_chat


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
    sglang_openai_files = patch_sglang_return_processor_outputs(sglang_dir)
    install_triton_configs(sglang_dir)
    py_compile.compile(str(parallelizer), doraise=True)
    py_compile.compile(str(blockdiag_batch), doraise=True)
    py_compile.compile(str(blockdiag_exchange), doraise=True)
    py_compile.compile(str(base_processor), doraise=True)
    for path in sglang_openai_files:
        py_compile.compile(str(path), doraise=True)

    relax_ptrace_scope()
    return 0


if __name__ == "__main__":
    sys.exit(main())
