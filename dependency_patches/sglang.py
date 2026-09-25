# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang processor tensor transport and tuned kernel configurations."""

import py_compile
from pathlib import Path

from ._utils import patch_file


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
    repo_root = Path(__file__).resolve().parent.parent
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
    import_anchor = "from transformers import BaseImageProcessor\n"
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
    return_token_ids: bool = False
""",
                """    return_prompt_token_ids: bool = False
    return_processor_outputs: bool = False
    return_token_ids: bool = False
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
                """            return_prompt_token_ids=(
""",
                """            return_processor_outputs=request.return_processor_outputs,
            return_prompt_token_ids=(
""",
            ),
        ],
        log_reason="SGLang OpenAI processor-output request forwarding",
    )

    return protocol, io_struct, tokenizer_manager, serving_chat


def apply() -> None:
    import sglang

    sglang_dir = Path(sglang.__file__).resolve().parent
    base_processor = sglang_dir / "srt" / "multimodal" / "processors" / "base_processor.py"
    patch_sglang_base_processor(base_processor)
    openai_files = patch_sglang_return_processor_outputs(sglang_dir)
    install_triton_configs(sglang_dir)
    for path in (base_processor, *openai_files):
        py_compile.compile(str(path), doraise=True)
