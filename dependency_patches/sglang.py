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
        # Unlink first so a file hardlinked to the uv cache is not overwritten in place.
        dst.unlink(missing_ok=True)
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
      - tools/triton_fp8_configs/N=*.json              -> gemm configs (flat)
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
        sglang_dir / "kernels" / "ops" / "gemm" / "configs",
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


def patch_sglang_processor_output_contract(api_types: Path) -> bool:
    """Admit processor-output media items in the ``/generate`` JSON schema.

    The schema decodes every dict media item as a ``MediaRef``, which rejects the
    processor-output dict that carries locally computed processor tensors.
    """
    media_ref_arm = """    if isinstance(value, dict):
        return decode_MediaRef(value)
    raise _invalid_type(value, expected="one of the MediaItem shapes")
"""
    processor_output_arm = """    if isinstance(value, dict):
        if value.get("format") == "processor_output":
            return value
        return decode_MediaRef(value)
    raise _invalid_type(value, expected="one of the MediaItem shapes")
"""
    return patch_file(
        api_types,
        [(media_ref_arm, processor_output_arm)],
        log_reason="SGLang /generate schema admits processor-output media items",
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


def patch_sglang_meta_info_tensor_envelopes(sglang_dir: Path) -> tuple[Path, ...]:
    """Return the per-token output arrays of ``meta_info`` as base64 arrays.

    Upstream builds one Python tuple per token, and per top-K entry, then serializes
    them as JSON numbers. On a long generation that blocks the HTTP process long enough
    to fail health checks. ``/generate`` and chat completions turn on SGLang's flat
    base64 output top-K for every non-streaming request, and with it the output logprobs
    come as tensor envelopes, encoded once on the finished output. Chat completions
    leaves the OpenAI choice logprobs empty: slim reads ``meta_info``. Streaming keeps
    the upstream lists, which its per-chunk choices read.
    """
    tokenizer_manager = sglang_dir / "srt" / "managers" / "tokenizer_manager.py"
    serving_chat = sglang_dir / "srt" / "entrypoints" / "openai" / "serving_chat.py"
    http_server = sglang_dir / "srt" / "entrypoints" / "http_server.py"

    slim_import = "from slim.utils.processing_utils import encode_processor_outputs\n"
    envelope_import = "from slim.utils.processing_utils import encode_array_to_b64_envelope\n"
    patch_file(
        tokenizer_manager,
        [
            (slim_import, slim_import + envelope_import),
            (
                """        if len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
            state.output_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                    state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_logprobs"] = state.input_token_logprobs
        meta_info["output_token_logprobs"] = state.output_token_logprobs
        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs)
""",
                """        # Flat output top logprobs bring the output logprobs as envelopes, encoded once
        # on the finished output.
        flat_output = state.obj.return_flat_raw_output_top_logprobs
        if not flat_output and len(state.output_token_logprobs_val) > len(state.output_token_logprobs):
            state.output_token_logprobs.extend(
                self.detokenize_logprob_tokens(
                    state.output_token_logprobs_val[len(state.output_token_logprobs) :],
                    state.output_token_logprobs_idx[len(state.output_token_logprobs) :],
                    return_text_in_logprobs,
                )
            )

        meta_info["input_token_logprobs"] = state.input_token_logprobs
        if not flat_output:
            meta_info["output_token_logprobs"] = state.output_token_logprobs
        elif finished:
            meta_info["output_token_logprobs"] = {
                "logprobs": encode_array_to_b64_envelope(state.output_token_logprobs_val, np.float32),
                "token_ids": encode_array_to_b64_envelope(state.output_token_logprobs_idx, np.int32),
            }
        meta_info["output_token_logprobs_length"] = len(state.output_token_logprobs_val)
""",
            ),
        ],
        log_reason="SGLang meta_info tensor envelopes",
    )

    patch_file(
        serving_chat,
        [
            (
                """            return_flat_raw_output_top_logprobs=request.return_flat_raw_output_top_logprobs,
            # GenerateReqInput rejects b64 without a flat flag; chat has only the output one
            return_flat_raw_top_logprobs_b64=(
                request.return_flat_raw_output_top_logprobs
                and request.return_flat_raw_top_logprobs_b64
            ),
""",
                """            # A non-streaming choice reads its output logprobs and top-K from meta_info.
            return_flat_raw_output_top_logprobs=not request.stream,
            return_flat_raw_top_logprobs_b64=not request.stream,
""",
            ),
            (
                """            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)
""",
                """            # The logprobs travel in meta_info as base64 arrays.
            choice_logprobs = None
""",
            ),
        ],
        log_reason="SGLang OpenAI chat tensor envelopes",
    )

    patch_file(
        http_server,
        [
            (
                '''async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
''',
                '''async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    # A non-streaming response carries its output logprobs and top-K as base64 arrays.
    obj.return_flat_raw_output_top_logprobs = obj.return_flat_raw_top_logprobs_b64 = not obj.stream
''',
            ),
        ],
        log_reason="SGLang /generate tensor envelopes",
    )

    return tokenizer_manager, serving_chat, http_server


def patch_sglang_qwen_vl_image_token(qwen_vl: Path) -> bool:
    """Count only the native vision tokens as image placeholders.

    Upstream also matches a literal ``<image>``, so any conversation text containing it
    (a user prompt, or the model's own earlier turn) adds an image placeholder without
    image data and the request fails. slim sends native vision tokens on every endpoint.
    """
    return patch_file(
        qwen_vl,
        [
            (
                """        # Also match the legacy sglang <image> sentinel used by /generate,
        # so the artifact fast path can normalize it before build_input_ids.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\\|vision_start\\|>(?:<\\|image_pad\\|>)+<\\|vision_end\\|>|<image>"
            ),
""",
                """        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\\|vision_start\\|>(?:<\\|image_pad\\|>)+<\\|vision_end\\|>"
            ),
""",
            ),
        ],
        log_reason="qwen_vl.py native-only image placeholder",
    )


def patch_sglang_flush_before_kv_release(weight_updater: Path) -> bool:
    """Flush the request and cache pools before the memory saver unmaps the KV cache.

    Clearing the hybrid pools (Mamba mapping, PLE short-conv and n-gram pools) writes device memory in the
    KV cache region, so it must run while that region is still mapped.
    """
    return patch_file(
        weight_updater,
        [
            (
                """            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()
""",
                """            self.flush_cache()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
""",
            )
        ],
        log_reason="SGLang flush before KV cache release",
    )


def apply() -> None:
    import sglang

    sglang_dir = Path(sglang.__file__).resolve().parent
    base_processor = sglang_dir / "srt" / "multimodal" / "processors" / "base_processor.py"
    patch_sglang_base_processor(base_processor)
    qwen_vl = sglang_dir / "srt" / "multimodal" / "processors" / "qwen_vl.py"
    patch_sglang_qwen_vl_image_token(qwen_vl)
    api_types = sglang_dir / "api" / "v1" / "api_types.py"
    patch_sglang_processor_output_contract(api_types)
    openai_files = patch_sglang_return_processor_outputs(sglang_dir)
    envelope_files = patch_sglang_meta_info_tensor_envelopes(sglang_dir)
    weight_updater = sglang_dir / "srt" / "managers" / "scheduler_components" / "weight_updater.py"
    patch_sglang_flush_before_kv_release(weight_updater)
    install_triton_configs(sglang_dir)
    for path in (base_processor, qwen_vl, api_types, weight_updater, *openai_files, *envelope_files):
        py_compile.compile(str(path), doraise=True)
