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


def patch_transformers_flash_attention() -> bool:
    """Guard s_aux against None in flash_attention_forward.

    transformers 5.6 vision encoder attention doesn't pass s_aux, so it arrives
    as None and crashes the .to(query.dtype) call.

    Fixed upstream in transformers#45589, released in v5.6.2; can deprecate this
    patch once we upgrade to transformers>=5.6.2 (blocked by sglang's
    transformers==5.6.0 pin).
    """
    import transformers

    return patch_file(
        Path(transformers.__file__).resolve().parent / "integrations" / "flash_attention.py",
        [
            (
                "s_aux=s_aux.to(query.dtype),",
                "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,",
            ),
        ],
        log_reason="flash_attention.py s_aux None guard",
    )


def patch_sglang_qwen_vl(qwen_vl: Path) -> bool:
    """Force legacy mm-load path + adapt to processor_output payloads.

    The legacy path is required when input_ids already contain expanded vision
    tokens (token-in/token-out rollout sending image_data as PNG data URLs).
    The processor_output branch below remains active for envelope-encoded
    payloads, so both rollout styles are supported.
    """
    return patch_file(
        qwen_vl,
        [
            (
                "base_output = self.load_mm_data(",
                "base_output = self.legacy_load_mm_data(",
            ),
            (
         '''image_grid_thw = None
        if hasattr(ret, "image_grid_thw"):
            image_grid_thw = ret.image_grid_thw

        if image_grid_thw is None and image_data and isinstance(image_data[0], dict):
            image_grid_thw = image_data[0].get("image_grid_thw")

        video_grid_thw = None
        if hasattr(ret, "video_grid_thw"):
            video_grid_thw = ret.video_grid_thw

        if video_grid_thw is None and request_obj.video_data:
            first_video = request_obj.video_data[0]
            if isinstance(first_video, dict):
                video_grid_thw = first_video.get("video_grid_thw")''',
         '''processor_output_dict = None
        for mm_data in (image_data, request_obj.video_data, request_obj.audio_data):
            if mm_data and isinstance(mm_data[0], dict) and mm_data[0].get("format") == "processor_output":
                processor_output_dict = mm_data[0]
                break

        image_grid_thw = None
        if hasattr(ret, "image_grid_thw"):
            image_grid_thw = ret.image_grid_thw
        if image_grid_thw is None and processor_output_dict is not None:
            image_grid_thw = processor_output_dict.get("image_grid_thw")

        video_grid_thw = None
        if hasattr(ret, "video_grid_thw"):
            video_grid_thw = ret.video_grid_thw
        if video_grid_thw is None and processor_output_dict is not None:
            video_grid_thw = processor_output_dict.get("video_grid_thw")''',
            ),
            (
         '''input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=second_per_grid_ts,''',
         '''input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,''',
            ),
        ],
        log_reason="qwen_vl.py legacy mm-load + processor_output adapter",
    )


def patch_sglang_base_processor(base_processor: Path) -> bool:
    """Accept processor_output envelope + suppress spurious mismatch warnings.

    The mismatch warnings fire under processor_output because a single dict
    carries data for all expanded vision-token blocks; the iterator
    legitimately exhausts before all tokens resolve.
    """
    return patch_file(
        base_processor,
        [
            (
                "import concurrent\n",
                "import concurrent\nimport pybase64\n",
            ),
            (
         '''if input_format == "processor_output":
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)''',
         '''if input_format == "processor_output":
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
            (
         '''except StopIteration:
                    logger.warning(
                        f"Mismatch: More \'{modality.name}\' tokens found than corresponding data provided."
                    )
                    return futures, task_info''',
         '''except StopIteration:
                    # Suppressed: with processor_output format a single dict
                    # carries data for all expanded vision-token blocks, so the
                    # iterator legitimately exhausts before all tokens resolve.
                    return futures, task_info''',
            ),
            (
         '''try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass''',
         '''try:
                next(iterator)
            except StopIteration:
                pass
            except Exception:
                pass''',
            ),
        ],
        log_reason="base_processor.py pybase64 + processor_output unpack + warning suppression",
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
    qwen_vl = sglang_dir / "srt" / "multimodal" / "processors" / "qwen_vl.py"

    patch_transformers_flash_attention()
    patch_sglang_qwen_vl(qwen_vl)
    patch_sglang_base_processor(base_processor)

    py_compile.compile(str(base_processor), doraise=True)
    py_compile.compile(str(qwen_vl), doraise=True)

    relax_ptrace_scope()
    return 0


if __name__ == "__main__":
    sys.exit(main())
