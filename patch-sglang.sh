#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv-test/bin/python" ]]; then
    PYTHON_BIN=".venv-test/bin/python"
  elif [[ -x "../.venv/bin/python" ]]; then
    PYTHON_BIN="../.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# Patch: guard s_aux against None in flash_attention_forward (transformers 5.6 regression)
# Vision encoder attention doesn't pass s_aux, so it arrives as None and crashes.
TRANS=$("${PYTHON_BIN}" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
FA="$TRANS/integrations/flash_attention.py"
if [ -f "$FA" ] && grep -q 's_aux=s_aux\.to(query\.dtype)' "$FA"; then
    sed -i 's/s_aux=s_aux\.to(query\.dtype),/s_aux=s_aux.to(query.dtype) if s_aux is not None else None,/' "$FA"
    echo "  Applied: flash_attention.py s_aux None guard (transformers 5.6 vision encoder fix)"
else
    echo "  Skipped: flash_attention.py (already patched or not found)"
fi

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import py_compile
import sglang


pkg = Path(sglang.__file__).resolve().parent
base_processor = pkg / "srt" / "multimodal" / "processors" / "base_processor.py"
qwen_vl = pkg / "srt" / "multimodal" / "processors" / "qwen_vl.py"


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text()
    original = text
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text)
        return True
    return False


base_changed = patch_file(
    base_processor,
    [
        (
            "import concurrent\n",
            "import base64\nimport concurrent\n",
        ),
        (
            '''            if input_format == "processor_output":
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)
''',
            '''            if input_format == "processor_output":
                for key, value in list(dict_item.items()):
                    if key == "format":
                        continue
                    if isinstance(value, dict) and value.get("__tensor__"):
                        dtype = getattr(torch, value["dtype"])
                        storage = base64.b64decode(value["data"].encode("ascii"))
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
                all_collected_items.extend(items)
''',
        ),
    ],
)

qwen_changed = patch_file(
    qwen_vl,
    [
        (
            '''        image_grid_thw = None
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
                video_grid_thw = first_video.get("video_grid_thw")
''',
            '''        processor_output_dict = None
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
            video_grid_thw = processor_output_dict.get("video_grid_thw")
''',
        ),
        (
            '''            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=second_per_grid_ts,
''',
            '''            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
''',
        ),
    ],
)

py_compile.compile(str(base_processor), doraise=True)
py_compile.compile(str(qwen_vl), doraise=True)

print(f"patched={base_changed or qwen_changed}")
print(f"base_processor={base_processor}")
print(f"qwen_vl={qwen_vl}")
PY
