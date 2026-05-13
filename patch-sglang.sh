#!/usr/bin/env bash
# Apply patches to sglang after uv sync
set -euo pipefail

SITE=$(uv run python -c "import sglang; print(sglang.__path__[0])")
echo "Patching sglang at: $SITE"

# Patch: force legacy mm-load path for pre-expanded vision tokens
# This is CRITICAL for slim's token-in token-out design with expanded vision tokens
PY="$SITE/srt/multimodal/processors/qwen_vl.py"
if [ -f "$PY" ] && grep -q 'base_output = self\.load_mm_data(' "$PY"; then
    sed -i 's/base_output = self\.load_mm_data(/base_output = self.legacy_load_mm_data(/' "$PY"
    echo "  Applied: qwen_vl.py legacy mm-load path fix (for pre-expanded vision tokens)"
else
    echo "  Skipped: qwen_vl.py (already patched or not found)"
fi

# # Patch: disable GPU-side JPEG decoding globally.
# BP="$SITE/srt/multimodal/processors/base_processor.py"
# if [ -f "$BP" ] && grep -q '^    gpu_image_decode = True' "$BP"; then
#     sed -i 's/^    gpu_image_decode = True.*/    gpu_image_decode = False  # disabled to avoid duplicate CUDA context on GPU0/' "$BP"
#     echo "  Applied: base_processor.py gpu_image_decode=False (fixes GPU0 duplicate CUDA context)"
# else
#     echo "  Skipped: base_processor.py gpu_image_decode (already patched or not found)"
# fi

# Patch: guard s_aux against None in flash_attention_forward (transformers 5.6 regression)
# Vision encoder attention doesn't pass s_aux, so it arrives as None and crashes.
TRANS=$(uv run python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
FA="$TRANS/integrations/flash_attention.py"
if [ -f "$FA" ] && grep -q 's_aux=s_aux\.to(query\.dtype)' "$FA"; then
    sed -i 's/s_aux=s_aux\.to(query\.dtype),/s_aux=s_aux.to(query.dtype) if s_aux is not None else None,/' "$FA"
    echo "  Applied: flash_attention.py s_aux None guard (transformers 5.6 vision encoder fix)"
else
    echo "  Skipped: flash_attention.py (already patched or not found)"
fi

sudo sysctl -w kernel.yama.ptrace_scope=0

echo "Done."
