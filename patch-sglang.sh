#!/usr/bin/env bash
# Apply patches to sglang v0.5.9 after uv sync
set -euo pipefail

SITE=$(python -c "import sglang; print(sglang.__path__[0])")
echo "Patching sglang at: $SITE"

# Patch 1: fix Qwen3-VL visual module weight key (sglang#19333)
PY="$SITE/srt/models/qwen3_vl.py"
if [ -f "$PY" ] && ! grep -q 'name = name.replace(r"model.visual.", r"visual.")' "$PY"; then
    sed -i '/name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")/a\                    name = name.replace(r"model.visual.", r"visual.")' "$PY"
    echo "  Applied: qwen3_vl.py visual weight key fix"
else
    echo "  Skipped: qwen3_vl.py (already patched or not found)"
fi

# Patch 2: force legacy mm-load path for pre-expanded vision tokens
PY="$SITE/srt/multimodal/processors/qwen_vl.py"
if [ -f "$PY" ] && grep -q 'base_output = self\.load_mm_data(' "$PY"; then
    sed -i 's/base_output = self\.load_mm_data(/base_output = self.legacy_load_mm_data(/' "$PY"
    echo "  Applied: qwen_vl.py legacy mm-load path fix"
else
    echo "  Skipped: qwen_vl.py (already patched or not found)"
fi

sudo sysctl -w kernel.yama.ptrace_scope=0

echo "Done."
