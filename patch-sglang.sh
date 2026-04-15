#!/usr/bin/env bash
# Apply patches to sglang v0.5.10.post1 after uv sync
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

sudo sysctl -w kernel.yama.ptrace_scope=0

echo "Done."
