#!/usr/bin/env bash
# Shared helpers for test scripts.
# Source this file: source "$(dirname "$0")/common.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present
if [[ -f "$REPO_DIR/.env" ]]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# Apply sglang/transformers patches (idempotent — Skipped if already applied).
(cd "$REPO_DIR" && uv run python patch_sglang.py)

cleanup() {
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    pkill -9 slim 2>/dev/null || true
    pkill -9 redis 2>/dev/null || true
}

start_ray() {
    cleanup
    uv run ray stop --force 2>/dev/null || true
    sleep 2
    uv run ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats
}

run_train() {
    local train_args="$1"
    local train_script="${2:-$REPO_DIR/train.py}"
    local ray_port="${RAY_DASHBOARD_PORT:-8265}"

    local runtime_env
    runtime_env=$(python -c "
import json, subprocess
def has_nvlink():
    try:
        out = subprocess.check_output('nvidia-smi topo -m 2>/dev/null | grep -o \"NV[0-9][0-9]*\" | wc -l', shell=True, text=True)
        return int(out.strip()) > 0
    except Exception:
        return False
env_vars = {
    'NCCL_NVLS_ENABLE': str(int(has_nvlink())),
    'no_proxy': '127.0.0.1,$MASTER_ADDR',
    'MASTER_ADDR': '$MASTER_ADDR',
}
import os
for k in ('CUDA_LAUNCH_BLOCKING', 'TORCH_USE_CUDA_DSA', 'TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'):
    if k in os.environ:
        env_vars[k] = os.environ[k]
print(json.dumps({'env_vars': env_vars}))
")

    export no_proxy="127.0.0.1"
    uv run ray job submit \
        --address="http://127.0.0.1:${ray_port}" \
        --runtime-env-json="$runtime_env" \
        -- python "$train_script" $train_args
}

wandb_args() {
    local group_name="$1"
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo ""
        return
    fi
    echo "--use-wandb --wandb-project slim --wandb-group $group_name --wandb-key '$WANDB_API_KEY' --disable-wandb-random-suffix"
}
