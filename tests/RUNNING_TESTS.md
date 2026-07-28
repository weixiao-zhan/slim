# Running Tests

## Prerequisites

```bash
# Install dependencies.
# AutoModel, FLA, FlashAttention-2, and FlashAttention-3 are core dependencies.
uv sync --extra dev
```

## Models

Models, datasets, and outputs all live under the repo root. 
The test scripts resolve these via `$REPO_DIR`.

```bash
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

### FP8 checkpoints (prerequisite for `sweep_fp8.sh`)

`sweep_fp8.sh` needs two pre-forged FP8 copies of the base model alongside the bf16 one:

```bash
# fp32 block scales
uv run python tools/convert_hf_to_fp8.py \
    --model-dir models/Qwen3.5-2B \
    --save-dir models/Qwen3.5-2B-FP8 \
    --ref-config tools/fp8_recipes/qwen35_official.json

# ue8m0 (power-of-two) block scales
uv run python tools/convert_hf_to_fp8.py \
    --model-dir models/Qwen3.5-2B \
    --save-dir models/Qwen3.5-2B-FP8-ue8m0 \
    --ref-config tools/fp8_recipes/qwen35_ue8m0.json
```

## Datasets

Two source datasets — **DAPO-Math-17k** (text math) and **geometry3k** (vision/VLM) —
plus a **mixed** set derived from both.

```bash
# DAPO-17k (math tests): open-r1/DAPO-Math-17k-Processed
# -> datasets/dapo17k/train.parquet
uv run python tests/prepare_dapo17k_tokenizer_ready.py

# Geo3K (vision tests): hiyouga/geometry3k
# -> datasets/geo3k/{train,test}.parquet
uv run python tests/prepare_geo3k_processor_ready.py

# Mixed (GRPO mixed tests): DAPO-Math-17k (text) + geometry3k (vision)
# -> datasets/mixed/{train,test_math,test_vision}.parquet
uv run python tests/prepare_mixed.py
```

## Running

Every script starts one ray cluster, then submits its job(s) with
`ray job submit --no-wait` and exits. Jobs queue on the cluster: each job's
driver blocks in `ray.get(pg.ready())` until the previous job's
`RolloutManager.dispose` releases the GPUs, so a sweep's combos run sequentially without restarting ray between them.

```bash
bash tests/test_grpo_profile.sh
bash tests/sweep_placement.sh
bash tests/sweep_dataset.sh
bash tests/sweep_algo.sh
bash tests/sweep_surrogate.sh
bash tests/sweep_fp8.sh
bash tests/sweep_moe_rollout.sh
bash tests/run_nemo_mixed_cp_ep.sh
```

Watch progress and grade externally via ray's own log management:

```bash
uv run ray job list                  # job ids + status
uv run ray job logs <id> --follow    # stream one job
# Grade a finished job: job success, finite actor/critic losses, non-degenerate
# rollout reward, >=2 weight updates. Args are <expect_actor> <expect_critic> (0/1).
uv run ray job logs <id> > out.log
uv run python tests/sanity_check.py out.log <expect_actor> <expect_critic>
```

### on Blackwell (SM120)

SM120 need following treatment:

- **Training attention:** packed Qwen3.5 uses AutoModel's FlashAttention 2 varlen CP kernel when available and falls back to PyTorch SDPA.
- **Rollout gemm:** SGL default to DeepGEMM when runing fp8 on backwell, which expects ue8m0 scales. To use fp32 block scales: use `--sglang-fp8-gemm-backend triton` in (`sweep_fp8.sh`)

## Available Tests

Two kinds of tests live here:

- **Sweeps** (`sweep_*.sh`) — each submits a matrix of combos along one axis on a
  single 8-GPU node. All combos share one `COMMON_ARGS` block sized to **3 rollout
  steps × 8 prompts × 4 samples**, `max-context-len 8192`, `max-tokens-per-gpu
  8192`. Run all combos with no args, or a subset by passing combo names.
- **Standalone tests** — single runs that exercise an orthogonal axis (precision,
  profiling) not covered by a sweep.

### Sweeps

| Sweep | Axis | Combos | Model |
|-------|------|--------|-------|
| `sweep_placement.sh` | (rollout-colocate, critic-colocate) placement | 6 PPO combos + HSDP control | Qwen3.5-2B |
| `sweep_dataset.sh` | PPO × data modality | `math`, `vision` | Qwen3.5-2B |
| `sweep_algo.sh` | advantage estimator | `grpo`, `gspo`, `ppo` | Qwen3.5-2B |
| `sweep_surrogate.sh` | GRPO policy surrogate | `ppo_clip`, `is`, `tis`, `cis` | Qwen3.5-2B |
| `sweep_fp8.sh` | GRPO+CIS rollout-weight precision | `bf16`, `fp8_fp32`, `fp8_ue8m0` | Qwen3.5-2B (+ FP8 forges) |
| `sweep_moe_rollout.sh` | MoE rollout parallelism (R3 on) | `tp1`, `tp4`, `tp4_ep4` | Qwen3.6-35B-A3B (MoE) |

```bash
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
uv run python tests/prepare_mixed.py
bash tests/sweep_placement.sh                       # all combos
bash tests/sweep_placement.sh colocate_critic       # a single combo
bash tests/sweep_surrogate.sh cis                   # just the CIS combo
```

The non-MoE sweeps use **Qwen3.5-2B**. `sweep_moe_rollout.sh` runs on the
**Qwen3.6-35B-A3B** MoE checkpoint, which lives on the NVMe disk and is referenced
through the `models/<name>` symlink convention (`models/Qwen3.6-35B-A3B ->
/opt/dlami/nvme/models/Qwen3.6-35B-A3B`). All sweeps use `flash_attention_3`
(Hopper); on A100/L40s switch to `flash_attention_2`, on Blackwell/SM120 use `sdpa`.

### Standalone tests

Tests default to the **mixed** (math+vision) dataset unless noted.

| Test | Algorithm | Notes |
|------|-----------|-------|
| `test_grpo_profile.sh` | GRPO + CIS | torch profiler harness |
| `run_nemo_mixed_cp_ep.sh` | GRPO | Qwen3.5 MoE, mixed math and Geometry3K, SGLang TP2, NeMo CP2 and EP8 |

## Environment Variables

Each test hardcodes its `MODEL_DIR`, `DATASET_DIR`, and `SAVE_DIR` near the top of the
script — edit the script directly to change paths. The only runtime env vars are:

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | (from `.env`) | Weights & Biases API key |
| `NUM_GPUS` | `8` | Number of GPUs for ray |
