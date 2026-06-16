# Running Tests

## Prerequisites

```bash
# Install dependencies
uv sync --extra dev
```

## Models
```bash
hf download Qwen/Qwen3.5-4B --local-dir $HOME/models/Qwen3.5-4B
```

Forge FP8 if needed
```bash
uv run python tools/convert_hf_to_fp8.py \
    --model-dir $HOME/models/Qwen3.5-4B \
    --save-dir $HOME/models/Qwen3.5-4B-FP8 \
    --ref-config tools/fp8_recipes/qwen35_official.json
```

## Datasets

Two source datasets — **DAPO-Math-17k** (text math) and **geometry3k** (vision/VLM) —
plus a **mixed** set derived from both.

```bash
# DAPO-17k (math tests): open-r1/DAPO-Math-17k-Processed
# -> ~/datasets/dapo17k/train.parquet
uv run python tests/prepare_dapo17k_tokenizer_ready.py

# Geo3K (vision tests): hiyouga/geometry3k
# -> ~/datasets/geo3k/{train,test}.parquet
uv run python tests/prepare_geo3k_processor_ready.py

# Mixed (GRPO mixed tests): DAPO-Math-17k (text) + geometry3k (vision)
# -> ~/datasets/mixed/{train,test_math,test_vision}.parquet
uv run python tests/prepare_mixed.py
```

## Running

```bash
# Run a test (pick one) — each script handles ray start/stop
bash tests/test_gspo_math.sh
bash tests/test_gspo_geo3k.sh
bash tests/test_ppo_geo3k.sh
bash tests/test_lora_gspo_geo3k.sh
bash tests/test_grpo_mixed_cis.sh
bash tests/test_grpo_mixed_fp8_cis.sh
```

### on Blackwell (SM120)

SM120 need following treatment:

- **Training attention:** FA3/FA2 have no SM120 kernel. Using sdpa as training-side attention `--attn-implementation flash_attention_3` → `--attn-implementation sdpa`
- **Rollout gemm:** SGL default to DeepGEMM when runing fp8 on backwell, which expects ue8m0 scales. To use fp32 block scales: use `--sglang-fp8-gemm-backend triton` in (`test_grpo_mixed_fp8_cis.sh`)

## Available Tests

All tests use **Qwen3.5-4B** (`test_grpo_mixed_fp8_cis.sh` additionally uses an
FP8-forged copy as the rollout checkpoint).

| Test | Algorithm | Dataset | Notes |
|------|-----------|---------|-------|
| `test_ppo_geo3k.sh` | PPO | Geo3K | 4 actor + 4 critic, VLM |
| `test_gspo_math.sh` | GSPO | DAPO-17k | 8 actor GPUs colocated |
| `test_gspo_geo3k.sh` | GSPO | Geo3K | 1 actor GPU, VLM |
| `test_lora_gspo_geo3k.sh` | GSPO + LoRA | Geo3K | LoRA r=128, PEFT, VLM |
| `test_grpo_mixed_cis.sh` | GRPO + CIS | Mixed (math+vision) | colocated |
| `test_grpo_mixed_fp8_cis.sh` | GRPO + CIS | Mixed (math+vision) | colocated, FP8 weight sync |

## Environment Variables

Each test hardcodes its `MODEL_DIR`, `DATASET_DIR`, and `SAVE_DIR` near the top of the
script — edit the script directly to change paths. The only runtime env vars are:

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | (from `.env`) | Weights & Biases API key |
| `NUM_GPUS` | `8` | Number of GPUs for ray |
