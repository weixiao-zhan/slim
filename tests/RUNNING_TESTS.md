# Running Tests

## Prerequisites

```bash
# Install dependencies
uv sync --extra dev

# Create .env file with credentials (if not already present)
cat > .env <<'EOF'
WANDB_API_KEY=<your-wandb-key>
EOF
```

## Model & Dataset Setup

### GSM8K (math tests)

```bash
huggingface-cli download Qwen/Qwen3-1.7B-Base \
    --local-dir $HOME/models/Qwen3-1.7B-Base
# Ensure $HOME/datasets/gsm8k/{train,test}.parquet exist
```

### Geo3K (VLM tests)

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir $HOME/models/Qwen3-VL-2B-Instruct
python tests/prepare_geo3k_processor_ready.py
```

## Running

```bash
# Run a test (pick one) — each script handles ray start/stop
bash tests/test_ppo_math.sh
bash tests/test_gspo_math.sh
bash tests/test_ppo_geo3k.sh
bash tests/test_gspo_geo3k.sh
bash tests/test_lora_gspo_geo3k.sh
```

## Available Tests

| Test | Algorithm | Task | Model | Notes |
|------|-----------|------|-------|-------|
| `test_gspo_math.sh` | GSPO | GSM8K | Qwen3-1.7B-Base | 8 actor GPUs colocated |
| `test_ppo_math.sh` | PPO | GSM8K | Qwen3-1.7B-Base | 4 actor + 4 critic GPUs |
| `test_gspo_geo3k.sh` | GSPO | Geo3K | Qwen3-VL-2B-Instruct | 8 actor GPUs, VLM |
| `test_ppo_geo3k.sh` | PPO | Geo3K | Qwen3-VL-2B-Instruct | 4 actor + 4 critic, VLM |
| `test_lora_gspo_geo3k.sh` | GSPO + LoRA | Geo3K | Qwen3-VL-2B-Instruct | LoRA r=128, PEFT |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL_DIR` | `$HOME/models/Qwen3-VL-2B-Instruct` | VLM model path (geo3k tests) |
| `VLM_DATASET_DIR` | `$HOME/datasets/geo3k` | Geo3K dataset path |
| `LLM_MODEL_DIR` | `$HOME/models/Qwen3-1.7B-Base` | LLM model path (math tests) |
| `LLM_DATASET_DIR` | `$HOME/datasets` | LLM dataset path |
| `SAVE_DIR` | `$HOME/outputs/<test>` | Checkpoint save path |
| `WANDB_API_KEY` | (from `.env`) | Weights & Biases API key |
| `NUM_GPUS` | `8` | Number of GPUs for ray |
