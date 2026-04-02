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
uv run huggingface-cli download Qwen/Qwen3-1.7B-Base \
    --local-dir /home/ubuntu/models/Qwen3-1.7B-Base
# Ensure /home/ubuntu/datasets/gsm8k/{train,test}.parquet exist
```

### Geo3K (VLM tests)

```bash
uv run huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir /home/ubuntu/models/Qwen3-VL-2B-Instruct
uv run python tests/prepare_geo3k_processor_ready.py
```

Geo3K canonical schema:
- `prompt`: chat-format messages with `{"type": "image"}` placeholders
- `label`: ground-truth answer
- `images`: ordered image list aligned with the prompt placeholders

Dataset schema note:
- Test datasets should stick to the finite supported top-level columns: `prompt`, `label`, and optional `images`, `videos`, `audio`, `tools`, `metadata`
- Put any extra task-specific fields under `metadata` instead of adding new top-level columns

## Running

```bash
# Start Ray
uv run ray stop --force; uv run ray start --head --num-gpus 8 --disable-usage-stats

# Run a test (pick one)
uv run python tests/test_ppo_math.py
uv run python tests/test_gspo_math.py
uv run python tests/test_ppo_geo3k.py
uv run python tests/test_gspo_geo3k.py
uv run python tests/test_lora_gspo_geo3k.py

# Follow logs
uv run ray job logs --follow $(uv run ray job list 2>&1 | \
    grep -oP "submission_id='[^']*'" | head -1 | grep -oP "'[^']*'" | tr -d "'")

# Stop Ray when done
uv run ray stop --force
```

## Available Tests

| Test | Algorithm | Task | Model | Notes |
|------|-----------|------|-------|-------|
| `test_grpo_math.py` | GRPO | GSM8K | Qwen3-1.7B-Base | 8 actor GPUs colocated |
| `test_ppo_math.py` | PPO | GSM8K | Qwen3-1.7B-Base | 4 actor + 4 critic GPUs |
| `test_grpo_geo3k.py` | GRPO | Geo3K | Qwen3-VL-2B-Instruct | 8 actor GPUs, VLM |
| `test_ppo_geo3k.py` | PPO | Geo3K | Qwen3-VL-2B-Instruct | 4 actor + 4 critic, VLM |
| `test_lora_gspo_geo3k.py` | GSPO + LoRA | Geo3K | Qwen3-VL-2B-Instruct | LoRA r=128, PEFT required |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL_DIR` | `/home/ubuntu/models/Qwen3-VL-2B-Instruct` | VLM model path (geo3k tests) |
| `VLM_DATASET_DIR` | `/home/ubuntu/datasets/geo3k` | Geo3K dataset path |
| `SAVE_DIR` | `/home/ubuntu/checkpoints/lora-gspo-geo3k-qwen3vl2b` | Checkpoint save path (lora test) |
| `WANDB_API_KEY` | (from `.env`) | Weights & Biases API key |
