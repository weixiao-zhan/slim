# Running Tests

## Layout

| Path | Contents | Hardware |
|------|----------|----------|
| `tests/unit/` | pytest tests, mirroring the `slim/` package | CPU; the fused-kernel tests need one GPU and skip without it |
| `tests/e2e/` | end-to-end sweeps and qualification harnesses, their data preparation, `common.sh` and `sanity_check.py` | 2 to 8 GPUs |
| `tests/e2e/nemo/` | `torchrun` qualification harnesses of the NeMo backend | 8 GPUs |

`tests/e2e/template_run.sh` and `tests/e2e/run_strands_35b_a3b_lora.sh` are training recipes, not tests.

## Prerequisites

```bash
# Install dependencies.
# AutoModel, FLA, FlashAttention-2, and FlashAttention-3 are core dependencies.
uv sync --extra dev
uv run python patch_dependencies.py
```

## Unit Tests

```bash
uv run pytest
```

## End-to-End Tests

### Models

Models, datasets, and outputs all live under the repo root. 
The test scripts resolve these via `$REPO_DIR`.

```bash
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

`sweep_e2e.sh` also needs `models/Qwen3.5-35B-A3B-tiny`, a random-weight Qwen3.5 MoE with 4 layers and 16 experts, and a weightless block-FP8 rollout checkpoint that it loads with `--sglang-load-format dummy`. The first weight sync fills it:

```bash
# fp32 block scales
uv run python tools/convert_hf_to_fp8.py \
    --model-dir models/Qwen3.5-4B \
    --save-dir models/Qwen3.5-4B-FP8 \
    --ref-config tools/quant_recipes/qwen35_128blockfp8.json \
    --config-only
```

### Datasets

The tests train on the **mixed** set: DAPO-Math-17k (text math) and geometry3k (vision).

```bash
# -> datasets/mixed/{train,train_balanced,test_math,test_vision}.parquet
uv run python tests/e2e/prepare_mixed.py
```

### Running

Each script starts one ray cluster and runs its jobs one at a time, because later jobs read the rollouts or checkpoints of earlier ones.

```bash
bash tests/e2e/sweep_e2e.sh
bash tests/e2e/run_nemo_mixed_cp_ep.sh
bash tests/e2e/run_nemo_grad_norm_matrix.sh
```

Watch progress and grade externally via ray's own log management:

```bash
uv run ray job list                  # job ids + status
uv run ray job logs <id> --follow    # stream one job
# Grade a finished job: job success, finite actor/critic losses, non-degenerate
# rollout reward, >=2 weight updates. Args are <expect_actor> <expect_critic> (0/1).
uv run ray job logs <id> > out.log
uv run python tests/e2e/sanity_check.py out.log <expect_actor> <expect_critic>
```

### Available Tests

Each test changes several independent axes in one run instead of the full product of their values, so every axis value runs at least once. The unit tests cover the combinations of surrogate, score centering, loss normalization, and batch partitioning.
Live combos generate with SGLang only for the axes that need the engines; the other combos replay saved rollouts.
Pass combo or case names to run a subset.

| Test | Covers | Runs | Model |
|------|--------|------|-------|
| `sweep_e2e.sh` | placement, trainer, weight sync, FP8 rollout, estimator, surrogate, batch layout, LoRA, MoE EP, profiler | 5 live + 8 replay | Qwen3.5-4B (+ FP8 forge), Qwen3.5-35B-A3B-tiny |
| `run_nemo_mixed_cp_ep.sh` | GRPO with TP2 rollout, CP2 and EP8 training, routing replay | 1 | Qwen3.6-35B-A3B |
| `run_nemo_grad_norm_matrix.sh` | gradient norm parity of CP and EP on captured rollouts | 1 capture + 2 | Qwen3.6-35B-A3B |

```bash
bash tests/e2e/sweep_e2e.sh                     # all combos
bash tests/e2e/sweep_e2e.sh ppo_colocate_critic # a single combo
bash tests/e2e/sweep_e2e.sh save resume         # resume needs the checkpoint of save
```

#### `sweep_e2e.sh`

Every combo trains on `train_balanced.parquet` with reasoning disabled, 4 prompts and 8 samples per prompt.
GPU counts scale with `NUM_GPUS` (N). `ppo_disagg` needs at least 4 GPUs.
Live combos run 2 rollouts. Colocated ones run `slim-train` and sync weights over CUDA IPC; disaggregated ones run `slim-train-async` and sync over NCCL.
Full-parameter combos use the Muon default with fp32 master weights; PEFT combos train fp32 adapters with `--optimizer adamw`.
`capture` saves its rollouts, and `prepare_heterogeneous_rollout.py` builds a 32-episode fixture from them in `outputs/sweep-e2e-replay/`.
Replay combos train on that fixture on N GPUs without SGLang and with round-robin assignment, so DP rank 1 holds the vision episodes and the other ranks hold pure text.
The tiny MoE has random weights and cannot generate useful rollouts, but it shares the tokenizer and processor of Qwen3.5-4B, so it replays the same fixture.

| Combo | Rollout | Placement | Model and weights | Estimator and surrogate | Other axes |
|-------|---------|-----------|-------------------|-------------------------|------------|
| `capture` | live, saved | actor and rollout N | 4B LoRA, FP8 rollout | GRPO, `mis` with score centering | KL to the shared reference, CP2, rollout profiler |
| `ppo_disagg` | live | actor N/4, critic N/4, rollout N/2 | 4B, bf16 rollout | PPO, `ppo_clip` | |
| `ppo_colocate_rollout` | live | actor N/2, critic N/2, rollout colocated | 4B LoRA actor and critic, bf16 rollout | PPO, `ppo_clip` | 2 optimizer steps per rollout, saves |
| `ppo_colocate_critic` | live | actor and critic N/2, rollout N/2 | 4B LoRA actor and critic, FP8 rollout | PPO, `ppo_clip` | token loss normalization |
| `ppo_colocate_all` | live | actor, critic and rollout N | 4B, bf16 rollout | PPO, `ppo_clip` | round-robin instead of Karmarkar-Karp |
| `grpo` | replay | actor N | 4B | GRPO, `cis` | trajectory loss normalization |
| `grpo_std` | replay | actor N | 4B | GRPO with group std normalization, `is` | CP2 |
| `gspo` | replay | actor N | 4B | GSPO | fixed micro-batch size 2 |
| `save` | replay | actor N | 4B LoRA | GRPO, `ppo_clip` | KL to the shared reference, saves |
| `resume` | replay | actor N | 4B LoRA | GRPO, `ppo_clip` | resumes `save` |
| `kl_adapter` | replay | actor N | 4B LoRA | GRPO, `ppo_clip` | adapter initialization, KL to the adapter reference |
| `moe` | replay | actor N | tiny MoE expert LoRA | GRPO, `ppo_clip` | EP2, KL to a separate reference model, policy update profiler |
| `ppo_resume` | replay | actor N/2, critic N/2 | 4B LoRA actor and critic | PPO, `ppo_clip` | resumes `ppo_colocate_rollout` |

### Environment Variables

Each test hardcodes its `MODEL_DIR`, `DATASET_DIR`, and `SAVE_DIR` near the top of the
script; edit the script directly to change paths. The only runtime env vars are:

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | (from `.env`) | Weights & Biases API key |
| `NUM_GPUS` | all visible GPUs | Number of GPUs for ray |
| `SWEEP_E2E_REPLAY_DIR` | `outputs/sweep-e2e-replay` | Rollouts of `capture` and the replay fixture |
| `SWEEP_E2E_CKPT_DIR` | `/mnt/sagemaker-nvme/checkpoints/sweep-e2e` | Checkpoints of `sweep_e2e.sh`, deleted when it exits |
