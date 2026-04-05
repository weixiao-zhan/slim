# Training Backend

slim uses PyTorch FSDP2 as its training backend. FSDP loads HuggingFace weights directly via `from_pretrained()` (using `AutoModelForCausalLM` for text models and `AutoModelForImageTextToText` for vision models) -- no checkpoint conversion needed.

## Installation

```bash
git clone <your-slim-repo> slim
cd slim
pip install -e .       # or: uv sync
```

Supported hardware: NVIDIA H100/H200, B200 series.

## GPU Allocation

```bash
--actor-num-nodes 1              # nodes for training
--actor-num-gpus-per-node 4      # GPUs per node for training
--rollout-num-gpus 4             # GPUs for inference (SGLang)
--rollout-num-gpus-per-engine 2  # GPUs per SGLang engine (like tp_size)
```

### Colocated Mode

Share GPUs between training and inference with CPU offloading:

```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--colocate                       # overrides --rollout-num-gpus to match actor GPUs
```

## Checkpoints

```bash
--hf-checkpoint /root/Model          # HF checkpoint for SGLang + tokenizer
--ref-load /root/Model               # reference model (for KL)
--load /root/Model_slim/        # actor checkpoint (resume training)
--ckpt-step 20                       # optional: resume a specific iter_0000020 checkpoint
--save /root/Model_slim/        # save path
--save-interval 20                   # save every N steps
```

FSDP loads HF checkpoints directly via `from_pretrained()`. To convert FSDP checkpoints back to HF format:

```bash
python tools/convert_fsdp_to_hf.py \
  --input-dir /path/to/fsdp_ckpt/ \
  --output-dir /root/Model-converted \
  --origin-hf-dir /root/Model
```

## Data Format

slim supports `.jsonl` and `.parquet` formats. Row slicing: `path/to/data.jsonl@[start:end]`.

Each row should use the supported finite column set:
- Required: `prompt`
- Optional: `label`, `images`, `tools`, `metadata`

For other multimodal inputs (e.g. videos), place them inside a `multimodal_inputs` dict rather than as top-level columns.

The default loaders and rollout path expect this fixed schema rather than arbitrary extra top-level columns.
If you need to carry additional per-row information, place it under `metadata`.

When the prompt is a list of chat messages, `apply_chat_template` is called automatically.
When the prompt is a plain string, it is tokenized directly.

Example data entry:
```json
{
  "prompt": [{"role": "user", "content": "Solve: ...", "step_loss_mask": 1}],
  "label": "34",
  "metadata": {"source": "custom-dataset"}
}
```

## Performance

```bash
--gradient-checkpointing         # reduce memory, more compute
--fsdp-cpu-offload               # offload params to CPU
--use-dynamic-batch-size         # pack samples efficiently (recommended)
--max-tokens-per-gpu 4608        # max tokens per GPU per micro-batch
```

## RL Algorithms

### GRPO (recommended)

```bash
--advantage-estimator grpo
--n-samples-per-prompt 8
--use-kl-loss
--kl-loss-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28
```

Also supports: `gspo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `ppo`.

### PPO

PPO requires a separate critic model (additional GPU allocation):

```bash
--advantage-estimator ppo
--critic-num-nodes 1
--critic-num-gpus-per-node 4
--critic-load /path/to/critic
```

### Optimizer

```bash
--optimizer adam
--lr 1e-6
--lr-decay-style constant
--weight-decay 0.1
```

## Multi-Node Training

Start a Ray cluster, then submit:

```bash
# Node 0 (HEAD)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8

# Other nodes
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8

# Submit job
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py --hf-checkpoint <model> ...
```

## FAQ

1. **Garbled text during training?** -- Check that `--hf-checkpoint` points to a valid HF checkpoint and that weights were synced correctly from FSDP to SGLang. See [Debugging](debug.md).

2. **Task stuck on Ray submission?** -- Verify total GPU count >= `actor_num_nodes * actor_num_gpus_per_node + rollout_num_gpus` (or just actor GPUs if `--colocate`).

3. **OOM during training?** -- Lower `--max-tokens-per-gpu`. Only active with `--use-dynamic-batch-size`.

4. **How to resume training?** -- Set `--load` to your `--save` directory. To resume a specific checkpoint, add `--ckpt-step N`.

5. **Batch size calculation?** -- One rollout produces `rollout_batch_size * n_samples_per_prompt` samples. Use `--num-steps-per-rollout` to control update frequency.

6. **High gradient norm / training crash?** -- Verify data matches model's chat template. See [Debugging](debug.md).

7. **SGLang generation hangs?** -- Check stop tokens: `--rollout-stop` or `--rollout-stop-token-ids`.

## Further Reading

- [Low Precision](low-precision.md) -- FP8 inference, INT4 QAT
- [Reproducibility](reproducibility.md) -- Deterministic bitwise training
- [Debugging](debug.md) -- Precision alignment, separate debugging
- [Profiling](profiling.md) -- Rollout performance analysis
- [CI](ci.md) -- GitHub Actions workflow
