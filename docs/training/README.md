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
--actor-num-gpus 4               # total GPUs for training
--actor-num-gpus-per-replica 4    # GPUs per FSDP replica (==total: full shard; ==1: DDP; between: HSDP)
--num-gpus-per-node 8            # physical GPUs per node (packing / ports)
--rollout-num-gpus 4             # GPUs for inference (SGLang)
--rollout-num-gpus-per-replica 2  # GPUs per SGLang engine (like tp_size)
```

### Colocated Mode

Share GPUs between training and inference with CPU offloading:

```bash
--actor-num-gpus 8
--rollout-colocate               # overrides --rollout-num-gpus to match training GPUs
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

## Precision

- `--master-weight-dtype fp32` promotes all weights and optimizer states to fp32 at load time (standard mixed-precision practice). Leave unset to use checkpoint's dtypes for weigths and optimzier states.
- `--compute-dtype bf16` (or `fp16`) runs forward/backward in that dtype via the FSDP2 `MixedPrecisionPolicy`, while gradient reduction stays in fp32. Leave unset to compute in the storage dtype.

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

Also supports: `gspo`, `ppo_gae`.

Policy surrogate defaults to PPO clipping:

```bash
--policy-surrogate ppo_clip
```

Also supports: `is`, `tis`, `cis`.

### PPO GAE

PPO GAE requires a separate critic model (additional GPU allocation):

```bash
--advantage-estimator ppo_gae
--critic-num-gpus 4              # must equal --actor-num-gpus
--critic-load /path/to/critic
# add --critic-colocate to time-share the critic on the actor GPUs
```

### Optimizer

```bash
--optimizer adam
--lr 1e-5
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

2. **Task stuck on Ray submission?** -- Verify total GPU count >= `actor_num_gpus + critic_num_gpus + rollout_num_gpus` (drop critic if disaggregated-off, drop rollout if `--rollout-colocate`, drop critic if `--critic-colocate`).

3. **OOM during training?** -- Lower `--max-tokens-per-gpu`. Only active with `--use-dynamic-batch-size`.

4. **How to resume training?** -- Set `--load` to your `--save` directory. To resume a specific checkpoint, add `--ckpt-step N`.

5. **Batch size calculation?** -- One rollout produces `rollout_batch_size * n_samples_per_prompt` samples. Use `--num-steps-per-rollout` to control update frequency.

6. **High gradient norm / training crash?** -- Verify data matches model's chat template. See [Debugging](debug.md).

7. **SGLang generation hangs?** -- Check stop tokens: `--rollout-stop` or `--rollout-stop-token-ids`.

## Further Reading

- [Reproducibility](reproducibility.md) -- Deterministic bitwise training
- [Debugging](debug.md) -- Precision alignment, separate debugging
- [Profiling](profiling.md) -- Rollout performance analysis
- [CI](ci.md) -- GitHub Actions workflow
