# Rollout (Inference)

Rollout is the data generation phase of the RL loop. slim uses SGLang as its inference backend, orchestrated by Ray.

## SGLang Setup

Loading SGLang requires only one parameter:

- `--hf-checkpoint`: The HuggingFace checkpoint used to initialize SGLang.

Notes:
- Before the first training step, slim syncs parameters from the FSDP backend to SGLang. So `--hf-checkpoint` doesn't need the latest training parameters.
- Override the max context length with `--sglang-context-length`.
- During colocated mode, reduce `--sglang-mem-fraction-static` (e.g., 0.8) to leave memory for training.

### Parameter Pass-Through

slim forwards SGLang parameters with the `--sglang-` prefix:

```bash
--sglang-mem-fraction-static 0.8    # SGLang's --mem-fraction-static
--sglang-context-length 16384       # SGLang's --context-length
--sglang-ep-size 8                  # SGLang's --ep-size
--sglang-enable-dp-attention        # SGLang's --enable-dp-attention
```

Resource scheduling parameters are set by slim directly:
- `--tp-size` is set via `--rollout-num-gpus-per-engine`
- `--model-path` is set via `--hf-checkpoint`

### Router

slim uses [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) to load-balance across SGLang servers. Configure with `--sglang-router-ip` and `--sglang-router-port`, or let slim start one automatically.

Pass sgl-router parameters with a `router` prefix: e.g., `--router-balance-abs-threshold 0`.

### SGLANG_ARGS Example

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

## Rollout Parameters

```bash
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.parquet
   --rollout-shuffle
   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 128

   --rollout-max-context-len 8192
   --rollout-temperature 1
   --balance-data
)
```

The default rollout path expects canonical dataset rows with a finite supported schema:
- Required: `prompt`
- Optional: `label`, `images`, `tools`, `metadata`, `multimodal_inputs`

Avoid adding arbitrary extra top-level dataset columns. Put task-specific auxiliary fields inside `metadata` instead.

The rollout-train constraint: **`(rollout-batch-size x n-samples-per-prompt) = (global-batch-size x num-steps-per-rollout)`**

## Dynamic Sampling

Enable DAPO-style dynamic sampling:

```bash
--rollout-batch-size 32
--n-samples-per-prompt 8
--over-sampling-batch-size 64
--dynamic-sampling-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

Samples 64 prompts, filters groups with zero reward variance, and re-samples when too many are discarded.

## Evaluation

```bash
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data aime /root/aime-2024/aime-2024.parquet
   --eval-n-samples-per-prompt 16
   --eval-max-context-len 16384
   --eval-top-p 1
)
```

## Further Reading

- [SGLang Config](sglang-config.md) -- Multi-model serving, PD disaggregation, YAML deployment
- [Speculative Decoding](speculative-decoding.md) -- MTP-based draft model acceleration
- [Fault Tolerance](fault-tolerance.md) -- Heartbeat-based rollout recovery
- [PD Disaggregation](pd-disaggregation.md) -- Prefill-decode separation
