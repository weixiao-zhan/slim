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
- `--tp-size` is set via `--rollout-num-gpus-per-replica`
- `--model-path` is set via `--hf-checkpoint`

### Router

slim uses [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) to load-balance across SGLang servers. Configure with `--router-ip` and `--router-port`, or let slim start one automatically.

Pass sgl-router parameters with a `router` prefix: e.g., `--router-balance-abs-threshold 0`.

### SGLANG_ARGS Example

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-replica 2
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

The rollout-train constraint: **`(rollout-batch-size x n-samples-per-prompt) = (global-batch-size x num-steps-per-rollout)`**

## Data Format

Datasets are loaded through HuggingFace `datasets`. A `.jsonl` or `.parquet` file is loaded via `load_dataset`, and a directory is loaded as a saved HF dataset via `load_from_disk`. Use `path:split` to pick a split and `@[start:end]` to slice rows, e.g. `path/to/ds:train@[0:100]`.

slim imposes no schema on a row/example and stores the entire row/example verbatim in `episode.example`. The rollout function receives it untouched and decides how to interpret it. So a custom generate / reward function can read whatever columns your task needs.

The **default** generate function (`slim.rollout.sglang_rollout`) looks for these fields in `episode.example`:
- `prompt` (required) — a plain string (tokenized directly) or a list of chat messages (`apply_chat_template` is applied automatically).
- `tools` — tool/function definitions passed to the chat template.
- `images`, `videos`, `audios` — top-level multimodal fields, forwarded to the processor (requires a multimodal checkpoint).

The default reward path additionally reads `label` and `metadata`.

Example data entry:
```json
{
  "prompt": [{"role": "user", "content": "Solve: ..."}],
  "label": "34",
  "metadata": {"source": "custom-dataset"}
}
```

## Dynamic Sampling

Enable DAPO-style dynamic sampling:

```bash
--rollout-batch-size 32
--n-samples-per-prompt 8
--over-sampling-batch-size 64
--rollout-group-filter-path slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

Keeps 64 prompts in flight running or queued on the concurrency semaphore, continuously refills the pool back up to 64 until a full `rollout-batch-size` of keepers is collected.

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
- [Low Precision Inference](low-precision.md) -- FP8 rollout
