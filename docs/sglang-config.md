# SGLang Engine Configuration

slim uses [SGLang](https://github.com/sgl-project/sglang) as its rollout (inference) backend.
The deployment has two layers:
- **SGLang engines**: each engine is a standalone HTTP server holding one replica of the model. Engines receive weight updates from training after each step.
- **Router** (sglang-router): a load-balancer sitting in front of the engines. All generation requests go through the router, which distributes them across engines.

## Basic Setup

| Flag | Purpose |
|------|---------|
| `--hf-checkpoint` | HuggingFace checkpoint for SGLang engines and tokenizer. Can point to a quantized (e.g. FP8) checkpoint. |
| `--rollout-num-gpus` | Total GPUs for rollout. |
| `--rollout-num-gpus-per-replica` | GPUs per engine (= TP size). The number of engines is `rollout_num_gpus / rollout_num_gpus_per_replica`. |

See [Placement](placement.md) for how engines are mapped onto GPUs.

### Parameter pass-through

`--sglang-` prefixed args are stripped the prefix and supplies the remainder directly to SGLang's `ServerArgs`:
```
--sglang-mem-fraction-static 0.85 →  ServerArgs(mem_fraction_static=0.85)
--sglang-context-length 16384     →  ServerArgs(context_length=16384)
--sglang-enable-dp-attention      →  ServerArgs(enable_dp_attention=True)
```

A small set of fields (`model_path`, `tp_size`, `port`, `base_gpu_id`) are managed by slim and cannot be overridden.

## YAML Config (`--sglang-config`)

For advanced deployments (PD disaggregation, multi-model serving, per-group overrides), pass a YAML config file:

```bash
--sglang-config config/rollout.yaml
```

Top-level structure:

```yaml
sglang:
  - name: actor
    model_path: /models/Qwen3-8B
    update_weights: true
    num_gpus_per_replica: 2
    server_groups:
      - worker_type: regular
        num_gpus: 8
```

### Model-level fields

| Field | Description |
|-------|-------------|
| `name` (required) | Unique model name (e.g. `"actor"`, `"ref"`, `"reward"`). |
| `model_path` | HF checkpoint path. Falls back to `--hf-checkpoint`. |
| `update_weights` | Whether this model receives weight updates. Auto-inferred from whether `model_path` matches `--hf-checkpoint`. |
| `num_gpus_per_replica` | Default GPUs per engine for all groups in this model. |
| `server_groups` (required) | List of server group configurations. |

### Server group fields

| Field | Description |
|-------|-------------|
| `worker_type` (required) | One of `regular`, `prefill`, `decode`, `placeholder`, `encoder`. |
| `num_gpus` (required) | Total GPUs allocated to this group. |
| `num_gpus_per_replica` | Per-group override for GPUs per engine. |
| `overrides` | SGLang `ServerArgs` field overrides (highest priority). |

### Worker types

| Type | Behavior |
|------|----------|
| `regular` | Handles both prefill and decode. |
| `prefill` | Prefill-only engine for PD disaggregation. |
| `decode` | Decode-only engine for PD disaggregation. |
| `placeholder` | Reserves GPU slots without creating engines. |
| `encoder` | Encoder-only engine for EPD disaggregation. |

### Resolution priority

1. Per-group `overrides` (highest)
2. Per-group `num_gpus_per_replica`
3. Model-level `num_gpus_per_replica`
4. CLI `--rollout-num-gpus-per-replica` (lowest)

All server groups within a single model entry must resolve to the same `model_path`.

### Multi-model serving

The YAML config supports multiple model entries, each with its own router and engine pool:

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 8

  - name: ref
    model_path: /models/Qwen3-8B-ref
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
```

Access model URLs in custom rollout functions via `get_model_url(args, "actor", "/generate")`.
Only models with `update_weights: true` receive weight updates; frozen models retain their initial weights.

### Mutual exclusion

These flags are mutually exclusive: `--sglang-config`, `--prefill-num-servers`, `--rollout-external`.

## PD Disaggregation

Prefill-Decode disaggregation splits inference into separate prefill and decode server groups.
Prefill engines handle prompt processing and transfer KV cache to decode engines, which handle autoregressive generation.

```yaml
sglang:
  - name: actor
    num_gpus_per_replica: 2
    server_groups:
      - worker_type: prefill
        num_gpus: 4
      - worker_type: decode
        num_gpus: 8
        num_gpus_per_replica: 4
```

This yields 2 prefill engines (TP=2) and 2 decode engines (TP=4).

Prefill is compute-bound so fewer large-TP engines are efficient; decode is memory-bound so more smaller-TP engines can increase throughput.
Total GPUs across all groups must equal `--rollout-num-gpus`.

## Speculative Decoding

For models with MTP layers (e.g. GLM-4.7, DeepSeek-V3/R1), speculative decoding could accelerate generation under low concurrency by drafting multiple tokens that the target model verifies in a batch.

```bash
--sglang-speculative-draft-model-path \ # For a separately trained draft model
--sglang-speculative-algorithm EAGLE \
--sglang-speculative-num-steps 3 \
--sglang-speculative-eagle-topk 1 \
--sglang-speculative-num-draft-tokens 4
```

Slim current does not support online MTP training (will revisit when HF transformers have good MTP support).

## Fault Tolerance

slim includes heart beat health monitoring (via `/health` not `/health_generate`; the later could timeout under extreme workloads and may kill health engine).
When an engine becomes unresponsive, it is killed and restarted with a fresh weight sync before the next rollout round.

| Flag | Default | Description |
|------|---------|-------------|
| `--rollout-health-check-first-wait` | 0 | Grace period (seconds) before health checks begin. Increase for large models with long compilation. |
| `--rollout-health-check-interval` | 30 | Seconds between health check rounds. |
| `--rollout-health-check-timeout` | 30 | Seconds to wait for a response before declaring the engine dead. |
| `--rollout-disable-fault-tolerance` | false | Disable fault tolerance entirely. |

## Low-Precision Inference (FP8)

slim can run rollout engines with block-FP8 quantized weights while training in full precision.
This reduces rollout GPU memory and increases throughput at the cost of a small log-prob mismatch between rollout and training (corrected by `--policy-surrogate`; see [Training Loss](training-loss.md)).

### Dual-checkpoint setup

```bash
--hf-checkpoint /models/Qwen3-8B-FP8    # FP8 for rollout
--load /models/Qwen3-8B                  # BF16 for training init
```

After each training step, updated BF16 weights are re-quantized online to match the FP8 format and pushed to engines.

### Block-FP8 format

- e4m3 weights, 128x128 blocks, one scale per block stored as `<weight>.weight_scale_inv`.
- Two scale formats: `fp32` (default, works on SM89/SM90/SM120) and `ue8m0` (power-of-two scales for DeepGEMM on Blackwell SM120).
- Modules listed in `quantization_config.modules_to_not_convert` remain unquantized.

### Conversion tool

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir /models/Qwen3-8B \
    --save-dir /models/Qwen3-8B-FP8 \
    --ref-config /models/Qwen3-8B-FP8-reference \
    --block-size 128 128 \
    --scale-fmt ue8m0
```

`--ref-config` points to an existing FP8 model whose `quantization_config` defines the recipe.

## Router

Each model gets its own sglang-router instance for load balancing.

| `--router-policy` | Description |
|--------------------|-------------|
| `cache_aware` (default) | Routes based on prefix cache hit potential. |
| `round_robin` | Simple round-robin. |
| `random` | Random selection. |
| `consistent_hashing` | Session affinity via `X-SMG-Routing-Key` header. |

For multi-turn conversations, use `consistent_hashing` with `episode.session_id` to pin all turns of a session to the same engine (preserving KV cache).

## External Engines

For pre-deployed SGLang engines managed externally:

```bash
--rollout-external \
--rollout-external-engine-addrs host1:30000 host2:30000
```

slim does not spawn engines in this mode.
It waits for each to become healthy, then uses them for generation.
Fault recovery is disabled (the external operator manages lifecycle).

---

**See also:** [Data Layout](data-layout.md) | [Placement & Weight Update](placement.md) | [Customization](customization.md)
