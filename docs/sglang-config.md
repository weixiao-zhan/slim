# SGLang Engine Configuration

slim uses [SGLang](https://github.com/sgl-project/sglang) as its rollout (inference) backend.
The deployment has two layers:
- **SGLang engines**: each engine is a standalone HTTP server holding one replica of the model. Engines receive weight updates from training after each step.
- **Router** (sglang-router): a load-balancer sitting in front of the engines. All generation requests go through the router, which distributes them across engines.

## Basic Setup

| Flag | Purpose |
|------|---------|
| `--hf-checkpoint` | HuggingFace checkpoint for SGLang engines and tokenizer. `--rollout-load` can point the engines to a quantized (e.g. FP8) checkpoint. |
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

## Low-Precision Inference (FP8)

slim can run rollout engines with block-FP8 quantized weights while training in full precision.
This reduces rollout GPU memory and increases throughput at the cost of a small log-prob mismatch between rollout and training (corrected by `--policy-surrogate`; see [Training Loss](training-loss.md)).

### Dual-checkpoint setup

```bash
--hf-checkpoint /models/Qwen3-8B         # BF16 for training init
--rollout-load /models/Qwen3-8B-FP8      # FP8 for rollout
```

After each training step, updated BF16 weights are re-quantized online to match the FP8 format and pushed to engines.

### Block-FP8 format

- e4m3 weights, 128x128 blocks, one scale per block stored as `<weight>.weight_scale_inv`.
- Two scale formats: `fp32` (default, works on SM89/SM90/SM120) and `ue8m0` (power-of-two scales for DeepGEMM on Blackwell SM120).
- Modules listed in `quantization_config.modules_to_not_convert` remain unquantized.
- Fused MoE experts (`mlp.experts.gate_up_proj` / `down_proj`) are written split per expert, so sglang loads their scales.

### Conversion tool

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir /models/Qwen3-8B \
    --save-dir /models/Qwen3-8B-FP8 \
    --ref-config /models/Qwen3-8B-FP8-reference \
    --block-size 128 128 \
    --scale-fmt ue8m0
```

`--ref-config` points to an existing FP8 model, or a recipe in `tools/quant_recipes/`, whose `quantization_config` defines the recipe.

### Other formats

Recipes in `tools/quant_recipes/` share one keep-list for the Qwen3.5 family:

| Recipe | Format | Tool |
|---|---|---|
| `qwen35_128blockfp8.json` | 128x128 block FP8, fp32 scales (official Qwen3.5 FP8) | `tools/convert_hf_to_fp8.py` |
| `qwen35_128blockfp8ue8m0.json` | 128x128 block FP8, UE8M0 scales (DeepGEMM) | `tools/convert_hf_to_fp8.py` |
| `qwen35_mxfp8.json` | OCP MXFP8: 1x32 blocks, UE8M0 scales | `tools/convert_hf_to_fp8.py` |
| `qwen35_nvfp4.json` | NVFP4 W4A4 (ModelOpt export, served as `modelopt_fp4`) | `tools/convert_hf_to_nvfp4.py` |

NVFP4 needs activation calibration, so it uses NVIDIA ModelOpt (`uv sync --extra quant`):

```bash
python tools/convert_hf_to_nvfp4.py \
    --model-dir models/Qwen3.5-9B \
    --save-dir models/Qwen3.5-9B-NVFP4 \
    --ref-config tools/quant_recipes/qwen35_nvfp4.json
```

## Router

Each model gets its own sglang-router instance for load balancing.

| `--router-policy` | Description |
|--------------------|-------------|
| `cache_aware` (default) | Routes based on prefix cache hit potential. |
| `round_robin` | Simple round-robin. |
| `random` | Random selection. |
| `consistent_hashing` | Session affinity via `X-SMG-Routing-Key` header. |

For multi-turn conversations, use `consistent_hashing` with `episode.session_id` to pin all turns of a session to the same engine (preserving KV cache).

## Endpoints and Tokenization Ownership

To avoid retokenization drift and ensure token-in-token-out, there should be one tokenization source of truth. Slim supports either slim's custome generate function or SGLang owns the process:

| SGLang Endpoint | Tokenization Owner | Use |
|----------|-------|-----|
| `/generate` | slim | Default rollout. slim tokenizes with `AutoTokenizer`/`AutoProcessor` and sends token ids plus processor tensors. |
| `/v1/chat/completions` | SGLang engine | Agent frameworks that speak OpenAI. The engine applies the chat template and returns the ids it ran. |

See [Data Layout](data-layout.md) for how the recorded fields are consumed by training. We patched sglang-router to pass additional fields though the rust router.

### `/generate` 

Request:

| Field | Content |
|-------|---------|
| `input_ids` | prompt token ids |
| `image_data[0]` = `{"format": "processor_output", ...}` | processor tensors |
| `sampling_params` | `temperature`, `max_new_tokens` from what is left of `episode.max_tokens`, and `sampling_seed` (`episode.sampling_seed`) under `--sglang-enable-deterministic-inference` |
| `return_logprob: true` | gates `meta_info.output_token_logprobs` |
| `top_logprobs_num` | `K` under `--use-score-centering K`; gates `meta_info.output_top_logprobs` |
| `return_routed_experts`, `routed_experts_start_len` | expert indices, under `--use-rollout-routing-replay`; set the start to the number of tokens already recorded, so only record and append the latest user turn + model turn. Note the capture is keyed by KV-cache slot and earlier tokens' should not be re-read |

Response:

| Field | Location |
|-------|----------|
| output token ids and logprobs | `meta_info.output_token_logprobs` = `{"logprobs": float32 [T], "token_ids": int32 [T]}` |
| sampler top-$K$ | `meta_info.output_top_logprobs` = `{"logprobs": float32 [T, K], "token_ids": int32 [T, K]}`, one row per output token |
| expert indices | `meta_info.routed_experts`, int32, one row per prediction over `[routed_experts_start_len, seqlen - 1)`, reshaped to `[prediction, num_layers, top_k]` |

Every array is a [tensor envelope](#tensor-envelope).

### `/v1/chat/completions` fields

In this setup, seperate API calls does not necessarily share prefix, thus should each be considered as seperate `trajectories`.

Request:

| Field | Content |
|-------|---------|
| `model`, `messages`, `tools`, `tool_choice`, `temperature`, `top_p`, `stop`, `seed`, `max_completion_tokens` | OpenAI convention, supplied by the agent framework. |
| `chat_template_kwargs` | `--apply-chat-template-kwargs`, e.g. `enable_thinking`, `preserve_thinking`; |
| `stream: false` | every field below is unavailable when streaming |
| `logprobs: true` | gates `choices[i].meta_info.output_token_logprobs` |
| `return_prompt_token_ids: true` | gates `choices[i].meta_info.prompt_token_ids` |
| `top_logprobs` | `K` under `--use-score-centering K`; gates `choices[i].meta_info.output_top_logprobs` |
| `return_meta_info: true` | gates `choices[i].meta_info`, and so every field in it |
| `return_processor_outputs: true` | gates `choices[i].meta_info.processor_outputs` |
| `return_routed_experts`, `routed_experts_start_len` | expert indices, under `--use-rollout-routing-replay`; always `0`, since one call is one whole trajectory |

Response:

| Field | Location |
|-------|----------|
| generated message, finish reason | `choices[i].message` (`.content`, `.tool_calls`) and `choices[i].finish_reason`, OpenAI convention, consumed by the agent framework |
| input token ids | `choices[i].meta_info.prompt_token_ids`, int32 `[P]` |
| output token ids and logprobs | `choices[i].meta_info.output_token_logprobs` = `{"logprobs": float32 [T], "token_ids": int32 [T]}` |
| sampler top-$K$ | `choices[i].meta_info.output_top_logprobs` = `{"logprobs": float32 [T, K], "token_ids": int32 [T, K]}` |
| processor tensors | `choices[i].meta_info.processor_outputs` |
| expert indices | `choices[i].meta_info.routed_experts`, int32, one row per prediction over `[0, seqlen - 1)`, reshaped to `[prediction, num_layers, top_k]` |

Every array is a [tensor envelope](#tensor-envelope). The OpenAI `choices[i].logprobs` and
`choices[i].prompt_token_ids` stay empty: slim reads `meta_info` only.

Prompt positions carry no logprobs; `logprob_start_len` is fixed at `-1` and is not a request field.

### Tensor envelope

Processor tensors travel as JSON in both directions, and the per-token arrays of `meta_info` come
back the same way, so each one is a base64 envelope of its raw bytes:

```json
{"__tensor__": true, "dtype": "float32", "shape": [256, 1536], "data": "<base64>"}
```

Field names match what `AutoProcessor` returns (`pixel_values`, `pixel_values_videos`,
`input_features`, `image_grid_thw`, ...). `slim/utils/processing_utils.py` owns the codec for both
directions; `patch_dependencies.py` wires SGLang's call sites to it rather than restating it.

Upstream SGLang returns the logprobs as one JSON list per token, and the top-$K$ as $K$ of them.
Building and serializing those lists runs on the engine's HTTP process, and on a long generation
it blocks the process for longer than a health check waits. Streaming requests keep the upstream
lists.

### Router passthrough

The gateway deserializes each request into a typed struct and re-serializes it to the selected
worker, so any field the struct omits never reaches the engine. Responses are proxied as raw bytes.
slim installs a forked `sglang-router` wheel that declares `return_prompt_token_ids`,
`return_processor_outputs`, `return_routed_experts`, and `routed_experts_start_len` on both
endpoints, and additionally `return_meta_info` and `return_cached_tokens_details` on
`/v1/chat/completions`.

`return_processor_outputs` is slim's addition to SGLang, applied by `patch_dependencies.py`. Every
other field is native to SGLang.

`ChatCompletionRequest.input_ids` is deliberately absent from the fork: precomputed ids on the
OpenAI endpoint would put tokenization back in slim's hands there.

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
