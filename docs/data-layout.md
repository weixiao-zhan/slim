# Data Layout

## Episode

`Episode` (defined in `slim/utils/types.py`) is the core data record flowing through rollout, reward, and training. Each episode represents a single prompt-response sequence.

An episode is created from a dataset row via `Episode.from_example(example)`, which stores a shallow copy of the raw dict in `episode.example`. All other fields start at their defaults (empty list for `tokens`, `None` for masks, etc.).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `example` | `dict` | Raw dataset row; rollout/RM functions read whatever columns they need |
| `generate_function_path` | `str \| None` | Override path to a custom generate function for this episode |
| `session_id` | `str \| None` | UUID for consistent-hashing router affinity |
| `tokens` | `list[int]` | Full token sequence (prompt + generated) |
| `loss_mask` | `list[int] \| None` | Edge-aligned; 0 for prompt edges, 1 for generated edges |
| `rollout_log_probs` | `list[float] \| None` | Edge-aligned log-probabilities under the rollout policy |
| `rollout_routed_experts` | `np.ndarray [num_edges, num_layers, top_k] \| None` | MoE expert indices recorded during rollout for replay in training |
| `multimodal_inputs` | `dict[str, Tensor] \| None` | Non-token-aligned processor outputs (pixel_values, image_grid_thw, etc.) |
| `reward` | `float \| None` | Raw scalar reward assigned by the reward model |
| `rollout_index` | `int \| None` | Stable position in the complete rollout batch |
| `advantages` | edge-aligned values or `None` | Policy training targets |
| `values` | edge-aligned values or `None` | Critic predictions used to construct PPO targets |
| `value_targets` | edge-aligned values or `None` | Critic regression targets |
| `text` | `str \| None` | Decoded full sequence (prompt + response) |
| `generated_text` | `str \| None` | Decoded response region only (loss_mask==1 tokens) |
| `non_generation_time` | `float` | Wall-clock time spent outside token generation |
| `max_tokens` | `int` | Maximum context length for this episode |
| `_sampling_params` | `dict \| None` | Transient rollout request params; cleared by `freeze()` |
| `status` | `str` | One of `PENDING`, `COMPLETED`, `TRUNCATED`, `ABORTED`, `FAILED` |

## Dataset Columns

The default rollout reads these keys from `episode.example`:

| Key | Type | Purpose |
|-----|------|---------|
| `prompt` | `list[dict]` (chat messages) or `str` | Input to the model. Chat format is passed through `apply_chat_template` with `add_generation_prompt=True`. |
| `label` | any | Ground-truth label for reward functions. Not consumed by rollout itself. |
| `tools` | `list[dict]` or `None` | Tool definitions passed to `apply_chat_template(tools=...)`. |
| `images` | `list` or `None` | Images for VLM; passed to the processor alongside the rendered prompt. |
| `videos` | `list` or `None` | Videos for VLM; passed to the processor. |
| `audios` | `list` or `None` | Audio for VLM; passed to the processor. |
| `metadata` | `dict` or `None` | Arbitrary metadata; eval datasets inject per-dataset config here. |

Custom columns are free-form. Custom generate functions and reward models access them via `episode.example[key]`.

## Token-in, Token-out

Generation is append-only: the generate function tokenizes the prompt into `episode.tokens`, then extends `tokens`, `loss_mask`, and `rollout_log_probs` as new tokens arrive.
This design keeps the episode as a single growing sequence — no separate prompt/response buffers, no index bookkeeping.

### Edge Alignment

`loss_mask`, `rollout_log_probs`, `rollout_routed_experts`, `advantages`, `values`, and `value_targets` are **edge-aligned**: length is `len(tokens) - 1`.
Entry `i` describes the prediction of `tokens[i+1]` given `tokens[:i+1]`.
Prompt edges are 0 in `loss_mask`; generated edges are 1.

`episode.ensure_edge_alignment()` materializes and validates rollout-produced edge fields before freeze.
`episode.set_train_targets()` validates and assigns the training targets after reward and value processing.

### Processor Output Format (VLM)

For VLMs, the HuggingFace processor converts raw media (images/videos) into dense tensors (`pixel_values`, `image_grid_thw`, etc.) and expands the token sequence with vision placeholder tokens.
slim sends these processor outputs directly to sglang rather than raw media.

This avoids soft-token drift: if sglang re-ran the processor internally, floating-point differences in vision encoding could produce different embeddings than what training sees, causing routing-replay mismatch.
The tradeoff is payload size (processor output tensors are much larger than compressed PNG/JPEG), so they are packed as base64 binary envelopes rather than serialized as JSON arrays.

The tensors are stored in `episode.multimodal_inputs` and sent to sglang inside the `image_data` field with `"format": "processor_output"`.

### Routing Replay

For MoE models, expert routing decisions made during rollout are recorded in `episode.rollout_routed_experts` and replayed during the training forward pass.
This ensures the same experts are activated in both passes; without replay, stochastic top-k selection would cause a train/inference mismatch in which tokens flow through which experts.

The gather operation (`gather_replayed_topk`) is differentiable: router weights still receive gradient, only the expert *choice* is frozen.
The buffer stays active across both the forward pass and gradient-checkpointing recomputation during backward.

## Freeze

After generation and reward assignment, `episode.ensure_edge_alignment()` validates lengths and `episode.freeze()` converts sequence fields to tensors:

- `tokens` → `torch.long`
- `loss_mask` → `torch.int`
- `rollout_log_probs` → `torch.float32`
- `rollout_routed_experts` → `torch.int32` (via `torch.from_numpy`)
- `_sampling_params` → cleared

After freeze, episodes are consumed by:

1. **Target preparation**: `AdvantageEstimator` restores global rollout order, computes group-relative GRPO or GSPO advantages without changing rewards, or combines PPO critic values with rewards to produce training targets.
2. **Role-local packing**: actor and critic independently call `pack_sequences` (`slim/backends/nemo/data_packing.py`). Packs contain `cu_seqlens`, per-sequence `position_ids`, and cumulative edge offsets for `loss_masks`, `rollout_log_probs`, `advantages`, `values`, `value_targets`, and routing fields.
3. **Training loop**: `unpack_sequences` slices token fields by `cu_seqlens` and edge fields by cumulative edge offsets.

Actor or critic precompute may create packs before targets are available. `update_packed_targets()` attaches `advantages`, `old_values`, and `value_targets` to those cached role-local packs before training.

---

**See also:** [SGLang Config](sglang-config.md) | [Training Loss](training-loss.md) | [Customization](customization.md)
