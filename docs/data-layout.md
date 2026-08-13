# Data Layout

slim uses a two-level hierarchy. An **`Episode`** is one complete experiment: a full
problem-solving attempt from prompt to final answer. A **`Trajectory`** is one contiguous
generation span by the policy, and an episode holds one or more.

Both live in `slim/utils/types.py`.

## Episode

An episode is created from a dataset row via `Episode.from_example(example)`, which stores a
shallow copy of the raw dict in `episode.example` and starts the episode with one empty
trajectory.

| Field | Type | Description |
|-------|------|-------------|
| `example` | `dict` | Raw dataset row; rollout/RM functions read whatever columns they need |
| `trajectories` | `list[Trajectory]` | The attempt's generation spans |
| `reward` | `float \| None` | Shorthand for one scalar scoring the whole attempt |
| `episode_index` | `int \| None` | Stable position in the complete rollout batch |
| `group_index` | `int \| None` | Prompt group this attempt belongs to |
| `generate_function_path` | `str \| None` | Override path to a custom generate function for this episode |
| `session_id` | `str \| None` | UUID for consistent-hashing router affinity |
| `max_tokens` | `int` | Maximum context length for this episode |
| `non_generation_time` | `float` | Wall-clock time spent outside token generation |
| `_sampling_params` | `dict \| None` | Transient rollout request params; cleared during source-token finalization |
| `status` | `str` | One of `PENDING`, `COMPLETED`, `TRUNCATED`, `ABORTED`, `FAILED` |

`status` is a property of the attempt. It answers episode-level questions: whether the
attempt may still be generated into, whether it is eligible for reward and decoding,
whether its group is complete, and what fraction of attempts hit the context limit. The
generate loop folds each generation call's finish reason into it through
`update_status_from_finish_reason`, so an episode whose third tool call is truncated is a
truncated attempt.

`Episode.trajectory` returns the sole trajectory and raises when there is more than one, so
the append-only rollout path reads naturally. `Episode.token_count` sums tokens over spans,
and `Episode.get_reward_value()` returns one scalar whichever level carries the reward.

## Trajectory

| Field | Type | Description |
|-------|------|-------------|
| `token_ids` | `list[int]` or `LongTensor [T]` | Full token sequence of this span (prompt + generated) |
| `loss_mask` | `list[int]` or `IntTensor [T]` | Source-token mask; 0 for prompt and terminal positions, 1 for generated predictions |
| `rollout_log_probs` | `list[float]` or `FloatTensor [T]` | Source-token-aligned log-probabilities under the rollout policy |
| `rollout_routed_experts` | `IntTensor [T, num_layers, top_k] \| None` | Source-token-aligned MoE expert indices recorded during rollout |
| `multimodal_inputs` | `dict[str, Tensor] \| None` | Non-token-aligned processor outputs (pixel_values, image_grid_thw, etc.) |
| `reward` | `float \| None` | Raw scalar reward for this span |
| `text` | `str \| None` | Decoded full span (prompt + response) |
| `generated_text` | `str \| None` | Decoded target tokens selected by active prediction slots |
| `advantages` | `FloatTensor [T] \| None` | Source-token-aligned policy training targets |
| `values` | `FloatTensor [T] \| None` | Source-token-aligned critic predictions used to construct PPO targets |
| `value_targets` | `FloatTensor [T] \| None` | Source-token-aligned critic regression targets |
| `episode_index` | `int \| None` | The attempt this span belongs to; `None` marks a padding span |
| `group_index` | `int \| None` | The prompt group this span belongs to |
| `loss_weight` | `float` | Per-document loss weight $w_d$ |

Trajectories carry no ordering or dependency relation. An episode is a *set* of generation
spans belonging to one attempt: a sub-agent dispatch fans out, a compression step replaces
context rather than extending it, and nothing in the layout says which span precedes which.

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

Custom columns are free-form. Custom generate functions and reward models access them via
`episode.example[key]`. A generate function that needs per-span detail, such as which of
five sub-agent branches aborted, keeps that in `episode.example` where its own reward
function reads it.

## Token-in, Token-out

Generation within one trajectory is append-only: the generate function tokenizes the prompt
into `trajectory.token_ids`, then extends `token_ids`, `loss_mask`, and `rollout_log_probs`
as new tokens arrive. Appending a new trajectory is how a rollout continues past a span
boundary.

### Source-Token Alignment

During generation, `loss_mask`, `rollout_log_probs`, and `rollout_routed_experts` are
temporary Python or NumPy values with length `len(token_ids) - 1`.
`trajectory.finalize_source_token_alignment()` appends one neutral terminal slot and
converts the sequence fields to CPU tensors with length `len(token_ids)`.
`episode.finalize_source_token_alignment()` does this for every span and clears
`_sampling_params`.

After finalization, entry `i` describes the prediction of `token_ids[i+1]` given
`token_ids[:i+1]`. The final source position has no prediction target, so its mask and
numeric training fields are zero. `trajectory.set_train_targets()` accepts only
source-token-aligned targets with this complete token length.

Finalization is a one-shot boundary per trajectory.

### Processor Output Format (VLM)

For VLMs, the HuggingFace processor converts raw media (images/videos) into dense tensors
(`pixel_values`, `image_grid_thw`, etc.) and expands the token sequence with vision
placeholder tokens. slim sends these processor outputs directly to sglang rather than raw
media.

This avoids soft-token drift: if sglang re-ran the processor internally, floating-point
differences in vision encoding could produce different embeddings than what training sees,
causing routing-replay mismatch. The tradeoff is payload size (processor output tensors are
much larger than compressed PNG/JPEG), so they are packed as base64 binary envelopes rather
than serialized as JSON arrays.

The tensors are stored in `trajectory.multimodal_inputs`. When slim owns tokenization it sends them
to sglang inside the `image_data` field with `"format": "processor_output"`; when the engine owns
tokenization it returns them the same way. See
[Tokenization Ownership](sglang-config.md#tokenization-ownership) for the field names on each
endpoint.

### Routing Replay

For MoE models, expert routing decisions made during rollout are recorded in
`trajectory.rollout_routed_experts` and replayed during the training forward pass. Routing
data is owned per span, so a second generation span cannot clobber the first's.

This ensures the same experts are activated in both passes; without replay, stochastic top-k
selection would cause a train/inference mismatch in which tokens flow through which experts.

The gather operation (`gather_replayed_topk`) is differentiable: router weights still receive
gradient, only the expert *choice* is frozen. The buffer stays active across both the forward
pass and gradient-checkpointing recomputation during backward.

## Flatten, Pad, Partition

A `Trajectory` *is* a packed document: its tokens form a contiguous block delimited by
`cu_seqlens`, and every reduction keyed on `document_ids` addresses it individually. The
rollout/train boundary therefore flattens episodes into trajectories and hands each DP rank
a `TrajectoryBatch` (`slim/utils/trajectory_batch.py`).

**Step count in episodes, work distribution by trajectory.** `global_batch_size` is
denominated in episodes, so the number of optimizer steps per rollout does not depend on how
many generation calls the rollout made and the LR decay horizon stays
`num_rollout * rollout_batch_size * n_samples_per_prompt // global_batch_size`. The unit of
*work* is the trajectory: an episode with 50 spans is 50 documents, 50 attention blocks, and
potentially 50 micro-batches, so distributing spans is what equalizes document counts per
rank.

`build_dp_batches` runs four steps:

1. **Flatten.** Every span is stamped with its episode's `episode_index`, `group_index`, and
   `loss_weight`. An `Episode.reward` is broadcast onto every span of the attempt and
   cleared; setting both levels is an error.
2. **Pad.** The flattened list is padded up to the next multiple of `dp_size * num_steps`
   with inert two-token spans: all-zero `loss_mask`, `loss_weight=0.0`, and no
   `episode_index`. Two tokens is the minimum `pack_sequences` accepts. The pad is bounded
   by `dp_size * num_steps - 1` spans per rollout.
3. **Partition across optimizer steps** with `get_seqlen_balanced_partitions(..., equal_size=True)`.
4. **Partition each step across ranks** the same way.

Every rank holds the same number of trajectories per step with balanced token sums, so pack
counts are equal by construction in fixed micro-batch mode and always reachable by splitting
in dynamic mode.

An episode's spans can land on different ranks and in different optimizer steps, so one
episode's gradient is split across steps and the episode-level baseline is zero-mean across a
rollout rather than within a step. `--balance-data` already makes this tradeoff for prompt
groups.

The upstream trim is group-aware and episode-denominated, truncating whole prompt groups so
group boundaries survive and `num_episodes % global_batch_size == 0`.

## Consumption

After the split, batches are consumed by:

1. **Target preparation**: `AdvantageEstimator` regroups spans by `episode_index`, computes
   group-relative GRPO or GSPO advantages without changing rewards, or combines PPO critic
   values with rewards to produce training targets. Baselines see an attempt as a unit.
2. **Role-local packing**: actor and critic independently call `pack_sequences`
   (`slim/backends/nemo/data_packing.py`). Every sequence field is concatenated using the
   same token boundaries recorded in `cu_seqlens`, and each pack carries the per-document
   `loss_weights`, `response_lengths`, and `reward` lists.
3. **Training loop**: `unpack_sequences` slices every sequence field directly by those token
   boundaries.

Actor or critic precompute may create packs before targets are available.
`update_packed_targets()` attaches `advantages`, `old_values`, and `value_targets` to those
cached role-local packs before training.

---

**See also:** [SGLang Config](sglang-config.md) | [Training Loss](training-loss.md) | [Customization](customization.md)
