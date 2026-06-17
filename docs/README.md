# slim Documentation

## Core Data Type: `Episode`

The `Episode` class (`slim.utils.types.Episode`) is the single data record that flows through the entire pipeline -- rollout generates it, reward scoring annotates it, and training consumes it.

### Lifecycle

1. **Created** from a dataset example via `Episode.from_example(example)` -- all sequence fields are Python lists.
2. **Mutated in-place** during async generation (tokens, loss_mask, rollout_log_probs are appended) and reward scoring (reward is set, generated_text is cached — the decoded loss_mask==1 tokens).
3. **Frozen** via `episode.freeze()` -- converts `tokens`, `loss_mask`, and `rollout_log_probs` to tensors.
4. **Consumed** by normalization, packing, and the training loop -- all tensor ops from this point.

### Key Fields

| Field | Pre-freeze type | Post-freeze type | Description |
|-------|----------------|-----------------|-------------|
| `example` | `dict` | -- | Raw dataset row. Generate and reward functions read whatever columns they need from this dict. |
| `tokens` | `list[int]` | `LongTensor` | Full token sequence (prompt + generation). |
| `loss_mask` | `list[int]` | `IntTensor` | **Edge-aligned** (length = `len(tokens) - 1`). `loss_mask[i]` indicates whether predicting `tokens[i+1]` contributes to loss. Prompt edges are `0`, generated edges are `1`. |
| `rollout_log_probs` | `list[float]` | `FloatTensor` | **Edge-aligned**. `rollout_log_probs[i]` is the log-probability of `tokens[i+1]` under the rollout policy. |
| `reward` | `float \| None` | `float \| None` | Scalar reward assigned by the reward model. |
| `generated_text` | `str \| None` | `str \| None` | Decoded text of loss_mask==1 tokens, cached during RM scoring. |
| `status` | `str` | -- | One of `PENDING`, `COMPLETED`, `TRUNCATED`, `ABORTED`, `FAILED`. |

### Dataset Columns and `episode.example`

`Episode.from_example(row)` stores the entire dataset row as-is in `episode.example`. The default rollout and reward paths (sglang_rollout, sft_rollout, rm_hub) expect these columns by convention:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `prompt` | `str \| list[dict]` | yes | Raw string or chat-format messages. |
| `label` | `str \| None` | no | Ground-truth label for built-in reward functions. |
| `tools` | `list[dict] \| None` | no | Tool/function definitions for chat template. |
| `images` / `videos` / `audios` | `list \| None` | no | Top-level multimodal fields, forwarded to the processor. |
| `metadata` | `dict \| None` | no | Passed through to reward functions (e.g. `rm_type`). |

Custom generate/reward functions can read any column from `episode.example` — there is no fixed schema. For example, a BAP reward function reads `episode.example["ground_truth"]` instead of `"label"`.

### Edge Alignment

Sequence-level fields (`loss_mask`, `rollout_log_probs`) are **edge-aligned**: they have length `len(tokens) - 1` because each entry describes the transition *from* `tokens[i]` *to* `tokens[i+1]`. Call `episode.ensure_edge_alignment()` before `freeze()` to validate (or materialize a default all-ones mask).

## [Rollout](rollout/README.md)
SGLang setup, parameter pass-through, rollout args, dynamic sampling, partial rollout, evaluation.
- [SGLang Config](rollout/sglang-config.md) -- Multi-model serving, PD disaggregation, YAML deployment
- [Speculative Decoding](rollout/speculative-decoding.md) -- MTP draft model acceleration
- [Fault Tolerance](rollout/fault-tolerance.md) -- Heartbeat-based recovery
- [PD Disaggregation](rollout/pd-disaggregation.md) -- Prefill-decode separation
- [Low Precision Inference](rollout/low-precision.md) -- FP8 rollout

## [Training](training/README.md)
Installation, GPU allocation, checkpoints, data format, RL algorithms (GRPO/PPO), multi-node, FAQ.
- [Reproducibility](training/reproducibility.md) -- Deterministic bitwise training
- [Debugging](training/debug.md) -- Precision alignment, separate debugging
- [Profiling](training/profiling.md) -- Rollout performance analysis
- [CI](training/ci.md) -- GitHub Actions workflow

## [Customization](customization/README.md)
All extension points: rollout functions, reward models, filters, loss functions, logging, multi-turn/agentic adaptation.
