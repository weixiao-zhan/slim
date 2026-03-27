# minislime Documentation

## Core Data Type: `Episode`

The `Episode` class (`slime.utils.types.Episode`) is the single data record that flows through the entire pipeline -- rollout generates it, reward scoring annotates it, and training consumes it.

### Lifecycle

1. **Created** from a dataset example via `Episode.from_example(example)` -- all sequence fields are Python lists.
2. **Mutated in-place** during async generation (tokens, loss_mask, rollout_log_probs are appended) and reward scoring (reward is set).
3. **Frozen** via `episode.freeze()` -- converts `tokens`, `loss_mask`, and `rollout_log_probs` to tensors.
4. **Consumed** by normalization, packing, and the training loop -- all tensor ops from this point.

### Key Fields

| Field | Pre-freeze type | Post-freeze type | Description |
|-------|----------------|-----------------|-------------|
| `tokens` | `list[int]` | `LongTensor` | Full token sequence (prompt + generation). |
| `loss_mask` | `list[int]` | `IntTensor` | **Edge-aligned** (length = `len(tokens) - 1`). `loss_mask[i]` indicates whether predicting `tokens[i+1]` contributes to loss. Prompt edges are `0`, generated edges are `1`. |
| `rollout_log_probs` | `list[float]` | `FloatTensor` | **Edge-aligned**. `rollout_log_probs[i]` is the log-probability of `tokens[i+1]` under the rollout policy. |
| `reward` | `float \| None` | `float \| None` | Scalar reward assigned by the reward model. |
| `prompt` | `str \| list[dict]` | -- | Original prompt (string or chat messages). |
| `label` | `str \| None` | -- | Optional ground-truth label for reward functions. |
| `status` | `str` | -- | One of `PENDING`, `COMPLETED`, `TRUNCATED`, `ABORTED`, `FAILED`. |

### Edge Alignment

Sequence-level fields (`loss_mask`, `rollout_log_probs`) are **edge-aligned**: they have length `len(tokens) - 1` because each entry describes the transition *from* `tokens[i]` *to* `tokens[i+1]`. Call `episode.ensure_edge_alignment()` before `freeze()` to validate (or materialize a default all-ones mask).

## [Rollout](rollout/README.md)
SGLang setup, parameter pass-through, rollout args, dynamic sampling, partial rollout, evaluation.
- [SGLang Config](rollout/sglang-config.md) -- Multi-model serving, PD disaggregation, YAML deployment
- [Slime Router](rollout/slime-router.md) -- Training-oriented HTTP router
- [Speculative Decoding](rollout/speculative-decoding.md) -- MTP draft model acceleration
- [On-Policy Distillation](rollout/on-policy-distillation.md) -- Teacher-student distillation
- [Fault Tolerance](rollout/fault-tolerance.md) -- Heartbeat-based recovery
- [PD Disaggregation](rollout/pd-disaggregation.md) -- Prefill-decode separation

## [Training](training/README.md)
Installation, GPU allocation, checkpoints, data format, RL algorithms (GRPO/PPO), multi-node, FAQ.
- [Low Precision](training/low-precision.md) -- FP8 inference, INT4 QAT
- [Reproducibility](training/reproducibility.md) -- Deterministic bitwise training
- [Debugging](training/debug.md) -- Precision alignment, separate debugging
- [Profiling](training/profiling.md) -- Rollout performance analysis
- [CI](training/ci.md) -- GitHub Actions workflow

## [Customization](customization/README.md)
All extension points: rollout functions, reward models, filters, loss functions, logging, multi-turn/agentic adaptation.

## [Examples](examples/README.md)
- [Qwen3-30B-A3B (MoE)](examples/qwen3-30B-A3B.md)
- [GLM-4.7-Flash (MoE + MTP)](examples/glm4.7-30B-A3B.md)
- [Qwen3-4B SFT](examples/qwen3-4b-base-openhermes.md)

Also see runnable [examples/](../examples/) for VLM, search, and tool-use workflows.
