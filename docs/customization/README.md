# Customization Guide

slim provides extensive customization capabilities through function path arguments. These allow you to inject custom logic at various stages of the training and rollout pipeline without modifying the core codebase.

## Overview of Customization Interfaces

Below is a summary of all available customization interfaces and their purposes.

| Interface Argument | Purpose |
| :--- | :--- |
| [`--rollout-function-path`](#1-rollout-function---rollout-function-path) | Override the entire rollout generation logic. |
| [`--custom-generate-function-path`](#2-custom-generate-function---custom-generate-function-path) | Override only the generation step (e.g., for RAG or tool use). |
| [`--custom-rm-path`](#3-reward-model---custom-rm-path) | Implement custom reward computation logic. |
| [`--dynamic-sampling-filter-path`](#4-dynamic-sampling-filter---dynamic-sampling-filter-path) | Filter episodes during dynamic sampling (e.g., DAPO). |
| [`--rollout-sample-filter-path`](#5-rollout-sample-filter---rollout-sample-filter-path) | Determine if individual episodes participate in loss calculation. |
| [`--rollout-all-samples-process-path`](#6-rollout-all-samples-process---rollout-all-samples-process-path) | Process all episodes (including filtered ones) after rollout. |
| [`--rollout-data-postprocess-path`](#7-rollout-data-postprocess---rollout-data-postprocess-path) | Post-process rollout data after log probs are computed. |
| [`--custom-loss-function-path`](#8-custom-loss-function---custom-loss-function-path) | Implement custom training loss computation. |
| [`--custom-tis-function-path`](#9-custom-tisrs-function---custom-tis-function-path) | Implement custom importance sampling for off-policy correction. |
| [`--custom-pg-loss-reducer-function-path`](#10-custom-pg-loss-reducer---custom-pg-loss-reducer-function-path) | Customize pg_loss reduction (e.g., for Dr.GRPO). |
| [`--custom-reward-post-process-path`](#11-reward-post-processing---custom-reward-post-process-path) | Custom post-processing of rewards before advantage computation. |
| [`--custom-rollout-log-function-path`](#12-logging-functions) | Custom logging for training rollouts. |
| [`--custom-eval-rollout-log-function-path`](#12-logging-functions) | Custom logging for evaluation rollouts. |
| [`--data-source-path`](#13-data-source---data-source-path) | Override the data source for rollout prompts. |
| [`--eval-function-path`](#14-evaluation-function---eval-function-path) | Override the rollout function specifically for evaluation. |

## Detailed Interface Reference

### 1. Rollout Function (`--rollout-function-path`)

**Default**: `slim.rollout.sglang_rollout.generate_rollout`

**Purpose**: Override the entire rollout generation logic.

**Signature**:
```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**Use Cases**:
- Implementing complex multi-turn conversations
- Adding custom sampling strategies
- Integrating external tools or APIs during generation


---

### 2. Custom Generate Function (`--custom-generate-function-path`)

**Default**: `None` (uses built-in generate function)

**Purpose**: Override only the generation step within the default rollout function.

**Signature**:
```python
async def custom_generate(args, episode: Episode, sampling_params: dict) -> Episode
```

An optional `evaluation` keyword argument is also supported:
```python
async def custom_generate(args, episode: Episode, sampling_params: dict, evaluation: bool = False) -> Episode
```

**Use Cases**:
- Implementing tool-calling or function-calling capabilities
- Adding retrieval-augmented generation (RAG)
- Multi-turn conversation handling


---

### 3. Reward Model (`--custom-rm-path`)

**Default**: `None` (uses built-in reward models based on `--rm-type`)

**Purpose**: Implement custom reward computation logic.

**Signature** (single episode mode):
```python
async def custom_rm(args, episode: Episode, **kwargs) -> float
```

**Signature** (batch mode, when `--group-rm` is enabled):
```python
async def batched_custom_rm(args, episodes: list[Episode], **kwargs) -> list[float]
```

**Use Cases**:
- Custom rule-based rewards
- Integration with external reward model services
- Multi-dimensional reward signals

**Built-in Options** (`--rm-type`):
- `math`: Mathematical answer verification
- `dapo`: DAPO-style scoring
- `deepscaler`: DeepScaler rule-based reward
- `f1`: F1 score computation
- `gpqa`: GPQA reward computation
- `ifbench`: IFBench reward computation
- `remote_rm`: Remote reward model service (requires `--rm-url`)

Additionally, any `--rm-type` value can be prefixed with `boxed_` (e.g. `boxed_math`) to first extract a boxed answer from the response before applying the reward function.

---

### 4. Dynamic Sampling Filter (`--dynamic-sampling-filter-path`)

**Default**: `None`

**Purpose**: Filter episodes during dynamic sampling (e.g., DAPO-style filtering).

**Signature**:
```python
def filter_function(args, episodes: list[Episode], **kwargs) -> DynamicFilterOutput
```

**Return Type**:
```python
@dataclass
class DynamicFilterOutput:
    keep: bool  # Whether to keep this episode group
    reason: str | None  # Reason for filtering (for logging)
```

**Use Cases**:
- Filtering out groups where all responses have the same reward
- Implementing curriculum learning strategies
- Quality-based group selection

**Example**: `slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`

---

### 5. Rollout Sample Filter (`--rollout-sample-filter-path`)

**Default**: `None`

**Purpose**: Determine whether individual episodes participate in loss calculation.

**Signature**:
```python
def filter_function(args, groups: list[RolloutGroup]) -> None
```

Where `RolloutGroup` has an `episodes: list[Episode]` attribute. The function should modify groups in-place (e.g., setting episode loss masks to all zeros to exclude them from training).

**Use Cases**:
- Filtering episodes based on response quality
- Implementing selective training strategies

---

### 6. Rollout All Samples Process (`--rollout-all-samples-process-path`)

**Default**: `None`

**Purpose**: Process all episodes (including filtered ones) after rollout.

**Signature**:
```python
def process_function(args, all_groups: list[list[Episode]], get_examples) -> None
```

**Note**: The third argument is the `get_examples` callable from the data source, not the data source object itself.

**Use Cases**:
- Logging and analysis of all generated episodes
- Computing statistics across filtered and kept episodes

---

### 7. Rollout Data Postprocess (`--rollout-data-postprocess-path`)

**Default**: `None`

**Purpose**: Post-process rollout data after log probabilities are computed.

**Signature**:
```python
def postprocess_function(args) -> None
```

**Use Cases**:
- Updating loss masks based on computed values
- Adding additional metadata to episodes

---

### 8. Custom Loss Function (`--custom-loss-function-path`)

**Default**: `None` (requires `--loss-type custom_loss`)

**Purpose**: Implement custom training loss computation.

**Use Cases**:
- Novel RL objectives
- Multi-objective optimization
- Custom regularization terms

---

### 9. Custom TIS/RS Function (`--custom-tis-function-path`)

**Default**: `None`

**Purpose**: Implement custom importance sampling for off-policy correction.

**Use Cases**:
- Custom importance sampling ratio computation
- Advanced off-policy correction methods

**Example**: Implement a function matching the TIS signature for custom importance sampling.

---

### 10. Custom pg_loss Reducer (`--custom-pg-loss-reducer-function-path`)

**Default**: `None`

**Purpose**: Customize the reduction of pg_loss while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean.

**Signature**:
```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**Use Cases**:
- Dr.GRPO: Divide by a constant instead of effective token count
- Custom loss normalization strategies

**Example**: Implement a `get_pg_loss_reducer` function for Dr.GRPO-style normalization.

---

### 11. Reward Post-Processing (`--custom-reward-post-process-path`)

**Default**: `None` (uses default GRPO normalization)

**Purpose**: Custom post-processing of rewards before advantage computation.

**Signature**:
```python
def reward_post_process(args, episodes: list[Episode]) -> None
```

The function modifies episode rewards in-place.

**Use Cases**:
- Custom reward normalization strategies
- Reward shaping

---

### 12. Logging Functions

#### Training Rollout Logging (`--custom-rollout-log-function-path`)

**Signature**:
```python
def log_rollout_data(rollout_id, args, episodes, rollout_extra_metrics, rollout_time) -> bool
```

**Return**: `True` to skip default logging, `False` to continue with default logging.

#### Evaluation Rollout Logging (`--custom-eval-rollout-log-function-path`)

**Signature**:
```python
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool
```

**Return**: `True` to skip default logging, `False` to continue with default logging.

---

### 13. Data Source (`--data-source-path`)

**Default**: `slim.rollout.data_source.RolloutDataSource`

**Purpose**: Override the data source for rollout prompts.

**Base Class**: `slim.rollout.data_source.DataSource`

**Required Methods**:
```python
class CustomDataSource(DataSource):
    def get_examples(self, num_prompts: int) -> list[dict]:
        """Return num_prompts raw dataset examples (dicts)."""

    def add_examples(self, examples: list[dict]):
        """Re-queue examples (e.g. aborted prompts) back into the source."""

    def save(self, rollout_id):
        """Save state for checkpointing."""

    def load(self, rollout_id=None):
        """Load state from checkpoint."""

    def __len__(self) -> int:
        """Length of the data source. May change when examples are added/fetched."""
```

---

### 14. Evaluation Function (`--eval-function-path`)

**Default**: Same as `--rollout-function-path`

**Purpose**: Override the rollout function specifically for evaluation.

**Use Cases**:
- Different sampling parameters for evaluation
- Evaluation-specific logic

---

## Testing Custom Function Paths

slim also provides CPU-only contract tests for customization interfaces. These tests resolve components through import-path strings, so they can validate both built-in hooks and user-defined implementations passed through the same CLI arguments used by training.

The tests live under `tests/plugin_contracts/` and are grouped by hook shape:

- `tests/plugin_contracts/test_plugin_rollout_contracts.py`
  Covers `--rollout-function-path`
- `tests/plugin_contracts/test_plugin_generate_contracts.py`
  Covers `--custom-generate-function-path`
- `tests/plugin_contracts/test_plugin_path_loading_contracts.py`
  Covers `--eval-function-path`, `--custom-rm-path`, `--dynamic-sampling-filter-path`, `--data-source-path`, `--rollout-sample-filter-path`, and `--rollout-all-samples-process-path`
- `tests/plugin_contracts/test_plugin_runtime_hook_contracts.py`
  Covers `--custom-rollout-log-function-path`, `--custom-eval-rollout-log-function-path`, `--custom-reward-post-process-path`, and `--rollout-data-postprocess-path`

Run all customization contract tests locally:

```bash
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```

Each test file can also be executed directly with `python tests/plugin_contracts/<file>.py`, which keeps them compatible with `run-ci-changed`.

A dedicated `run-ci-plugin-contracts` CI label is also available. Adding it to a PR triggers all four contract test files in parallel (no GPU required).

For user-defined implementations, you can either export environment variables such as `SLIME_CONTRACT_ROLLOUT_FUNCTION_PATH` and `SLIME_CONTRACT_CUSTOM_RM_PATH`, or pass overrides directly when running a test file, for example:

```bash
python tests/plugin_contracts/test_plugin_rollout_contracts.py \
  --rollout-function-path my_project.custom_rollout.generate_rollout
```

To validate your own custom implementation, replace the plugin paths used in these tests with your module path and keep the same assertions on signatures, return structure, and side effects.

---

## Multi-Turn / Agentic Adaptation Guide

slim supports complex agent scenarios (multi-turn interaction, tool calling) by overriding the default rollout and reward logic through custom functions.

### Three Steps

1. **Data Preparation**: Map conversation history, labels, and metadata to the supported `Episode` fields (`prompt`, `label`, `metadata`, `tools`). Keep dataset top-level columns within the standard finite schema and store extra task-specific fields inside `metadata`.

2. **Custom Generation Function** (`--custom-generate-function-path`):
   ```python
   async def generate(args, episode: Episode, sampling_params) -> Episode:
   ```
   Implement the interaction loop: model generates action -> execute tool -> append observation -> repeat.

3. **Custom Reward Function** (`--custom-rm-path`):
   ```python
   async def reward_func(args, episode: Episode, **kwargs) -> float:
   ```

### Loss Masking

For multi-turn training, `loss_mask` controls which tokens contribute to loss:
- **Model-generated** tokens (thinking, actions) -> `loss_mask = 1`
- **Tool/environment** tokens (API results) -> `loss_mask = 0`

### Generation Pseudocode

```python
async def generate(args, episode: Episode, sampling_params) -> Episode:
    # episode.tokens starts with tokenized prompt; episode.loss_mask has 0s for prompt edges

    for _ in range(max_turns):
        model_output = await call_sglang(episode.tokens, ...)
        episode.tokens.extend(model_tokens)
        episode.loss_mask.extend([1] * len(model_tokens))

        action, content = parse_action(model_output)
        if action == "search":
            tool_output = await google_search(content)
            episode.tokens.extend(tool_tokens)
            episode.loss_mask.extend([0] * len(tool_tokens))
        elif action == "answer":
            break

    episode.status = Episode.Status.COMPLETED
    return episode
```

### Configuration

```bash
CUSTOM_ARGS=(
    --custom-generate-function-path your_module.generate
    --custom-rm-path your_module.reward_func
)
```

