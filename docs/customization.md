# Customization

slim exposes every major pipeline stage as a pluggable `--*-path` CLI argument.
Each accepts a dotted Python path (e.g. `my_package.module.function`) loaded at runtime.

## Extension Points

| CLI Argument | Default | Purpose |
|---|---|---|
| `--rollout-function-path` | `slim.rollout.sglang_rollout.generate_rollout` | Top-level rollout orchestration |
| `--custom-generate-function-path` | None | Per-episode generation logic within the default rollout |
| `--custom-rm-path` | None | Reward model scoring |
| `--rollout-group-filter-path` | None | Dynamic group-level filter (e.g. DAPO) |
| `--rollout-sample-filter-path` | None | Sample-level masking after rollout |
| `--rollout-all-samples-process-path` | None | Post-rollout hook over all groups (including filtered) |
| `--custom-loss-function-path` | None | Custom training loss (requires `--loss-type custom_loss`) |
| `--custom-mismatch-correction-function-path` | None | Importance-weight correction for train/rollout mismatch |
| `--custom-reward-post-process-path` | None | Reward shaping before advantage estimation |
| `--custom-rollout-log-function-path` | None | Custom logging for train rollout |
| `--custom-eval-rollout-log-function-path` | None | Custom logging for eval rollout |
| `--data-source-path` | `slim.rollout.data_source.RolloutDataSource` | Dataset iteration and state persistence |
| `--eval-function-path` | (same as rollout-function-path) | Evaluation rollout orchestration |

## Key Interfaces

### Rollout function (`--rollout-function-path`)

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

Replaces the entire rollout loop.
Override this for fundamentally different sampling strategies (tree search, offline data, non-SGLang backend).

### Generate function (`--custom-generate-function-path`)

```python
async def custom_generate(state: GenerateState, episode: Episode) -> Episode
```

Substitutes only the per-episode generation logic.
The episode arrives holding one trajectory with prompt tokens already set; the function appends generated tokens to `episode.trajectory`, sets its `loss_mask`, and returns the completed episode.
An agentic workload appends further `Trajectory` objects to `episode.trajectories`, one per contiguous generation span, and sets `episode.status` for the attempt as a whole.

Per-episode override: set `episode.generate_function_path` to route specific episodes to a different generate function.

### Reward model (`--custom-rm-path`)

```python
async def custom_rm(args, episode: Episode, **kwargs) -> float

# Batch mode (--group-rm):
async def custom_rm(args, episodes: list[Episode], **kwargs) -> list[float]
```

The returned scalar lands on `episode.reward` and is broadcast to every span of the attempt.
A reward function that wants to score spans individually writes `trajectory.reward` on each span instead and returns nothing for the episode; setting both levels is an error.
Each span carries its own `generated_text`, so span boundaries reach the reward function rather than being flattened into one decode. Join them when the flat view is what you want.

Built-in `--rm-type` options (used when `--custom-rm-path` is not set): `math`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `remote_rm`, `random`.

### Group filter (`--rollout-group-filter-path`)

```python
def group_filter(args, episodes: list[Episode], **kwargs) -> DynamicFilterOutput
```

Returns `DynamicFilterOutput(keep=bool, reason=str|None)`.
If `keep=False`, the group is discarded and a new prompt is sampled.
Built-in example: `slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.

### Sample filter (`--rollout-sample-filter-path`)

```python
def sample_filter(args, groups: list[RolloutGroup]) -> None
```

Operates in-place on kept groups.
To exclude a span from training, set its `trajectory.loss_mask` to all zeros; zeroing every span of an episode excludes the whole attempt.

### Data source (`--data-source-path`)

```python
class DataSource(abc.ABC):
    def get_examples(self, num_prompts: int) -> list[dict]: ...
    def add_examples(self, examples: list[dict]): ...
    def save(self, rollout_id): ...
    def load(self, rollout_id=None): ...
    def __len__(self) -> int: ...
```

### Loss function (`--custom-loss-function-path`)

```python
def custom_loss(args, unpacked_batches: list[dict]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Set `--loss-type custom_loss` to use it. The function owns the entire loss math
(policy, entropy, KL as it sees fit) and replaces the built-in policy loss.
`unpacked_batches` is the per-document list from `unpack_sequences`, one entry per
trajectory; each dict carries the raw `reward`, `loss_weights`, `cur_log_probs`,
`advantages`, `loss_masks`, and the old/ref log-probs when available.
Return `(loss, metrics)` where `loss` is the summed-microbatch loss (the
framework applies global-batch normalization and `backward()`) and `metrics` is
a dict of scalar tensors logged under `train/`.

### Logging (`--custom-rollout-log-function-path`)

```python
def log_rollout_data(rollout_id, args, episodes, rollout_extra_metrics, rollout_time) -> bool
```

Return `True` to skip default W&B logging, `False` to run both.

## Multi-Turn / Agentic Adaptation

Multi-turn interactions (tool-calling agents, conversational RL) require three pieces:

**1. Data**: the `prompt` field stores the initial conversation. Custom fields (tools, environment state, ground truth) are accessed via `episode.example`.

**2. Custom generate function**: a loop that alternates between model generation and tool execution, controlling `loss_mask`:
- Model-generated tokens: `loss_mask = 1` (train on these)
- Tool/environment tokens: `loss_mask = 0` (excluded from loss)

When each turn extends one growing context, the attempt stays a single trajectory:

```python
async def multi_turn_generate(state, episode):
    trajectory = episode.trajectory
    for turn in range(max_turns):
        episode = await generate(state, episode)  # appends with loss_mask=1
        tool_call = parse_tool_call(trajectory.generated_text)
        if tool_call is None:
            break
        tool_tokens = state.tokenizer.encode(execute_tool(tool_call))
        trajectory.token_ids.extend(tool_tokens)
        trajectory.loss_mask.extend([0] * len(tool_tokens))
        trajectory.rollout_log_probs.extend([0.0] * len(tool_tokens))
    episode.status = Episode.Status.COMPLETED
    return episode
```

When a turn does not extend the previous context, append a trajectory instead. Sub-agent
dispatch fans out and context compression replaces context rather than extending it, so each
generation call becomes its own span:

```python
async def compressing_generate(state, episode):
    for turn in range(max_turns):
        episode = await generate(state, episode)
        if is_done(episode.trajectories[-1].generated_text):
            break
        # A fresh span over the compressed context, not an extension of the old one.
        compressed = compress(episode.trajectories[-1])
        episode.trajectories.append(Trajectory(token_ids=compressed))
    episode.status = Episode.Status.COMPLETED
    return episode
```

Trajectories carry no ordering relation. A workload whose reward depends on span order records
that order in `episode.example`, where its own reward function reads it.

**3. Custom reward function**: scores the final outcome of the interaction, either as one
`episode.reward` for the attempt or as a `trajectory.reward` per span.

```bash
--custom-generate-function-path my_project.agent.multi_turn_generate \
--custom-rm-path my_project.rewards.multi_turn_rm
```

## Testing Custom Hooks

Contract tests in `tests/plugin_contracts/` validate hook signatures and return types without GPU:

```bash
pytest tests/plugin_contracts/ -v
```

Override the implementation under test via env vars (e.g. `SLIME_CONTRACT_CUSTOM_GENERATE_FUNCTION_PATH=my_module.fn`) or CLI args when running test files directly.

CI label: `run-ci-plugin-contracts`.

---

**See also:** [Data Layout](data-layout.md) | [SGLang Config](sglang-config.md) | [Training Loss](training-loss.md)
