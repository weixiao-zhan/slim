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

## Where Hooks Run

Rollout is split between one `RolloutManager` coordinator process and one `RolloutWorker` process on every node of the Ray cluster, head node included; see [Distributed Rollout](distributed-rollout.md).

| Process | Hooks |
|---|---|
| `RolloutWorker` (every node) | `--custom-generate-function-path`, `--custom-rm-path` |
| `RolloutManager` (coordinator) | `--rollout-function-path`, `--eval-function-path`, `--data-source-path`, `--rollout-group-filter-path`, `--rollout-sample-filter-path`, `--rollout-all-samples-process-path`, `--custom-rollout-log-function-path`, `--custom-eval-rollout-log-function-path` |
| `AdvantageEstimator` | `--custom-reward-post-process-path` |

Every node imports the custom generate and reward modules and loads the tokenizer and processor from `--hf-checkpoint`, so both must be reachable from every node.
Hooks running in the coordinator or `AdvantageEstimator` see `trajectory.multimodal_inputs` as a `ray.ObjectRef`; they pass it through without reading it (see [Multimodal Input Lifecycle](data-layout.md#multimodal-input-lifecycle)).

## Key Interfaces

### Rollout function (`--rollout-function-path`)

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

Replaces the entire rollout loop.
Override this for fundamentally different sampling strategies (tree search, offline data, non-SGLang backend).
It runs in the `RolloutManager` process and reaches the rollout workers through `RolloutWorkerPool(args)`.

### Generate function (`--custom-generate-function-path`)

```python
async def custom_generate(state: GenerateState, episode: Episode) -> Episode
```

Substitutes only the per-episode generation logic.
The episode arrives with no trajectory; the function appends every `Trajectory` it generates, each with its own `token_ids`, `loss_mask`, and `rollout_log_probs`, and returns the episode.
An agentic workload appends further `Trajectory` objects to `episode.trajectories`, one per contiguous generation, and sets `episode.status` for the attempt as a whole.

The function owns the sampling fields of its own requests, but should honor the token budget from `episode.max_tokens` and the `--rollout-temperature` that training scales its logits by.

Per-episode override: set `episode.generate_function_path` to route specific episodes to a different generate function.

The function runs inside the `RolloutWorker` of the node its group was dispatched to, and all episodes of a group run in the same worker.
Environments are node-local: a function that starts Docker containers talks to the daemon of that node, which may be the head node or any worker node.

Abort is cooperative, so the function must return on its own in both the normal and the abort path:

- A response with `finish_reason == "abort"` ends the episode: set `Episode.Status.ABORTED` and return.
- Between environment steps, check `state.aborted` and return when it is set; an engine abort cannot interrupt a running environment call.
- Create containers inside the `try` whose `finally` removes them.
- Run blocking or CPU-heavy calls (environment calls, heavy processor or reward work) through `asyncio.to_thread` or a process pool. One event loop serves all groups on the node, so a blocking call stalls every episode on that node and delays abort.

See [Generate Function Contract](distributed-rollout.md#generate-function-contract) for an example.

### Reward model (`--custom-rm-path`)

```python
async def custom_rm(args, episode: Episode, **kwargs) -> None

# Batch mode (--group-rm):
async def custom_rm(args, episodes: list[Episode], **kwargs) -> None
```

A reward function can set one scalar per episode on `episode.reward`, which is auto broadcast to every trajectory, or set `trajectory.reward` on each of `episode.trajectories`, and returns nothing; setting both levels is an error.
The reward function runs in the same `RolloutWorker` as the generate function, on the same event loop, so heavy scoring goes through `asyncio.to_thread` or a process pool as well.
The built-in `--rm-type` options (`math`, `deepscaler`, `f1`, `gpqa`, `ifbench`, `random`), used when `--custom-rm-path` is not set, score `episode.trajectories[-1].generated_text` against `episode.example["label"]`, except `random`, which ignores both.

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
To exclude a trajectory from training, set its `trajectory.loss_mask` to all zeros; zeroing every trajectory of an episode excludes the whole attempt.

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

**2. Custom generate function**: there are two common approches: append-only vs per-API-trajectories.

Append-only keeps one growing context as a single trajectory, and slim owns tokenization: the generate function calls `/generate` with token ids, then extends `token_ids`, `loss_mask`, and `rollout_log_probs` accordingly.

One trajectory per API call let the engine handles tokenization, which lets an agent framework can talk to rollout engine via `/v1/chat/completions`. 
Slim has patched sglang engine and router to return extra fields to recoard each API call as a seperate `Trajectory`. Interleaved thinking, sub-agent dispatch, and context compression fit here.

See [Endpoints and Tokenization Ownership](sglang-config.md#endpoints-and-tokenization-ownership) for the request and response fields.

**3. Custom reward function**: scores the `episode.reward` based on final outcome or scores all `trajectory.reward`.

```bash
--custom-generate-function-path my_project.agent.generate \
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
