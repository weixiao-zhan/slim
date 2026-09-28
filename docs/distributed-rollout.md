# Distributed Rollout Migration

This document plans how slim moves rollout from one asyncio loop in `RolloutManager` to one rollout worker process on every node of the Ray cluster.
Agentic rollouts (Docker environments, code execution, agent frameworks) spend much of their CPU outside token generation, and one process with one GIL on one node cannot keep the engines busy.

## Principles

- **One example, one group.** A dataset example becomes one `RolloutGroup` of `n_samples_per_prompt` episodes, exactly as today. The group is the dispatch unit: all episodes of a group run in the same worker.
- **One worker per node.** Each node runs one `RolloutWorker` process. A worker runs many groups at once on its own event loop, with the episode semaphore that bounds the whole rollout today, sized to its share.
- **Custom functions own the data.** The framework ships the raw example dict to a worker and does not interpret it. The generate function decides how to read columns, load media, start containers, and which processor outputs to carry.
- **Environments are node-local.** A generate function that starts Docker talks to the daemon of the node its worker runs on, which may be the head node or any worker node.
- **Abort is cooperative.** Nothing cancels a task or kills a process. Abort raises a flag and aborts in-flight engine requests; every generate function returns on its own, so its `finally` always runs.
- **Stages cooperate instead of checking each other.** Each field has exactly one form at each stage, documented below. A stage relies on the contract of the stage before it and does not re-validate, branch on types, or guard against forms the pipeline never produces.

## Current Structure

`RolloutManager` (`slim/ray/rollout.py`) is one Ray actor with `num_cpus=1`.
It owns the data source and calls `generate_rollout` (`slim/rollout/sglang_rollout.py`), which runs every group as an asyncio task on the background loop of `slim/utils/async_utils.py`.
`GenerateState` is a process singleton holding the tokenizer, processor, the episode semaphore of size $P \cdot c$, and the pending task set, where $P$ is the number of rollout replicas and $c$ is `--rollout-concurrency-per-replica`.
Tokenization, processor calls, reward functions, agent frameworks, base64 decoding, and JSON parsing all share that loop.
Finished episodes are flattened and split into one `TrajectoryBatch` per DP rank, and `ray.put` from the manager process.

## Target Structure

```
                ┌──────────────── RolloutManager (coordinator) ────────────────┐
                │ data source · dynamic filter · batch hooks · abort · DP split │
                │ RolloutWorkerPool: per-worker episode load, pending tasks     │
                └──────▲ RolloutGroup (bulk fields as ObjectRef)    │ RolloutGroup (PENDING)
                       │                                            ▼
  every node ── RolloutWorker (Ray async actor, many groups, semaphore ⌈P·c / N⌉)
                 GenerateState · HTTP client · generate_and_rm_group · local Docker daemon
                 ray.put(multimodal_inputs) stays in this node's object store
                       │                                         │ HTTP
                       │                                         ▼
                       │                                   sglang router → engines
                       ▼
  trainer DP rank ── ray.get(TrajectoryBatch) → ray.get(multimodal_inputs refs)
```

| Component | Process | Owns |
|---|---|---|
| `RolloutManager` | one actor, as today | data source, dynamic filter, batch-level hooks, abort, DP split, logging, tracking, engine lifecycle |
| `RolloutWorkerPool` | singleton inside the `RolloutManager` process | worker handles, per-worker episode load, pending group tasks, `aborted` flag |
| `RolloutWorker` | one async actor on every alive node | `GenerateState`, HTTP client, the groups assigned to it |
| generate function | runs inside a worker | example interpretation, media loading, containers, processor outputs |

### Worker

```python
@ray.remote
class RolloutWorker:
    def __init__(self, args, concurrency: int):
        configure_logger()
        init_http_client(concurrency)
        self.args = args
        self.state = GenerateState(args, concurrency)

    async def run_group(self, group: RolloutGroup, evaluation: bool) -> RolloutGroup:
        group = await generate_and_rm_group(self.args, group, evaluation=evaluation)
        offload_bulk_fields(group)
        return group

    async def abort(self) -> None:
        self.state.aborted = True

    async def reset(self) -> None:
        self.state.reset()
```

- `RolloutWorker` is an async actor, so every `run_group` call and `abort` run concurrently on the one event loop thread of the actor.
- `generate_and_rm`, `generate_and_rm_group`, the default `generate`, and reward functions run unchanged inside the worker, including the semaphore, decoding, `--group-rm`, session ids, and deterministic sampling seeds.
- `GenerateState(args, concurrency)` sizes its semaphore to the worker's share. It keeps the tokenizer, processor, chat template kwargs, routing replay shape, and the `aborted` flag, and drops the pending task set and `submit_generate_tasks`, which move to the pool.
- `init_http_client(concurrency)` takes the connection limit as a parameter and builds the client with `trust_env=False`, so proxy variables on a remote node do not capture traffic to the router.
- The worker does not initialize tracking; rollout logging and W&B stay in `RolloutManager`.

### Placement and Sizing

The pool creates one worker on every alive Ray node with a CPU, head node included, with `NodeAffinitySchedulingStrategy` and `num_cpus=1`, and waits for them to become ready.
With $N$ workers, the total episode concurrency stays $P \cdot c$, the size of today's semaphore, and each worker's semaphore is $\lceil P \cdot c / N \rceil$.
Every node imports the custom generate and reward modules and loads the tokenizer and processor from `--hf-checkpoint`, so both must be reachable from every node.

The pool exists only where rollout servers exist: `RolloutManager` creates it after `start_rollout_servers`, and `--debug-train-only` creates none.

### Pool

```python
class RolloutWorkerPool(metaclass=SingletonMeta):
    async def run_group(self, group: RolloutGroup, evaluation: bool) -> RolloutGroup: ...
    def submit_generate_tasks(self, groups: list[RolloutGroup]) -> None: ...
    async def abort(self) -> list[dict]: ...    # drains, returns examples of incomplete groups
    def reset(self) -> None: ...
```

`run_group` is what `generate_and_rm_group` is to the coordinator today:

1. Return the group untouched if the pool is aborted.
2. Wait for a worker whose episode load is below its semaphore size, and pick the least loaded one.
3. Add the group's episode count to that worker's load, await `worker.run_group.remote(group, evaluation)`, and subtract it again in a `finally`.

A worker therefore holds at most one group beyond what its semaphore admits, so its semaphore always has episodes queued, and groups that do not fit anywhere wait in the coordinator where any worker that frees capacity picks them up.
Engine slots freed by a long-tail episode go to episodes of other groups on the same worker, as with today's single semaphore.

The coordinator reaches workers only by calling them; workers never call `RolloutManager`, which is a synchronous actor blocked in `generate` for the whole rollout.
A custom `--rollout-function-path` keeps its signature and runs in the `RolloutManager` process; it reaches the workers through `RolloutWorkerPool(args)`.

### Train Rollout Loop

`generate_rollout_async` keeps its shape: build groups, submit, wait for the first completion, filter, repeat.
`state.submit_generate_tasks` and `state.pendings` become `pool.submit_generate_tasks` and `pool.pendings`, whose tasks call `pool.run_group`.
The refill counts from `over_sampling_batch_size` and `rollout_batch_size` stay as they are.
The dynamic filter, `--rollout-sample-filter-path`, and `--rollout-all-samples-process-path` run in the `RolloutManager` process on returned groups, with the same signatures.

### Abort

`RolloutWorkerPool.abort` runs when the batch is full:

1. Set the pool's `aborted` flag, so pending tasks that have not reached a worker return their group untouched.
2. Call `abort` on every worker, which sets `state.aborted` in that worker. All groups of a worker share its `GenerateState`, and threads the generate function starts through `asyncio.to_thread` read the same object.
3. Post `abort_request` with `abort_all` to every engine behind the router once per second until every pending task has returned. This is the existing loop of `sglang_rollout.abort`. `abort_all` aborts only the requests running at that moment, so the repeat catches requests a worker posted before it saw its flag.
4. Collect the examples of groups that are not `completed` for `data_source.add_examples`.
5. Call `reset` on every worker and on the pool.

The coordinator keeps its own HTTP client for `GET /workers` and the abort posts, with a connection limit covering every engine.
An aborted group returns through the normal path, so its bulk refs are dropped with it.

### Generate Function Contract

A generate function runs to completion in both the normal path and the abort path:

- A request whose response has `finish_reason == "abort"` ends the episode: set `Episode.Status.ABORTED` and return.
- Between environment steps, check `state.aborted` and return when it is set. An engine abort cannot interrupt a running `docker exec`, so this check bounds abort latency to one environment step.
- Create containers inside the `try` whose `finally` removes them.
- Run blocking or CPU-heavy calls (environment calls, heavy processor or reward work) through `asyncio.to_thread` or a process pool. A worker has one event loop for all groups on its node, so a blocking call on the loop thread stalls every episode on that node and delays the worker's `abort` until it returns.

```python
async def generate(state: GenerateState, episode: Episode) -> Episode:
    container = None
    try:
        container = await start_container(episode.example)
        while not done:
            if state.aborted:
                episode.status = Episode.Status.ABORTED
                return episode
            ...  # model call; finish_reason "abort" → ABORTED, return
            ...  # environment step in the container, via asyncio.to_thread
        return episode
    finally:
        if container is not None:
            await remove_container(container)
```

### Bulk Field Lifecycle

`Trajectory.multimodal_inputs` carries whatever processor outputs the generate function attached, one full dict per trajectory, including every later-turn image of a multi-turn attempt.
It moves by reference so the coordinator never holds the tensors:

| Stage | Form of `multimodal_inputs` |
|---|---|
| generate function, reward function (worker) | `dict[str, Tensor] \| None` |
| `offload_bulk_fields` at the end of `RolloutWorker.run_group` | `ray.put(dict)` in the worker, becomes `ObjectRef \| None` |
| filters, batch hooks, rollout logging, `build_dp_batches`, `AdvantageEstimator` and `--custom-reward-post-process-path` | `ObjectRef \| None`, never read |
| `process_rollout_data` (trainer) | one `ray.get` over the batch's refs, written back as `dict[str, Tensor] \| None` |
| packing and training | `dict[str, Tensor] \| None`, unchanged |

The worker owns the objects and lives for the run, and Ray keeps nested refs alive while the batch that contains them is alive.
`AdvantageEstimator.compute_training_targets` then fetches and re-puts batches without moving any processor tensor.
`--save-debug-rollout-data` resolves the refs inside the `dataclasses.asdict` output it saves, for both train and eval, so the live episodes keep their refs; `--load-debug-rollout-data` puts the loaded dicts back with `ray.put`.

Token-aligned fields (`token_ids`, `loss_mask`, `rollout_log_probs`, `rollout_routed_experts`) stay inline, because source-token finalization and DP padding in `RolloutManager` read them.

### Eval

`eval_rollout_single_dataset` builds one group per dataset row with `eval_n_samples_per_prompt` episodes, sets `generate_function_path`, `max_tokens`, and `metadata` on each episode as today, and gathers `pool.run_group(group, evaluation=True)` over the rows.
Going through `generate_and_rm_group` gives eval episodes session ids, so the `consistent_hashing` router policy routes them by session like train episodes.
Each call waits for its own group, so the eval datasets that `eval_rollout` gathers concurrently share the pool without taking each other's results, and the worker semaphores bound eval to $P \cdot c$ episodes as today.

### Data Source

`load_hf_dataset` and `RolloutDataSource` stay as they are.

## Changes by File

| File | Change |
|---|---|
| `slim/rollout/worker.py` (new) | `RolloutWorker`, `RolloutWorkerPool`, `offload_bulk_fields` |
| `slim/rollout/sglang_rollout.py` | `GenerateState(args, concurrency)` without pending tasks; `generate_rollout_async`, `abort`, and eval dispatch through the pool |
| `slim/utils/data.py` | `process_rollout_data` resolves `multimodal_inputs` refs |
| `slim/ray/rollout.py` | `RolloutManager` creates the pool after the servers start; debug save resolves refs in the saved copy and debug load re-puts them |
| `slim/utils/http_utils.py` | `init_http_client(concurrency)` with `trust_env=False`; remove the distributed POST actors |
| `slim/utils/arguments.py` | remove `--use-distributed-post` |
| `docs/customization.md` | generate function runs in a worker; abort contract; node-local environments; hooks see `multimodal_inputs` as `ObjectRef` |
| `docs/data-layout.md` | `multimodal_inputs` lifecycle |
| `docs/placement.md` | rollout worker placement and sizing |
| `docs/dev-utils.md` | `profile_rollout` traces the coordinator only |
| `tests/plugin_contracts/test_plugin_generate_contracts.py` | fake `GenerateState` without the pending task set |
| `tests/` | pool dispatch and load accounting, abort drain with a slow environment step, ref round trip through `process_rollout_data` |

`--rollout-concurrency-per-replica` keeps its engine meaning (`max_running_requests`, CUDA graph batch size) and keeps bounding rollout-side concurrency, now split over the worker semaphores.
`--use-distributed-post` goes away because every worker posts from its own node.

## Out of Scope

- More than one worker per node. A node's Python-side work shares one GIL; when that becomes the bottleneck, the worker count per node becomes configurable without changing the pool or the contracts.
- Moving `rollout_routed_experts` by reference, which needs source-token finalization to move into the worker.
- Resolving `multimodal_inputs` refs at packing time; `compute_log_probs` and `train` each call `process_rollout_data`, so a rank fetches the processor outputs twice.
- Staleness-bounded fully asynchronous rollout; `train_async.py` keeps its one-step overlap.
- Placement on a subset of nodes selected by a custom Ray resource.
