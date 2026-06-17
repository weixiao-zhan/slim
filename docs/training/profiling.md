# Profiling

slim has four profiling tools, covered in the sections below:

| | Tool | Covers | Output |
| --- | --- | --- | --- |
| **A. Backend training profiler** | `torch.profiler` | time spent in the policy-gradient update (fwd+bwd) and logprob recompute | chrome trace |
| **B. Rollout generation profiler** | VizTracer | time spent in the rollout generation orchestration: multi-turn dialog, vision encoding, tokenize, reward, await-engine | chrome trace |
| **C. Memory recorder** | torch / memray | GPU and/or host memory usage | `.pickle` / `.bin` |
| **D. SGLang engine profiler** | SGLang endpoints | GPU inference kernels (prefill/decode) during generation | chrome trace |

The chrome traces (`.json.gz`) open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

---

The performance profilers (A and B) share one CLI flag family; the target you pass selects which tool runs.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--profile-target` | *(empty = off)* | One or more of `train_pg`, `train_log_probs`, `train_overall` (Part A), `rollout` (Part B). Passing any target turns profiling on. |
| `--profile-step-start` | `10` | Start of the capture window (inclusive). |
| `--profile-step-end` | `12` | End of the capture window (exclusive). |
| `--profile-dir` | `./profiles` | Output dir for all traces (and memory snapshots). |

Each profiled step writes **its own independent trace file**; nothing is overwritten or accumulated across steps.

---

## Part A — Backend training profiler

`torch.profiler` over the training loop. Targets:

| Target | Scope of one trace | "step" unit for the window |
| --- | --- | --- |
| `train_pg` | the policy-gradient update (fwd+bwd over microbatches) | microbatch |
| `train_log_probs` | logprob recompute loop (only has content when `--old-logprob-source actor`) | microbatch |
| `train_overall` | the whole train loop, spanning rollouts | rollout |

Files: `train_pg_rank_<r>_time<ts>.json.gz`, `train_log_probs_...`, `train_overall_...`.

```bash
python train.py \
    --profile-target train_pg \
    --profile-step-start 0 \
    --profile-step-end 1 \
    --profile-dir ./profiles \
    ... (other arguments)
```

> **Host-RAM tip.** Torch traces (which include Python stacks) are accumulated in host RAM before being written and can be large. If host RAM is tight, narrow the window (`--profile-step-start/end`) and drop `--master-weight-dtype fp32` for profiling runs.

---

## Part B — Rollout generation profiler

VizTracer over the `rollout` target — one rollout step's generation phase (all its `generate` calls + reward: multi-turn dialog, vision encoding, tokenize, reward, await-engine). Here the window's "step" unit is the **rollout index**, and one independent trace is written per rollout step: `rollout_rank<r>_rollout<id>.json.gz`. Each concurrent episode (asyncio task) is on its own track, so in-flight episodes are legible rather than collapsed into one event-loop stack.

```bash
python train.py \
    --profile-target rollout \
    --profile-step-start 0 \
    --profile-step-end 1 \
    --profile-dir ./profiles \
    ... (other arguments)
```

---

## Part C — Memory recorder

Independent of the performance profiler (does not need `--profile-target`); records *memory* rather than time.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--memory-recorder` | *(empty = off)* | One or more of `torch`, `memray`. Passing any recorder turns it on. |
| `--memory-snapshot-num-steps` | `None` | Stop recording and dump the snapshot after this many rollouts (**required for memray**). |

| Recorder | Records | Output | View with |
| --- | --- | --- | --- |
| `torch` | GPU / CUDA caching-allocator history; auto-dumps on CUDA OOM via an observer | `.pickle` | <https://pytorch.org/memory_viz> |
| `memray` | host / CPU RAM allocations | `.bin` | `memray` flamegraph / table |

You can pass both (`--memory-recorder torch memray`) to track GPU and host memory simultaneously. Snapshots are written to `--profile-dir`.

```bash
python train.py \
    --memory-recorder torch memray \
    --memory-snapshot-num-steps 3 \
    --profile-dir ./profiles \
    ... (other arguments)
```

---

## Part D — SGLang engine (GPU inference) profiling

For the inference kernels inside the SGLang engine workers, use SGLang's own profiling interface.

### 1. Sleeping the rollout process

For flexible stress testing and profiling, it is often useful to make the slim rollout process wait after initialization instead of generating immediately. Swap the rollout function via startup args (no source changes):

```bash
python train.py \
    --rollout-function-path slim.rollout.sleep_rollout.sleep \
    ... (other arguments)
```

This enters an infinite wait loop, letting you manually send requests or run stress tools.

### 2. Obtaining the SGLang engine list

Engines (workers) register with the router. Retrieve active engines from the router's `/workers` endpoint. The router address is printed at startup:

```
Router launched at 127.0.0.1:3000
```

```bash
curl http://127.0.0.1:3000/workers
```

### 3. Automated profiling tool

`tools/profile_rollout.py` profiles all engines at once. By default it starts profiling on every worker and stops after 3 steps:

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

**Key parameters:**
* `--router-url`: the router URL.
* `--num-steps`: number of steps to record (default 3).
* `--output-dir`: directory for trace files.
* `--activities`: activities to monitor, e.g. `GPU` `CPU`.
* `--profile-by-stage`: profile by stage (prefill/decode).

Stop early (if `--num-steps` was not set):

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action stop
```

### 4. Running stress tests

While the rollout process waits via `sleep_rollout`:
1. Start profiling with `tools/profile_rollout.py`.
2. Send requests with stress tools (e.g. SGLang's built-in benchmark tools) to the router or engines.
3. Wait for profiling to finish (if `--num-steps` was set) or stop it manually.
4. Collect the `.json` trace files from `output_dir` and view them in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).
