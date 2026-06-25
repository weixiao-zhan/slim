# Dev Utils

## Profiling

Slim has three profiling components. 

### 1. Profiler

`--profile-target` accepts one or more of the following:

| Target | Tool | Scope |
|--------|------|-------|
| `train_pg` | torch.profiler | Policy gradient update (fwd+bwd over microbatches) |
| `train_log_probs` | torch.profiler | Log-prob recompute loop |
| `train_overall` | torch.profiler | Entire train loop spanning rollouts |
| `rollout` | VizTracer | Rollout step (generation + reward); each concurrent episode on its own track |

Other flags: `--profile-step-start` (default 10), `--profile-step-end` (default 12), `--profile-dir` (default `./profiles`).
Output: `<profile-dir>/<target>_rank_<rank>_....json.gz`.
Traces open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

### 2. Memory recorder

`--memory-recorder` accepts one or more of the following:

| Recorder | Records | Output | View with |
|----------|---------|--------|-----------|
| `torch` | CUDA allocator history; auto-dumps on OOM | `.pickle` | [pytorch.org/memory_viz](https://pytorch.org/memory_viz) |
| `memray` | Host RAM allocations | `.bin` | `memray flamegraph` |

Other flags: `--memory-snapshot-num-steps` (required for `memray`; stops recording and dumps after N rollouts).
Independent of `--profile-target`.

### 3. SGLang engine profiler

`tools/profile_rollout.py` triggers profiling on all SGLang engines via the router:

```bash
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:3000 \
    --action start \
    --num-steps 3 \
    --activities GPU \
    --profile-by-stage
```

`--rollout-function-path slim.rollout.sleep_rollout.sleep` replaces generation with an infinite sleep loop, useful for holding the cluster idle while stress-testing engines with this profiler.

---

## Debugging

### Separate debugging modes

| Flag | Effect |
|------|--------|
| `--debug-rollout-only` | Skip training, run only the rollout pipeline. |
| `--debug-train-only` | Skip SGLang, run only the training loop (requires `--load-debug-rollout-data`). |
| `--save-debug-rollout-data` | Save rollout tensors to `<path>.format(rollout_id)` after each step. |
| `--load-debug-rollout-data` | Load saved rollout data instead of generating. Implies `--debug-train-only`. |

### Weight-update verification

`--check-weight-update-equal` snapshots engine weights at startup, zeros them, then verifies the first FSDP→SGLang sync restores the exact values.

### CUDA IMA debugging

1. Set `CUDA_LAUNCH_BLOCKING=1` for accurate stack traces.
2. Toggle off potential causes: speculative decoding, CUDA graphs, DeepEP.
3. For persistent issues: `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_FILE=core.cuda`.

---

## Reproducibility

| config | Flag / env var |
|-------------|----------------|
| Deterministic rollout | `--sglang-enable-deterministic-inference` and `--sglang-attention-backend flashinfer` |
| Deterministic PyTorch ops | `TORCH_USE_DETERMINISTIC_ALGORITHMS=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

---

## Tests

See [tests/RUNNING_TESTS.md](../tests/RUNNING_TESTS.md) for details.

---

**See also:** [Placement & Weight Update](placement.md) | [Training Loss](training-loss.md) | [Customization](customization.md)
