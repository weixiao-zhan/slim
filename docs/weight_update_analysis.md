# Weight Update: 2x2x2 Configuration Analysis

## Axes

| Axis | Values |
|------|--------|
| Nodes | **Single node** vs **Multi node** |
| GPU sharing | **Colocate** (`--rollout-colocate`, sync `train.py`) vs **Separate** (async `train_async.py` or sync `train.py` without `--rollout-colocate`) |
| GPUs per engine | **=1** (1 GPU per engine) vs **>1** (multiple GPUs per engine) |

> **Note**: slim treats each sglang engine as an opaque unit of `num_gpus_per_replica` GPUs.
> How sglang internally splits those GPUs into TP/PP/DP/EP is configured via sglang args
> (`--sglang-pp-size`, `--sglang-tp-size`, `--sglang-ep-size`) and is transparent to the

## Weight Updater Selection

```python
self.weight_updater = (
    UpdateWeightFromTensor(self.args, self.model)    # Colocate
    if self.args.rollout_colocate
    else UpdateWeightFromDistributed(self.args, self.model)  # Separate
)
```

- **Colocate** → `UpdateWeightFromTensor`: Gloo IPC gather to source rank → send to sglang engine via Ray IPC
- **Separate** → `UpdateWeightFromDistributed`: NCCL broadcast from rank 0 → sglang workers

Both inherit from `UpdateWeight`, which drives the outer loop:

```
for each param in model.state_dict():          # returns FSDP2 DTensors
    redistribute(Replicate, async_op=True)      # enqueues NCCL all-gather on FSDP mesh
    append to bucket
    if bucket full:
        wait_and_update_bucket_weights(bucket)  # wait() + update_bucket_weights() + barrier()
```

`redistribute(Replicate)` is an NCCL all-gather across the entire FSDP mesh — every training rank must participate. A Gloo barrier after each bucket keeps all ranks synchronized.

---

## Process Groups Reference

| Group | Backend | Scope | Used By |
|-------|---------|-------|---------|
| **FSDP mesh** | NCCL | All training ranks (all nodes) | `redistribute(Replicate)` all-gather |
| **Gloo barrier** | Gloo | All training ranks (all nodes) | `dist.barrier()` between buckets |
| **IPC gather** (colocate only) | Gloo | Training ranks mapped to one engine | `gather_object` to source rank |
| **Model update** (separate only) | NCCL | Training rank 0 + all sglang engine workers | `dist.broadcast` to engines |

---

## The 8 Configurations

### 1. Single Node + Colocate + 1 GPU/engine

Example: 8 GPUs, 8 sglang engines (1 GPU each)

```
GPUs:       [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
FSDP:        r0   r1   r2   r3   r4   r5   r6   r7
Engines:     e0   e1   e2   e3   e4   e5   e6   e7
IPC groups: {0} {1}  {2}  {3}  {4}  {5}  {6}  {7}
Sources:     *    *    *    *    *    *    *    *
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromTensor` |
| FSDP mesh | 8 ranks |
| IPC groups | 8 singleton groups |
| Source ranks | All 8 (every rank is its own source) |

All ranks send to their own engine in parallel. Natural synchronization — no rank races ahead.

---

### 2. Single Node + Colocate + Multi GPU/engine

Example: 8 GPUs, 2 sglang engines (4 GPUs each)

```
GPUs:       [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
FSDP:        r0   r1   r2   r3   r4   r5   r6   r7
Engines:     ----engine 0----   ----engine 1----
IPC groups: {  0,  1,  2,  3} {  4,  5,  6,  7}
Sources:     *                  *
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromTensor` |
| FSDP mesh | 8 ranks |
| IPC groups | 2 groups of 4 |
| Source ranks | 2 (ranks 0, 4) |

4 ranks per group gather to source. Only source ranks send to sglang. Barrier keeps non-source ranks from racing ahead.

---

### 3. Single Node + Separate + 1 GPU/engine

Example: 4 train GPUs + 4 rollout GPUs on same node, 4 sglang engines (1 GPU each)

```
GPUs:       [0]  [1]  [2]  [3] | [4]  [5]  [6]  [7]
FSDP:        r0   r1   r2   r3 |
Engines:                        |  e0   e1   e2   e3
Broadcast:   r0 ─────────────────► all 4 engines
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromDistributed` |
| FSDP mesh | 4 ranks |
| Model update group | rank 0 + 4 sglang workers |
| Source ranks | 1 (rank 0 only) |

Rank 0 broadcasts full params to all engines. Ranks 1-3 skip send. Barrier keeps them synchronized.

---

### 4. Single Node + Separate + Multi GPU/engine

Example: 4 train GPUs + 4 rollout GPUs on same node, 1 sglang engine (4 GPUs)

```
GPUs:       [0]  [1]  [2]  [3] | [4]  [5]  [6]  [7]
FSDP:        r0   r1   r2   r3 |
Engine:                         |  ----engine 0 (4 GPUs)----
Broadcast:   r0 ─────────────────► 4 engine workers
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromDistributed` |
| FSDP mesh | 4 ranks |
| Model update group | rank 0 + 4 sglang workers |
| Source ranks | 1 (rank 0 only) |

Same as case 3 — rank 0 broadcasts, barrier synchronizes.

---

### 5. Multi Node + Colocate + 1 GPU/engine

Example: 2 nodes x 8 GPUs, 16 sglang engines (1 GPU each), HSDP

```
Node A:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
Node B:     [8]  [9]  [10] [11] [12] [13] [14] [15]
FSDP:        r0...r7                r8...r15         (HSDP 2D mesh)
Engines:     e0...e7                e8...e15
IPC groups: {0} {1} ... {7}       {8} {9} ... {15}
Sources:     *   *       *         *   *        *    (all 16)
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromTensor` |
| FSDP mesh | 16 ranks (2D HSDP) |
| IPC groups | 16 singleton groups |
| Source ranks | All 16 |

All ranks send to their own engine in parallel. Natural synchronization.

---

### 6. Multi Node + Colocate + Multi GPU/engine

Example: 2 nodes x 8 GPUs, 4 sglang engines (4 GPUs each), HSDP

```
Node A:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
Node B:     [8]  [9]  [10] [11] [12] [13] [14] [15]
FSDP:        r0...r7                r8...r15
Engines:     ---eng 0---  ---eng 1---  ---eng 2---  ---eng 3---
IPC groups: {0, 1, 2, 3} {4, 5, 6, 7} {8,9,10,11} {12,13,14,15}
Sources:     *                *              *              *     (4 of 16)
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromTensor` |
| FSDP mesh | 16 ranks (2D HSDP) |
| IPC groups | 4 groups of 4 |
| Source ranks | 4 (ranks 0, 4, 8, 12) |

4 source ranks send to engines. Barrier keeps the other 12 synchronized.

---

### 7. Multi Node + Separate + 1 GPU/engine

Example: 2 train nodes (16 GPUs) + 2 rollout nodes (16 GPUs), 16 engines (1 GPU each)

```
Train Node A:   [r0..r7]     Train Node B:   [r8..r15]
Rollout Node C: [e0..e7]     Rollout Node D: [e8..e15]

FSDP mesh: {r0..r15}
Broadcast:  r0 ──────► all 16 sglang workers
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromDistributed` |
| FSDP mesh | 16 ranks |
| Model update group | rank 0 + 16 sglang workers |
| Source ranks | 1 (rank 0 only) |

Rank 0 broadcasts full params. Barrier keeps the other 15 ranks synchronized.

---

### 8. Multi Node + Separate + Multi GPU/engine

Example: 2 train nodes (16 GPUs) + 1 rollout node (8 GPUs), 2 engines (4 GPUs each)

```
Train Node A:   [r0..r7]     Train Node B:   [r8..r15]
Rollout Node C: [---eng 0 (4 GPUs)---] [---eng 1 (4 GPUs)---]

FSDP mesh: {r0..r15}
Broadcast:  r0 ──────► 8 sglang workers
```

| Property | Value |
|----------|-------|
| Updater | `UpdateWeightFromDistributed` |
| FSDP mesh | 16 ranks |
| Model update group | rank 0 + 8 sglang workers |
| Source ranks | 1 (rank 0 only) |

Same as case 7 — rank 0 broadcasts, barrier synchronizes.

---

## Summary

| # | Nodes | GPU Sharing | GPUs/engine | Updater | Source Ranks |
|---|-------|------------|-----|---------|-------------|
| 1 | Single | Colocate | 1 | FromTensor | All |
| 2 | Single | Colocate | >1 | FromTensor | 1 per engine |
| 3 | Single | Separate | 1 | FromDistributed | Rank 0 only |
| 4 | Single | Separate | >1 | FromDistributed | Rank 0 only |
| 5 | Multi | Colocate | 1 | FromTensor | All |
| 6 | Multi | Colocate | >1 | FromTensor | 1 per engine |
| 7 | Multi | Separate | 1 | FromDistributed | Rank 0 only |
| 8 | Multi | Separate | >1 | FromDistributed | Rank 0 only |

**Synchronization**: `wait_and_update_bucket_weights` ends with a Gloo barrier so that all FSDP ranks complete each bucket before any rank proceeds to the next `redistribute(Replicate)` all-gather. Cases 1 and 5 (colocate + 1 GPU/engine) are naturally synchronized since every rank is a source; the barrier is a no-op in practice.

---

## Weight Data Flow

In both paths, **slim always sends full un-sharded HF-format params**. The sglang engine is responsible for slicing them to fit its internal parallelism layout (TP/PP/DP/EP). Slim never needs to know how the engine splits its GPUs internally.

### Separate (`UpdateWeightFromDistributed`)

```
Training rank 0            All sglang engine workers
     │                      │   │   │   │
     │  NCCL broadcast      │   │   │   │
     │  (full param)        │   │   │   │
     ├─────────────────────►│   │   │   │
     ├─────────────────────────►│   │   │
     ├─────────────────────────────►│   │
     ├─────────────────────────────────►│
     │                      │   │   │   │
                            each worker keeps only
                            the slice it needs
```

- Training rank 0 has the full param (after `redistribute(Replicate)`)
- Rank 0 does `dist.broadcast` via a dedicated NCCL group to **all** sglang workers across all engines
- Each sglang worker receives the full param and internally keeps only the slice it needs based on its placement

### Colocate (`UpdateWeightFromTensor`)

```
Training ranks (IPC group)       sglang engine process
  r0  r1  r2  r3                      │
  │   │   │   │   Gloo gather         │
  │   │   │   ├──►r0                  │
  │   │   ├──────►r0                  │
  │   ├──────────►r0                  │
  │               │   Ray IPC         │
  │               ├──────────────────►│
  │               │  (full params)    │
                                      │
                              engine distributes
                              to its workers internally
```

- All training ranks in the IPC group have the full param (with FSDP they are all identical after `redistribute(Replicate)`)
- They `gather_object` (Gloo) to the source rank (first rank of the group)
- The source rank sends via Ray IPC (`update_weights_from_tensor`) to the sglang engine process
- The sglang engine internally distributes to its workers, each keeping what it needs
