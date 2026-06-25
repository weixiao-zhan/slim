# Placement: actor / critic / rollout

This document describes how slim places the **actor**, **critic**, and **rollout** processes onto GPUs. Note the **actor** oversee both "policy" and "refernce" model and manage with inprocess offload since they share the same compute graph. 

## Notation

Each role declars **two levels** of parallel: an overall GPU total and a
per-replica size. 

| role | overall total | gpu per replica | # replicas (derived) |
|---|---|---|---|
| actor   | `--actor_num_gpus` (`A`)   | `--actor_num_gpus_per_replica`   | `A / actor_num_gpus_per_replica` |
| critic  | `--critic_num_gpus` (`C`)  | `--critic_num_gpus_per_replica`  | `C / critic_num_gpus_per_replica` |
| rollout | `--rollout_num_gpus` (`R`) | `--rollout_num_gpus_per_replica` | `R / rollout_num_gpus_per_replica` |

One **replica** is one copy of the sharded model and data-parallel *across* replicas.

| role | within a replica | across replicas |
|---|---|---|
| actor / critic | FSDP shard | data-parallel (DDP all-reduce) i.e. HSDP |
| rollout | one sglang engine with  TP <br> and optional PP / EP / PD-disaggregation | router-balanced (DP) |


## Colocate

**actor**, **critic**, and **rollout** processes can time share same GPUs with colocate.
Slim supports two independent colocate strategy.

| args | meaning |
|---|---|
| `--rollout-colocate` | rollout engines share the **training** GPUs (time-shared via offload) |
| `--critic-colocate` | critic shares the **actor** GPUs instead of taking its own |

## Placement

A single PACK'd Ray placement group (PG) is created over all nodes sorted by `(node_ip, gpu_id)`, then sliced by following per-role offsets (`placement_group.py:create_placement_groups`).

| # | rollout colocate | critic colocate | PG total | critic_offset | rollout_offset | async-capable | layout |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| 1 | ✗ | ✗ | A+C+R | A | A+C | ✅ | <pre>0 1    \| 2 3    \| 4 5 6 7<br>└actor─┘ └critic┘ └rollout┘</pre> |
| 2 | ✓ | ✗ | A+C   | A | 0   | ❌ | <pre>0 1 2 3 \| 4 5 6 7<br>└actor──┘ └critic─┘<br>└rollout──────────┘</pre> |
| 3 | ✗ | ✓ | A+R   | 0 | A   | ✅ | <pre>0 1 2 3 \| 4 5 6 7<br>└actor──┘ └rollout┘<br>└critic─┘</pre> |
| 4 | ✓ | ✓ | A     | 0 | 0   | ❌ | <pre>0 1 2 3<br>└actor──┘<br>└critic─┘<br>└rollout┘</pre> |

### Ray bundle resource quota

Ray abstract each GPU as one PG **bundle** (`{"GPU": 1, "CPU": 1}`). 
When roles colocate, several processes land on the same bundle, so each is given a fractional `num_gpus` quota and Ray only co-schedules them if the quotas sum to `<= 1.0`.
The quotas are **scheduling weights, not memory limits**: GPU memory isolation is disabled (`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`), so every co-scheduled process sees the whole card. 

Slim manage time-slicing with offload (`sleep`/`wake_up` premitive).
Slim set fixed quotas: actor `0.4`, critic `0.4`, rollout `0.2`, so all colocate placement are valid.

## Step timeline

### no critic-colocate (# 1 & 2)

When actor and critic are on **disjoint** GPUs they run **concurrently**. Each have their own NCCL world.

```
                 GPU memory resident
phase               A      C      
─────────────────────────────────
1. compute          ■      ■      ref/actor-old log-probs (A) ∥ compute_values (C)
────────────────────┼──────┼──── ◀ barrier: ray.get(values_refs + logprobs_refs)
2. train            ■      ■      actor.train (actor reads values via Ray ref) ∥ critic.train
────────────────────┼──────┼──── ◀ barrier: ray.get(actor.train); ray.get(critic_train_handle)
3. actor.update_wts ■      ░      actor pushes weights → rollout engines

legend:
■ active resident ░ idle resident · offloaded (CPU)
```

### under colocate-critic (# 3 & 4)

When the critic shares the actor's GPUs, the actor and critic **cannot run
concurrently**. Two NCCL worlds issuing collectives on one device will hang or
OOM.

```
                  GPU memory resident
phase                A      C 
─────────────────────────────────
1. critic.wake_up()  ·      ■
2. compute_values    ·      ■        critic forward → per-token values
3. critic.train      ·      ■        critic fwd/bwd/optim
4. critic.sleep() ───┼──────┼──── ◀ HANDOFF: must offload C before waking A
5. actor.wake_up()   ■      ·
6. compute_log_probs ■      ·        ref/actor-old log-probs (ref swaps in-proc)
7. actor.train       ■      ·        actor fwd/bwd/optim (reads values via Ray ref)
8. actor.update_wts  ■      ·        push weights → rollout engines
9. actor.sleep()     ·      ·

legend:
■ active resident ░ idle resident · offloaded (CPU)
```

---

## Weight Update

On-policy RL syncs model weights from the FSDP actor to the SGLang rollout engines at every training step.
Slim treats each engine as an opaque unit of `rollout_num_gpus_per_replica` GPUs.
How the engine internally splits those GPUs (TP/PP/DP/EP) is configured via sglang args and is transparent to the weight update path; the sglang engine discards any param slice it does not need.

![Weight sync paths](images/WeightSync.png)

### Colocate path

After training, the actor model is offloaded to CPU.
During weight update, params are streamed from CPU to GPU one bucket at a time: each param is moved to GPU, all-gathered across the FSDP mesh, pushed to the engine, then freed.
Only one bucket of params lives on GPU at a time, so the full model never needs to fit in GPU memory alongside the engine.

For each bucket, ranks serialize their tensors as CUDA IPC handles.
Gloo collects all handles to the source rank (first rank in the IPC group, e.g. A0, A2).
The source rank issues a single Ray RPC to the engine, passing all handles.
The engine dispatches each handle to its corresponding TP worker, which opens it from its colocated actor rank (zero-copy, same GPU).
The actual tensor data never crosses GPUs; only the small IPC handle metadata is gathered.

### Separate path

The same FSDP all-gather by buckets followed by 
Only A0 broadcasts to all engine TP workers directly.
Each TP worker receives the full param and keeps only its slice.

### sync with LoRA

Before weight sync, the adapter is merged into the base weights (`merge_adapter()`), synced as full HF params, then unmerged when wake up for training.
In colocate mode with CPU offload, the merge happens during `sleep()` before offloading.

### sync with low precision inference

A quantizer quantizes each bucket from BF16 to block-FP8 format between all-gather and send.
Rollout engines receive weights in the same quantized format they were initialized with.

### sync with failed engines

Slim also keeps a heart beat health check on all sglang engines and kills any non-responsive ones. 
The killed engine are restarted before weight sync and receive fresh memory and join next batch of training.
