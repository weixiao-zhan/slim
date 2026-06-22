# Placement: actor / critic / rollout

This document describes how slim places the **actor**, **critic**, and **rollout**
(sglang) processes onto GPUs.

## Two orthogonal switches

| switch | meaning | code today |
|---|---|---|
| --rollout-colocate | rollout engines share the **training** GPUs (time-shared via offload) | the current `--colocate` |
| --critic-colocate | critic shares the **actor** GPUs instead of taking its own | *new (proposed)* |

They are independent: RC controls whether the *rollout* segment overlaps
training; CC controls whether the *critic* segment overlaps the actor. Today
only RC exists, and critic is always on its own GPUs (parallel to the actor).

## Notation

Each role is declared at **two levels** — an overall GPU total and a
per-replica size; the replica count is derived. Same naming shape for all three:

| role | overall total | gpu per replica | # replicas (derived) |
|---|---|---|---|
| actor   | `actor_num_gpus` (`A`)   | `actor_num_gpu_per_replica`   | `A / actor_num_gpu_per_replica` |
| critic  | `critic_num_gpus` (`C`)  | `critic_num_gpu_per_replica`  | `C / critic_num_gpu_per_replica` |
| rollout | `rollout_num_gpus` (`R`) | `rollout_num_gpu_per_replica` | `R / rollout_num_gpu_per_replica` |

**Placement (which physical GPUs each role gets) depends only on the totals
`A` / `C` / `R`.** The per-replica size is sub-structure *inside* a role's
segment — it never moves segment boundaries (one exception in colocate, below).

### Sub-segment structure: what a replica means

A **replica** is the same concept for all three roles: the model is sharded
*within* a replica and data-parallel *across* replicas. Only the within-replica
strategy is backend-specific.

| role | within a replica | across replicas |
|---|---|---|
| actor / critic | FSDP shard | data-parallel (DDP all-reduce) |
| rollout | TP (a replica *is* one sglang engine) | independent, router-balanced (DP) |

The invariant: **across replicas is always data-parallel, regardless of
backend.** Only the **within-replica** strategy is backend-specific — FSDP
shards there; a future Megatron backend would put TP / CP / EP there. So the
precise statement is: within a replica it is FSDP full-shard, between replicas
it is *replication* (DDP), and the 2D combination of the two *is* what "hybrid
sharding (HSDP)" names. One knob spans the whole spectrum (shown for the
actor): `actor_num_gpu_per_replica == A` → a single replica, pure full-shard
FSDP; `actor_num_gpu_per_replica == 1` → `A` replicas, pure DDP; in between →
HSDP.

This sub-structure does **not** move the segment boundaries above — it only
subdivides a segment — *except* in the colocate combos (2 / 4), where each
rollout replica's (engine's) TP boundary (and, under CC, each critic replica)
must align with actor ranks for IPC weight updates.

### Proposed argument changes

| arg | status | replaces |
|---|---|---|
| `actor_num_gpus`              | new | `actor_num_nodes × actor_num_gpus_per_node` |
| `critic_num_gpus`             | new | `critic_num_nodes × critic_num_gpus_per_node` |
| `rollout_num_gpus`            | keep | (already the total) |
| `actor_num_gpu_per_replica`   | new | (was implied by `--fsdp-strategy`) |
| `critic_num_gpu_per_replica`  | new | (was implied by `--fsdp-strategy`) |
| `rollout_num_gpu_per_replica` | **rename** | `rollout_num_gpus_per_engine` |
| `--fsdp-strategy {full,hybrid}` | **retire** | folded into `*_num_gpu_per_replica` |

Net effect: each role goes from `{num_nodes, num_gpus_per_node, [fsdp_strategy]}`
to `{num_gpus, num_gpu_per_replica}` — a total and a replica size. The
logical node-split disappears (replica size, not node count, decides the
shard group); the physical `--num-gpus-per-node` used for packing / NUMA / port
layout stays.

> **Status:** `*_num_gpu_per_replica` is a *proposed* unification, not a current
> argument. Today the trainer mesh is hardcoded (`actor.py:_setup_device_mesh`):
> `fsdp_strategy=hybrid` pins the replica to one node (shard =
> `actor_num_gpus_per_node`, replicate across `actor_num_nodes`); otherwise a
> 1D mesh full-shards across all `A` GPUs. There is no per-replica argument
> yet, and the critic reuses the actor's mesh dims.
>
> **`--fsdp-strategy {full,hybrid}` should retire once `*_num_gpu_per_replica`
> lands.** It is exactly the two pinned points of that knob: `full` =
> `actor_num_gpu_per_replica == A` (single replica), `hybrid` =
> `actor_num_gpu_per_replica == actor_num_gpus_per_node` (one replica per node).
> The per-replica arg expresses both plus every value in between, so the
> strategy flag becomes redundant. More broadly, the replica size is
> backend-agnostic: it sets how many GPUs a replica spans, and the backend
> decides the within-replica strategy (FSDP shard today; TP / CP / EP if/when a
> Megatron backend is added) — so this abstraction survives a Megatron
> migration unchanged.

> **CC requires `C == A`.** The actor and critic are two separate
> `RayTrainGroup`s; each takes the first `world_size` reordered bundles of its
> placement-group slice (`actor_group.py`). Only when `C == A` does critic
> rank *i* land on the same physical GPU as actor rank *i*, which is what
> `connect_actor_critic()`'s rank-wise `zip` assumes. `C != A` must be
> rejected with an assert.

## Placement formula

A single PACK'd placement group is created, sorted by `(node_ip, gpu_id)`, then
sliced by per-role offsets (`placement_group.py:create_placement_groups`).
Adding CC generalizes the offsets to:

```
train_span     = A           if CC else A + C      # bundles the trainers occupy
critic_offset  = 0           if CC else A
rollout_offset = 0           if RC else train_span
R (forced)     = train_span  if RC else user-R     # RC forces rollout to fill training
PG total       = train_span  if RC else train_span + R
```

The 2×2 of (RC, CC):

| # | combo | RC | CC | PG total | critic_offset | rollout_offset | async-capable |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | disagg + parallel **(current default)**     | ✗ | ✗ | A+C+R | A | A+C | ✅ |
| 2 | colo-rollout + parallel **(current `--colocate`)** | ✓ | ✗ | A+C   | A | 0   | ❌ |
| 3 | disagg + **colo-critic** *(new)*            | ✗ | ✓ | A+R   | 0 | A   | ✅ |
| 4 | colo-rollout + **colo-critic** *(new)*      | ✓ | ✓ | A     | 0 | 0   | ❌ |

## Physical layout (A = C = 4, and R = 4 when disaggregated)

```
组合1  disagg+parallel   (12 GPU)
 0 1 2 3 | 4 5 6 7 | 8 9 10 11
 └actor──┘ └critic─┘ └rollout─┘          全部独占,互不重叠

组合2  colo-rollout+parallel   (8 GPU)   ← 现 --colocate
 0 1 2 3 | 4 5 6 7
 └actor──┘ └critic─┘
 └────── rollout ──┘                      rollout 叠满整个训练段 (R 强制=8)

组合3  disagg+colo-critic   (8 GPU)       ← 新, async 友好
 0 1 2 3 | 4 5 6 7
 └actor──┘ └rollout┘
 └critic─┘                                actor、critic 重叠同段; rollout 独立

组合4  colo-rollout+colo-critic   (4 GPU) ← 新, 最激进
 0 1 2 3
 └actor─┘
 └critic┘
 └rollout                                 三者全叠在同 4 张卡 (R 强制=4)
```

## Per-bundle resource quota

Ray's `num_gpus` is a **logical scheduling quota**, not a memory limit. Each
bundle is `{"GPU": 1, "CPU": 1}`; the sum of `num_gpus` of all actors packed
onto one bundle must be `<= 1.0`. Current quotas: training actor `0.4`
(`placement_group.py`), rollout engine `0.2` (`rollout.py`). Physical memory
isolation is *disabled* (`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`) so
every co-scheduled process sees the whole card; real memory de-confliction is
done by offload (`sleep`/`wake_up`) time-slicing.

| combo | quota on a **training** GPU | quota on a **rollout** GPU |
|---|---|---|
| 1 disagg+parallel | actor 0.4 (0.6 idle) / critic 0.4 | 0.2 |
| 2 colo-rollout     | actor 0.4 + rollout 0.2 = **0.6** | — |
| 3 colo-critic      | actor 0.4 + critic 0.4 = **0.8**  | 0.2 |
| 4 full colocate    | actor 0.4 + critic 0.4 + rollout 0.2 = **1.0** | — |

Combo 4 fills exactly `1.0` — the `0.4 / 0.4 / 0.2` split was chosen to make
three-way single-card colocation legal.

## Step timeline for parallel actor critic (combos 1 & 2)

When actor and critic are on **disjoint** GPUs they run **concurrently** — each
is its own NCCL world on its own cards, so there is no collision and no
hand-off. This is the current behavior (`train.py`): a compute phase and a
train phase, each running actor and critic in parallel with a barrier between.

```
                 GPU memory resident
phase              A      C      legend: ■ resident & active
──────────────────────────────────  ░ resident idle   · offloaded (CPU)
   actor ⊂ [0,A),  critic ⊂ disjoint [A,A+C)  — never the same card
1. compute         ■      ■      compute_log_probs (A) ∥ compute_values (C)
───────────────────┼──────┼──── ◀ barrier: ray.get(values_refs + logprobs_refs)
2. train           ■      ■      actor.train ∥ critic.train; actor reads values via Ray ref
───────────────────┼──────┼──── ◀ barrier: ray.get(actor.train); ray.get(critic_train_handle)
3. actor.update_wts ■     ░      only actor pushes weights → rollout engines
```

Notes:

- **A and C are both ■ throughout** — the visual opposite of the colo-critic
  timeline below, where they are never resident together. Concurrency is free
  here precisely because the GPUs are disjoint; that is exactly what
  colo-critic trades away.
- **No A↔C hand-off, no inter-trainer OOM.** Each card hosts exactly one
  trainer, so neither has to offload for the other.
- **The only offload is trainer↔rollout (the RC axis), and A & C move in
  lockstep.** Combo 1 (disagg): rollout is on its own GPUs, `offload_train` is
  optional/off, so both trainers stay resident across `generate` too — this is
  the async-capable combo. Combo 2 (colo-rollout): `offload_train` /
  `offload_rollout` are forced, so both trainers sleep during `generate` and
  wake for the step *together* (against rollout, never against each other).
- **`num_critic_only_steps` warm-up.** For `rollout_id < num_critic_only_steps`
  (or `--critic-train-only`), the actor compute/train is skipped entirely and
  the critic trains alone; the diagram above is the steady state after warm-up.
- **Critic still does not push weights** — same as colo-critic, only the actor
  (step 3) syncs to rollout engines.

## Step timeline under colocate-critic (combos 3 & 4)

When the critic shares the actor's GPUs, the actor and critic **cannot run
concurrently** — two NCCL worlds issuing collectives on one device will hang or
OOM. The current loop runs `compute_values` ∥ `compute_log_probs` and then
trains actor ∥ critic concurrently (`train.py`); under CC this must be
**serialized**, with an explicit critic-sleep before the actor wakes.

```
                 GPU memory resident
phase                A      C      legend: ■ resident & active
─────────────────────────────────   ░ resident idle   · offloaded (CPU)
1. critic.wake_up()  ·      ■
2. compute_values    ·      ■        critic forward → per-token values
3. critic.train      ·      ■        critic fwd/bwd/optim
4. critic.sleep() ───┼──────┼──── ◀ HANDOFF: must offload C before waking A
                     ·      ·            (else A+C both resident → OOM)
5. actor.wake_up()   ■      ·
6. compute_log_probs ■      ·        ref/actor-old log-probs (ref swaps in-proc)
7. actor.train       ■      ·        actor fwd/bwd/optim (reads values via Ray ref)
8. actor.update_wts  ■      ·        push weights → rollout engines
9. actor.sleep()     ·      ·        end of step
```

Notes:

- **Handoff (step 4) is the critical OOM point.** `critic.sleep()` does
  `model.cpu()` + `clear_memory()`; it must complete *before* `actor.wake_up()`
  (step 5). The current end-of-loop `offload_train()` is too late for this — CC
  needs an explicit inter-phase offload. The transient peak (critic still
  draining while actor allocates) is what to validate empirically first.
- **`values` survive the handoff.** The critic returns values as Ray
  object-store refs, so the actor reads them in step 7 even though the critic
  is now on CPU.
- **Critic does not push weights.** `update_weights()` is a no-op for the
  critic; only the actor (step 8) syncs to rollout engines. This is why the
  async weight-update path is unaffected by CC.
- **Combo 3 can run async** (rollout is on separate GPUs, so generation of
  step N+1 overlaps training of step N). Combo 4 cannot — rollout shares the
  training GPUs, same as combo 2. Async + critic additionally requires removing
  the `NotImplementedError(use_critic)` guard in `train_async.py`.
