# NeMo AutoModel Backend Plan

## Objective

Replace direct Hugging Face model construction and hand-managed FSDP2 model parallelism in the training backend with NeMo AutoModel.

The target training stack is:

$$
\text{FSDP2} + \text{expert parallelism} + \text{context parallelism}
$$

for packed mixed-modality Qwen3.5 and Qwen3.6 MoE VLM training. SGLang remains the rollout engine, `Episode` remains the rollout-to-training data contract, and the existing policy objectives remain the loss contract.

NeMo AutoModel still uses Hugging Face configurations, processors, checkpoint formats, and selected model components. This plan replaces the Hugging Face `AutoModelForCausalLM` and `AutoModelForImageTextToText` training path. It does not remove the `transformers` dependency.

## Scope Decisions

| Area | Initial decision |
|---|---|
| Model family | Qwen3.5 MoE VLM first, followed by Qwen3.6 MoE VLM |
| Model scale | Qualify 35B and 122B first, then 200B to 300B, then Qwen3.5-397B-A17B |
| Training parallelism | FSDP2 + EP + CP |
| Sequence format | One indexed BSHD pack per logical data replica |
| Packed attention | Contiguous block-diagonal CP |
| GDN | FLA packed CP with real per-document boundaries |
| Vision tower | Frame-level CP sharding when media volume warrants it |
| Rollout | Existing SGLang integration |
| Routing | Preserve rollout routing replay |
| Losses | Preserve GRPO, GSPO, PPO, KL, entropy, mismatch correction, and custom loss contracts |
| PP | Deferred; mixed-media PP+CP is outside the initial scope |
| TP and ETP | Outside the initial scope |
| Installed-package patching | Not used for model or distributed correctness |

EP+CP is the useful first target because the two dimensions address different limits:

- FSDP2 shards model and optimizer state.
- EP partitions MoE expert ownership and dispatches tokens to those experts.
- CP partitions sequence activations and attention work.

This combination can reach large MoE models and long trajectories without making TP or PP a prerequisite. The expected practical target is 100B to 300B MoE training at 64K context on a sufficiently large H100 or H200 cluster. Qwen3.5-397B-A17B at 64K is a qualification target, not an assumed capability.

TP and PP are not only useful for trillion-parameter models. TP reduces the per-rank footprint and compute of individual dense operations, while PP reduces the number of layers resident and active on a pipeline stage. Both add communication, scheduling, checkpoint, loss, and multimodal transport complexity. They are deferred because FSDP2+EP has demonstrated large-model capacity and CP independently addresses long-sequence activations, not because TP or PP becomes useful only above a fixed parameter count.

The initial scope has the following capacity boundaries:

- FSDP2 shards parameters, gradients, and optimizer state across the FSDP mesh.
- EP reduces routed-expert residency and expert compute per rank. It does not shard embeddings, attention, shared experts, the vision tower, or other dense components.
- CP reduces sequence activation residency and distributes attention and GDN sequence work. It does not reduce model-state memory.
- More GPUs can increase all three dimensions, but communication topology, expert balance, activation checkpointing, optimizer precision, and actor-to-SGLang synchronization determine whether the result is usable.
- There is no model-size or context-length maximum derivable from EP and CP degrees alone.

## Current Slim Contracts

The migration keeps the framework-level contracts that are independent of the training model implementation.

### Episode

`Episode` remains the boundary between rollout and training. It carries:

- Full prompt and response tokens.
- Edge-aligned loss masks.
- Rollout log probabilities.
- Advantages, returns, and optional values.
- Rollout-selected experts.
- Processor-produced image and video tensors.

The data contract is documented in [Data Layout](data-layout.md). Edge-aligned field entry $i$ describes the prediction:

$$
p(x_{i+1} \mid x_{\leq i})
$$

### Packing

The existing packing implementation in `slim/backends/fsdp_utils/data_packing.py` already provides:

- Balanced assignment of episodes to packs.
- Global `cu_seqlens`.
- Token-aligned and edge-aligned field handling.
- Concatenation of mixed text, image, and video episodes.
- Per-document media counts.
- Stable media ordering.

The NeMo backend should consume the same episodes but produce a NeMo-native packed batch. It should not route NeMo execution through the Hugging Face-specific model argument builder.

### Mixed Modality

Processor outputs are generated once and retained in `Episode.multimodal_inputs`. The same processor result is used by training and SGLang rollout. This contract prevents token and visual-placeholder drift.

Mixed modality means a physical pack may contain any sequence of:

- Text-only episodes.
- Episodes with one or more images.
- Episodes with one or more videos.
- Episodes containing both image and video inputs when the model processor supports them.

Media tensors remain ragged and are concatenated in document order. Per-document item counts provide the offsets required to reconstruct the media slice for mRoPE and vision embedding.

### Routing Replay

SGLang records expert choices as:

```text
[num_edges, num_layers, top_k]
```

The training model must replay those choices while gathering fresh router probabilities so router weights remain differentiable. Replay must stay active during the original forward and activation-checkpoint recomputation in backward.

### Losses

The built-in policy loss and custom loss interface operate on per-episode edge-aligned tensors. The NeMo backend must preserve:

- Per-token and per-sample reduction.
- Sequence-level GSPO ratios.
- Rollout and actor-old log probability selection.
- Mismatch metrics and correction.
- Reference-model KL.
- Optional entropy.
- Critic value loss for PPO-GAE.

## NeMo Support Assessment

### Upstream State

NeMo AutoModel main contains the generic block-diagonal varlen CP engine from [PR #2989](https://github.com/NVIDIA-NeMo/Automodel/pull/2989).

The Qwen VLM integration is in the stacked draft [PR #3186](https://github.com/NVIDIA-NeMo/Automodel/pull/3186). It depends on:

- [PR #2937](https://github.com/NVIDIA-NeMo/Automodel/pull/2937), unified CP input preparation and token sharding verbs.
- [PR #2989](https://github.com/NVIDIA-NeMo/Automodel/pull/2989), block-diagonal varlen CP for packed sequences.
- [PR #2990](https://github.com/NVIDIA-NeMo/Automodel/pull/2990), frame-level CP vision-tower sharding.

PR #3186 is explicitly a draft and should not be treated as a released interface. Its exact revision is nevertheless the best implementation baseline because it contains the complete model-level design and end-to-end validation.

This assessment uses PR #3186 revision [`c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2`](https://github.com/NVIDIA-NeMo/Automodel/tree/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2). Phase 0 must record the revision actually selected for implementation because the stacked branch can be rebased.

The code-level assessment is based on the pinned revision's [Qwen3.5 MoE model integration](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/nemo_automodel/components/models/qwen3_5_moe/model.py), [packed GDN CP implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/nemo_automodel/components/models/qwen3_5_moe/cp_linear_attn.py), [block-diagonal CP sharder](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/nemo_automodel/components/distributed/blockdiag_cp/batch.py), and [unified token sharding verbs](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/nemo_automodel/components/distributed/cp_sharder.py). These paths, not the presence or absence of a ready-made recipe, determine the matrix below.

The draft reports a 100-step run with:

- Qwen3.5-122B-A10B.
- 16 nodes and 128 H100 GPUs.
- EP8, CP16, and inferred DP8.
- 128K neat-packed sequences.
- Trainable vision tower.
- Real image and video inputs.
- Full activation checkpointing.
- Approximately 64.2 GiB peak allocation per GPU.
- Approximately 49.53 tokens/s/GPU and 6,340 tokens/s aggregate throughput.

This result establishes feasibility for packed mixed-modality EP+CP. It does not establish production performance or correctness for Qwen3.5-397B-A17B or Qwen3.6 checkpoints.

### Scale Evidence

The available runs isolate different scaling questions. They must not be combined into a capacity claim that no single run demonstrates.

| Model and run | Topology | Sequence | What it establishes |
|---|---|---:|---|
| [Qwen3.5-122B-A10B VLM integration](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/examples/vlm_finetune/qwen3_5_moe/qwen3_5_122b_128k_ep8cp16.yaml) | 128 H100, FSDP2, EP8, CP16, PP1, TP1 | 128K packed | Packed mixed image/video EP+CP, trainable vision, and hybrid full-attention/GDN execution |
| [Nemotron-3-Ultra-550B-A55B benchmark](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/examples/llm_benchmark/nemotron/nemotron_ultra_v3_te_deepep.yaml) | 128 H100, FSDP2 over 128 ranks, EP64, CP1, PP1, TP1 | 4K | Very large MoE parameter residency without TP or PP; it does not provide long-context evidence |
| [Qwen3.5-397B-A17B VLM recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml) | 256 H100, FSDP2, EP32, PP8, CP1, TP1 | 2K | Existing full-model mixed-modality recipe uses PP at this scale; it does not show that EP+CP alone fits 397B |
| [Qwen3.6-35B-A3B VLM parity run](https://github.com/NVIDIA-NeMo/Automodel/blob/c4d3ed711dea41a6ff897752cc66bf1c0ccd73e2/examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_medpix_ep8cp2_4k.yaml) | 8 H100, FSDP2, EP8, CP2, PP1, TP1 | 4K non-packed | Qwen3.6 hybrid VLM EP+CP model path; packed long-context behavior remains unqualified |

The 122B run is the strongest evidence for the target data and CP design. The 550B run is evidence that FSDP2+EP can cover very large sparse parameter sets, but its 4K sequence does not answer the 64K activation question. A 200B to 300B MoE VLM at 64K is therefore a reasonable scale-qualification target given enough H100 or H200 GPUs, not a guaranteed configuration. The first sizing run must measure model-state headroom separately from sequence activation and K/V communication headroom.

These are supervised fine-tuning or training benchmarks, not on-policy RL runs. Slim also needs capacity for the reference model when enabled, optimizer state, rollout engines, packed RL metadata, checkpoint staging, and actor-to-SGLang weight transfer. Cluster sizing must include those costs rather than extrapolating directly from the upstream GPU counts.

### Support Matrix

The implementation-status column describes code paths, including the pinned stacked draft. Qualification means an end-to-end run exists for the relevant model and combination.

| Capability | Code support and evidence | Migration decision |
|---|---|---|
| Dense model + FSDP2 | Supported | Preserve through the NeMo backend |
| Dense model + CP | Supported for model-specific attention paths | Add after the MoE VLM target |
| MoE + EP | Supported | Required |
| MoE + CP | Model-specific | Required for Qwen hybrid attention/GDN |
| MoE + EP+CP | Implemented and validated by the Qwen3.5-122B draft | Required |
| MoE VLM + EP+PP at CP1 | Implemented; the Qwen3.5-397B recipe uses EP32+PP8 | Existing upstream option, outside the initial backend |
| VLM without packing + CP | Model-specific | Supported where the selected NeMo model declares it |
| Qwen VLM indexed packing at CP1 | Implemented | Required parity baseline |
| TE THD packed VLM + CP | Rejected for current mRoPE VLM path | Not used |
| Indexed BSHD packed VLM + block-diagonal CP | Implemented by the draft Qwen integration | Required |
| Packed GDN + CP | Implemented by the draft using FLA CP | Required |
| Trainable vision + CP | Implemented by frame-level vision sharding | Required at scale |
| Mixed-media EP+CP | Validated by the draft at 122B | Required |
| Text-only Qwen PP+CP | Permitted by the Qwen model path and covered by generic PP+CP infrastructure | Deferred and requires slim parity tests |
| Mixed-media Qwen PP+CP | Explicitly rejected in the Qwen forward when the first PP stage must carry image/video data under CP | Deferred |
| Text-only EP+PP+CP | The dimensions compose in the NeMo mesh and Qwen guard, but no target-scale Qwen run qualifies the combination | Deferred |
| Mixed-media EP+PP+CP | Blocked by the same Qwen PP+CP media guard | Deferred |
| TP+EP+CP | Outside the validated Qwen path | Out of scope |
| ETP | Not required for the target scale | Out of scope |
| Routing replay | NeMo primitive exists; slim rollout replay is not qualified with packed CP | Required integration work |
| Qwen3.6 MoE VLM | Non-packed EP8+CP2 is implemented and exercised at 35B; packed long-context EP+CP is unqualified | Qualification target |
| Qwen3.5-397B-A17B | No equivalent published packed EP+CP run | Qualification target |

The mixed-media PP+CP rejection is a current Qwen model-integration constraint, not a mathematical incompatibility between VLMs, PP, and CP. NeMo has other VLM PP+CP paths and shared PP media transport. The Qwen code rejects this case because media embedding and sequence CP sharding occur on the first pipeline stage while the current microbatch media channel does not implement the required combined layout. Supporting it requires model and pipeline work beyond changing a recipe.

## Tensor Layouts

### BSHD

BSHD represents attention tensors as:

```text
[batch, sequence, heads, head_dim]
```

The target uses an indexed BSHD input with physical batch size one. Document identity is carried separately for every token. The single physical sequence may contain many logical episodes.

### THD

THD represents packed attention tensors as:

```text
[total_tokens, heads, head_dim]
```

`cu_seqlens` identifies document boundaries. TE can context-shard THD inputs with load-balanced partition indices, but the current VLM recipe rejects THD packing with CP because the Qwen mRoPE VLM path is not composed with that sharder.

### Target Layout

The public batch remains indexed BSHD. Full-attention kernels may internally consume varlen metadata equivalent to THD, but the model-facing pack is not routed through the rejected TE THD VLM recipe.

This distinction is important:

- The TE THD rejection is an integration restriction.
- It is not a fundamental incompatibility between packing, VLMs, EP, and CP.
- The block-diagonal BSHD path supplies the missing document-aware CP semantics.

## Context Parallel Implementations

NeMo contains several CP mechanisms. They are not interchangeable.

### Standard Non-Packed CP

The standard PyTorch or TE path uses a load-balanced head-tail sequence partition. TE can use P2P ring communication or all-gather depending on the attention module and configuration.

This path assumes one causal document. Dropping the explicit packed mask would allow attention across document boundaries, so it is not valid for the target pack.

### TE THD CP

TE THD CP partitions every document according to `cu_seqlens` and is a strong path for supported LLMs. Current Qwen VLM mRoPE preparation is not composed with it. The migration should not remove the guard and silently run this combination.

### Block-Diagonal Packed CP

The target full-attention path uses:

- A contiguous query shard per CP rank.
- Replicated global document IDs.
- Per-document causal masking.
- Differentiable K/V exchange across the CP group.
- FlashAttention or TE varlen kernels when available.
- Dense SDPA only as a correctness fallback.

The default K/V exchange is full all-gather. Node-local alternatives can send only needed K/V through halo or all-to-all-v exchange. Cross-node needed-only exchange currently falls back to all-gather unless explicitly enabled.

This path is not Ulysses. It does not partition attention heads and therefore does not introduce a head-count divisibility requirement. It is also not the standard TE causal ring path.

For a single long document spanning all CP ranks, later query shards require most preceding K/V. Full K/V communication can therefore remain a performance limit even when memory fits. The 122B at 128K result proves execution feasibility, not optimal communication scaling.

### Gated DeltaNet CP

Qwen3.5 and Qwen3.6 hybrid models alternate full-attention and Gated DeltaNet layers. GDN has different CP requirements:

- Causal convolution needs the preceding kernel-width tokens.
- Recurrent state must flow in sequence order.
- Convolution and recurrent state must reset at every document boundary.

The target path gives each rank a contiguous sequence interval and builds the FLA CP context from the real global `cu_seqlens`. It does not model the entire physical pack as one document. Packed GDN therefore avoids cross-document state leakage without a new GDN kernel.

Full attention and GDN must consume the same global document boundaries and the same padded sequence layout.

## Packed Batch Contract

Each logical data replica owns one global pack. Every rank in its CP group receives the same pack before CP sharding.

Let $S$ be the original physical pack length. The sharder pads to:

$$
S_{\text{padded}} \equiv 0 \pmod{2C}
$$

where $C$ is the CP degree.

The batch contract is:

| Field | Shape before CP | Meaning |
|---|---|---|
| `input_ids` | `[1, S]` | Full concatenated episode tokens |
| `labels` | `[1, S]` | Next-token targets within each document; `-100` at document ends and padding |
| `_packed_seq_ids` | `[1, S]` | One-based document ID; zero for padding |
| `position_ids` | `[3, 1, S]` | Per-document Qwen mRoPE positions |
| `loss_mask` | `[1, S]` | Token-slot form of the edge loss mask |
| `rollout_log_probs` | `[1, S]` | Token-slot rollout log probabilities |
| `actor_old_log_probs` | `[1, S]` | Token-slot actor-old log probabilities |
| `ref_log_probs` | `[1, S]` | Token-slot reference log probabilities |
| `advantages` | `[1, S]` | Token-slot advantages |
| `returns` | `[1, S]` | Token-slot returns |
| `old_values` | `[1, S]` | Optional token-slot critic values |
| `rollout_routed_experts` | `[1, S, layers, top_k]` | Token-slot expert selections |
| media tensors | Ragged | Concatenated in document order |
| media counts | Per document | Offsets into each media tensor |

### Edge-to-Token Mapping

For a document with tokens:

$$
x_0, x_1, \ldots, x_{n-1}
$$

the model input retains all $n$ tokens. At source position $t$:

$$
\text{labels}[t] =
\begin{cases}
x_{t+1}, & t < n - 1 \\
-100, & t = n - 1
\end{cases}
$$

Every edge-aligned RL field is written to source positions $0$ through $n-2$. Position $n-1$ receives the field's ignored fill value.

This representation:

- Eliminates synthetic cross-document edges.
- Handles a next-token target whose source position is on a CP boundary.
- Lets every RL tensor use the same CP token index map.
- Avoids gathering full vocabulary logits.

### Packing Policy

The initial implementation should:

- Use one physical pack per microbatch and local batch size one.
- Keep document order stable after balanced assignment.
- Pad only the physical pack tail.
- Permit documents to cross CP rank boundaries.
- Permit one document to span the entire CP group.
- Preserve mixed text, image, and video order.
- Compute mRoPE independently for each logical document before sharding.
- Reject malformed placeholder/media-count combinations before entering distributed collectives.

## Distributed Topology

Current slim treats every training rank as a data-parallel rank. That is incorrect once CP is enabled.

The NeMo device mesh defines:

- A logical DP coordinate used to select rollout data.
- A CP coordinate used to shard one logical sample.
- An EP mesh composed over the model/FSDP mesh.
- A flattened FSDP shard mesh that may include DP-shard and CP dimensions.

For the validated 128-GPU configuration:

$$
128 = DP8 \times CP16
$$

EP8 is composed over that mesh and is not an additional factor in the world-size product.

The rollout manager must create one training partition per logical DP rank. All CP ranks with the same DP coordinate read the same partition and build the same pack. Different logical DP coordinates read different episode partitions.

Batch and loss normalization must use logical DP size, not global world size.

## Forward and Backward Flow

The target training step is:

1. Resolve the logical DP partition and build one global pack.
2. Convert every edge-aligned RL field into token slots.
3. Construct per-document labels, document IDs, and mRoPE.
4. Resolve NeMo's `ContextParallelismSharder`.
5. Shard auxiliary token tensors with the sharder token verb.
6. Encode image and video frames, using CP vision sharding when enabled.
7. Scatter visual embeddings into the full token embedding stream.
8. Contiguously shard the resulting embeddings.
9. Run block-diagonal full attention with per-document causal masking.
10. Run GDN with the same global `cu_seqlens`.
11. Run EP dispatch on the CP-local valid tokens.
12. Compute local target log probabilities from local logits and local labels.
13. Gather only token scalar outputs when a sequence-level loss requires the full logical document.
14. Compute policy, value, entropy, KL, mismatch, or custom losses.
15. Backpropagate with CP-aware normalization.
16. Step the optimizer and synchronize updated actor weights to SGLang.

## Loss and Gradient Design

### Log Probabilities

Each CP rank computes selective log probabilities for its local labels:

$$
\log p_\theta(\text{label}_t \mid x_{\leq t})
$$

The implementation must not gather `[S, vocab]` logits. NeMo's sharder token verbs can differentiably gather `[S]` log probabilities into global token order.

The first production implementation should gather token scalars when required to preserve:

- Existing per-episode unpacking.
- GSPO sequence ratios.
- Custom loss functions.
- Mismatch correction.
- Per-sample reporting.

Local-only reduction can be added after parity is established.

### Entropy

Entropy still requires a vocabulary reduction on each local token. The full vocabulary result remains local; only the resulting entropy scalar per token is gathered.

### CP Consumer Multiplicity

A differentiable CP gather has sum semantics in backward. If every CP rank consumes the same gathered global loss, the loss scaling must compensate for the number of consumers and for FSDP's gradient reduction semantics.

No scaling formula should be accepted from inspection alone. CP1 and CP2/CP4 gradient parity tests must determine and pin the exact normalization for:

- Per-sample mean loss.
- Per-token loss.
- Gradient accumulation.
- Reference KL.
- Entropy.
- MoE auxiliary loss.
- Critic value loss.

### Custom Losses

The custom loss API should continue receiving per-episode dictionaries. The NeMo adapter reconstructs those dictionaries from gathered scalar token fields, not from gathered hidden states or vocabulary logits.

## Routing Replay Design

The rollout routing tensor is converted from edge alignment to token slots before CP. The same sharder used for labels and advantages shards routing replay.

Each local router receives:

```text
[local_valid_tokens, num_moe_layers, top_k]
```

The adapter must:

- Map NeMo router instances to the corresponding model layer.
- Select the local token rows in the model's exact flattening order.
- Preserve fresh routing probabilities and replay only expert indices.
- Keep replay active until `loss.backward()` has completed.
- Make replay state visible during activation-checkpoint recomputation.
- Clear state after every microbatch and on exceptions.
- Reject a layer count, token count, or `top_k` mismatch before expert dispatch.

Routing replay requires parity tests with EP and CP active together. The upstream 122B packed run does not replace this slim-specific acceptance requirement.

## Vision and mRoPE Design

Qwen mRoPE depends on the token placeholders and each media item's grid. Position construction must happen from the global document view.

For every document:

1. Slice its token IDs from the physical pack.
2. Slice its image and video grids using per-document media counts.
3. Build modality token-type IDs.
4. Call the model's mRoPE index function.
5. Append the resulting position axes in pack order.

Vision embeddings are scattered before language-sequence CP sharding.

Without vision sharding, every CP rank repeats the full vision tower. This is correct but wastes compute and activation memory. The frame-level sharder assigns independent image or video frames to CP ranks, computes local vision outputs, and differentiably gathers them in original order before placeholder scatter.

The initial path should support both:

- Replicated vision for a correctness baseline.
- Frame-sharded vision for production scale.

## Model Construction

The actor, reference model, and critic should use NeMo-native construction and distributed setup.

The target actor flow is:

1. Load the Hugging Face configuration and processor.
2. Build `NeMoAutoModelForImageTextToText` from the selected checkpoint.
3. Configure the NeMo backend for Qwen full attention, GDN, and MoE experts.
4. Build the FSDP2, EP, and CP meshes.
5. Apply activation checkpointing after model parallelism is configured.
6. Load checkpoint tensors through NeMo's state-dict adapter.
7. Build the optimizer over the final sharded parameters.

The reference model should use the same model implementation and mesh semantics so actor-old, reference, and current log probabilities have comparable numerics.

### Critic

The actor path is the first milestone. GRPO and GSPO do not require a critic.

PPO-GAE requires a NeMo-native scalar value head, checkpoint handling for that head, CP token gathering for values, and value-loss gradient parity. Critic support is a separate completion gate and should not block the initial actor-only EP+CP milestone.

## Checkpointing

Checkpoint support must cover:

- NeMo FSDP2 distributed checkpoints.
- Optimizer and scheduler state.
- EP-sharded grouped experts.
- CP-independent model parameter layout.
- Qwen3.5 and Qwen3.6 checkpoint naming.
- MTP expert layout when present.
- Added critic value head.
- Consolidated HF-compatible export where required.

Checkpoint resume must reproduce loss and routing behavior, not only load without error.

## SGLang Weight Synchronization

NeMo's Qwen state-dict adapter can expose HF-compatible parameter names and layouts. That should be the source contract for SGLang updates.

The current weight updater calls `Replicate()` on each DTensor before transmission. At 300B to 400B scale, replicating a grouped expert tensor on every training rank can exhaust memory and create unnecessary communication.

The NeMo updater must:

- Iterate state-dict tensors in bounded buckets.
- Convert NeMo names and grouped layouts to the HF/SGLang contract.
- Gather each tensor only to ranks that transmit it.
- Stream expert shards without materializing all experts on every rank.
- Preserve tied embedding and language-head semantics.
- Support BF16 and configured inference quantization.
- Keep colocated CUDA IPC and separate NCCL transport modes.
- Verify every SGLang worker receives the shard it owns.
- Release gathered storage before processing the next bucket.

For Qwen3.5-397B-A17B, weight synchronization is at least as important as the training forward. Model execution fitting in memory does not imply that full-tensor synchronization fits.

## Dependency Strategy

Core model and distributed semantics should not be implemented by editing installed NeMo source with exact-text replacement.

The dependency strategy is:

1. Pin an exact NeMo AutoModel Git revision with `uv`.
2. Use the stacked Qwen integration revision until its prerequisite PRs land.
3. Move the pin to the merged integration revision after focused parity tests pass.
4. Maintain a small NeMo fork only when the upstream revision cannot be consumed directly.
5. Keep slim-owned code limited to RL data, loss, replay, checkpoint, and SGLang synchronization concerns.

The environment must qualify:

- Python 3.12.
- PyTorch 2.11 and CUDA 13.0.
- `transformers==5.12.1`.
- Transformer Engine required by the selected NeMo revision.
- `flash-linear-attention>=0.4.2`.
- A causal-conv1d build compatible with the active PyTorch and CUDA ABI.
- FlashAttention for the block-diagonal varlen fast path.

The current causal-conv1d wheel source names a PyTorch 2.10 build while slim uses PyTorch 2.11. The NeMo environment phase must resolve and test that ABI combination before model work proceeds.

## Backend Ownership

The target module boundary is:

```text
slim/backends/nemo_utils/
    arguments and capability validation
    device mesh and logical DP mapping
    actor and reference model lifecycle
    critic lifecycle
    packed RL batch adapter
    CP loss and metric reduction
    routing replay adapter
    checkpoint adapter
    SGLang weight updater
    Qwen acceptance helpers
```

NeMo owns:

- Model architecture.
- Attention and GDN kernels.
- EP token dispatch.
- CP attention transport.
- Vision frame sharding.
- FSDP2 application.
- Model state-dict conversion.

Slim owns:

- Episode semantics.
- Rollout partitioning.
- RL field alignment.
- Policy and value objectives.
- Rollout routing replay input.
- Actor/reference/critic coordination.
- SGLang weight synchronization.
- Training metrics and lifecycle.

## Configuration and Validation Rules

The first supported configuration is equivalent to:

```yaml
training_backend: nemo
distributed:
  strategy: fsdp2
  tp_size: 1
  pp_size: 1
  ep_size: 8
  cp_size: 2
  sequence_parallel: false
model:
  backend:
    attn: sdpa
packed_sequence:
  enabled: true
  local_batch_size: 1
```

The exact user-facing argument schema should follow the repository's existing argument system. The backend must validate the resulting semantics:

- `world_size` must satisfy the NeMo mesh.
- `local_batch_size` must be one for packed block-diagonal CP.
- The number of experts must be divisible by EP size.
- Packed Qwen CP requires the model-owned block-diagonal attention route.
- Mixed-media CP requires global mRoPE preparation.
- Mixed-media `pp_size > 1` with `cp_size > 1` is rejected by the initial Qwen path.
- `tp_size > 1`, ETP, and all PP modes are rejected by the initial slim backend even where upstream NeMo has partial support.
- Every CP group must agree on pack length, document IDs, and media metadata.
- Tail padding must be ignored by labels, RL fields, routing, and MoE statistics.

## Implementation Phases

### Phase 0: Dependency and Baselines

Deliverables:

- Pin the exact NeMo revision.
- Resolve PyTorch, Transformer Engine, FLA, FlashAttention, and causal-conv1d compatibility.
- Record CP1 outputs from the current backend for small Qwen3.5 dense and MoE VLM fixtures.
- Record actor log probabilities, loss, gradients, checkpoint keys, and SGLang outputs.

Exit criteria:

- The NeMo model loads and runs CP1 forward/backward.
- The environment can run activation checkpointing and FLA GDN.

### Phase 1: NeMo Actor and Reference Backend

Deliverables:

- NeMo-native actor and reference model construction.
- FSDP2 optimizer and model lifecycle.
- CP1 mixed-modality packing.
- GRPO and GSPO without CP.
- NeMo checkpoint save and resume.

Exit criteria:

- CP1 packed log probability, loss, and gradient parity against the current backend.
- Reference KL and actor-old recomputation parity.

### Phase 2: Expert Parallelism

Deliverables:

- EP mesh configuration.
- Qwen grouped-expert checkpoint loading.
- EP dispatcher selection.
- MoE auxiliary loss and load statistics.
- EP-aware state-dict export.

Exit criteria:

- EP1 and EP2/EP4 forward and gradient parity.
- No padding tokens enter expert load statistics.

### Phase 3: Packed Context Parallelism

Deliverables:

- Global indexed pack contract.
- Logical DP data assignment.
- NeMo CP sharder integration.
- Block-diagonal full attention.
- Packed GDN with real document resets.
- Mixed image/video mRoPE.
- Replicated and frame-sharded vision modes.

Exit criteria:

- CP2 and CP4 parity for text-only, image-only, video-only, and mixed packs.
- Correct documents that end before, at, and after a CP boundary.
- Correct single documents spanning every CP rank.

### Phase 4: RL and Routing Completion

Deliverables:

- Token-slot conversion for every RL field.
- Differentiable scalar gather.
- Built-in policy loss.
- Custom loss.
- Entropy, KL, mismatch correction, and metrics.
- Routing replay through activation checkpointing.

Exit criteria:

- CP1 and CP2/CP4 gradient parity for every supported loss mode.
- EP+CP routing replay parity with the rollout-selected experts.
- No stale replay state after exceptions or microbatch transitions.

### Phase 5: SGLang Synchronization

Deliverables:

- NeMo-to-HF parameter naming.
- Grouped expert streaming.
- Colocated and separate update paths.
- Quantized update compatibility.
- Update equality checks.

Exit criteria:

- SGLang output parity before and after a no-op update.
- Updated SGLang outputs match the NeMo actor for a fixed prompt.
- Peak synchronization memory remains bounded by the configured bucket size plus one converted tensor.

### Phase 6: Critic and PPO-GAE

Deliverables:

- NeMo scalar value head.
- Value checkpoint and optimizer groups.
- CP value gathering.
- PPO-GAE actor/critic lifecycle.

Exit criteria:

- Value and value-gradient parity at CP1 and CP2.
- Actor and critic colocate modes complete without collective conflicts.

### Phase 7: Scale Qualification

Qualification order:

1. Qwen3.5-35B MoE VLM at 4K to 16K.
2. Qwen3.5-122B-A10B at 64K and 128K.
3. A 200B to 300B MoE checkpoint at 64K.
4. Qwen3.6 target checkpoint.
5. Qwen3.5-397B-A17B at 64K.

Each qualification records:

- Peak allocated and reserved memory.
- Tokens per second per GPU.
- Attention and EP communication time.
- Vision tower time.
- Optimizer-step time.
- Weight synchronization time and memory.
- Loss and gradient finiteness.
- Checkpoint save and resume.
- SGLang output parity.

## Test Matrix

### Unit Tests

- Edge-to-token conversion.
- Label construction at document boundaries.
- CP tail padding.
- mRoPE media offsets.
- Image/video concatenation order.
- Routing replay layer and token indexing.
- NeMo-to-HF state-dict name mapping.
- Capability rejection messages.

### Two-to-Eight GPU Tests

- CP2 and CP4 forward parity.
- CP2 and CP4 gradient parity.
- EP2 and EP4 parity.
- EP2+CP2 smoke.
- Full-attention document isolation.
- GDN convolution and recurrent reset.
- Activation-checkpoint recomputation.
- All-padding local shard.
- One document spanning all ranks.
- Documents with boundaries on every rank.
- Vision replicated versus frame-sharded parity.

### RL Tests

- GRPO with per-sample reduction.
- Per-token reduction.
- GSPO full-sequence ratio.
- PPO clip, IS, TIS, and CIS.
- Reference KL estimators.
- Entropy.
- Rollout versus actor-old baselines.
- Mismatch correction.
- Custom loss.
- Critic value clipping.

### System Tests

- Mixed text/image/video rollout to training.
- Routing capture in SGLang and replay in NeMo.
- Actor checkpoint resume.
- Actor-to-SGLang update.
- Colocated weight update.
- Separate weight update.
- Engine restart followed by full refresh.

## Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Upstream Qwen integration is a stacked draft | Interface and commit history can change | Pin an exact revision and isolate NeMo calls behind the backend boundary |
| GDN receives incorrect document boundaries | Cross-document recurrent leakage | Share one global `cu_seqlens` source with attention and add reset parity tests |
| Incorrect CP loss scaling | Silent gradient magnitude error | Require CP1 versus CP2/CP4 gradient parity for every loss reduction |
| Current weight update replicates large DTensors | OOM during actor-to-SGLang update | Implement source-targeted streaming gather |
| Cross-node K/V all-gather is slow | Poor 64K utilization | Measure first; qualify topology-aware halo/A2A only after correctness |
| Media ordering changes during packing | Incorrect visual embeddings and mRoPE | Keep stable document order and validate placeholder counts before collectives |
| Qwen3.6 checkpoint layout differs | Load or update mismatch | Add checkpoint-key and numerical acceptance tests per checkpoint |
| Routing replay differs during recomputation | Activation checkpoint error or expert drift | Keep replay active through backward and test recomputation |
| FLA or causal-conv1d ABI mismatch | Import or kernel failure | Resolve the environment before model integration |
| Critic head is not represented by the NeMo adapter | PPO-GAE checkpoint failure | Treat critic as a separate implementation phase |

## Effort Estimate

With the pinned upstream Qwen draft:

- Functional slim actor prototype: approximately 2 to 3 engineer-weeks.
- Production EP+CP actor with all RL losses, routing replay, checkpointing, and SGLang synchronization: approximately 5 to 8 engineer-weeks.
- Critic and PPO-GAE: additional focused work within the production window or a follow-up milestone.
- Multi-node 64K and 128K qualification: additional cluster and debugging time.

A clean-room attention, GDN, and vision CP implementation is outside the recommended scope because it duplicates the upstream design and substantially increases correctness risk.

## Completion Criteria

The migration is complete when:

- Direct Hugging Face AutoModel construction is absent from actor, reference, and supported critic training paths.
- CP1 packed VLM behavior matches the existing backend.
- Packed mixed-modality EP+CP passes forward and gradient parity.
- GDN and attention preserve every document boundary.
- All supported RL fields and losses remain aligned.
- Routing replay survives activation checkpointing.
- Checkpoint save and resume are deterministic within the accepted numerical tolerance.
- SGLang weight synchronization is HF-compatible and memory-bounded.
- The selected production checkpoint completes a 64K multi-node training and update cycle.
- Out-of-scope TP, ETP, and PP combinations fail before training begins.
