# NeMo AutoModel Backend

## Scope

`slim.backends.nemo` is Slim's training backend. It combines Slim's Ray and RL lifecycle with NeMo AutoModel model construction, FSDP2, context parallelism, expert parallelism, optimizer utilities, and distributed checkpointing.

First-party support covers:

| Model | Training topology |
|---|---|
| Dense Qwen3.5 | FSDP2 and context parallelism |
| Qwen3.5 MoE | FSDP2, context parallelism, and expert parallelism |
| Qwen4, language model only | FSDP2, context parallelism, and expert parallelism greater than one |

Tensor parallelism, pipeline parallelism, and sequence parallelism are not supported by this backend.
Training updates either all model parameters or LoRA adapters on a frozen base, as [PEFT](#peft) describes.

SGLang remains the rollout engine. Slim owns `Episode` and `Trajectory` processing, RL objectives, reference and critic execution, routing replay, metrics, and policy weight synchronization.

## Patches

AutoModel 0.7.0 provides Qwen3.5 token sharding, block-diagonal attention, CP-aware Gated DeltaNet, and CP vision frame sharding.
`dependency_patches/nemo_automodel.py`, applied by `patch_dependencies.py`, enables the CP1 batch context, singleton K/V gather, packed Gated DeltaNet dispatch at CP1, FLA's varlen short convolution for packed Gated DeltaNet outside CP, native contiguous primary sharding for dense and MoE, the original `config.json` in consolidated exports, and dynamic-shape compilation of Qwen4's sparse FlexAttention, so that packs of every length reuse one kernel.
Dense blocks use AutoModel's Qwen3.5 MoE full-attention class for packed dispatch; CP1 uses the same packed path as higher CP degrees.
`models/qwen3_5.py` selects the packed runtime, binds CP meshes, and synchronizes heterogeneous vision branches across the FSDP group.
`packed_cp_forward.py` owns the RL batch fields, token sharding, and forward context.
The corresponding upstream tracker is [NVIDIA-NeMo/Automodel#2985](https://github.com/NVIDIA-NeMo/Automodel/issues/2985).

## Model Construction

Backend-level model, forward, packing, LoRA, and trainer modules are model-independent. `slim.backends.nemo.models.model_family(config)` selects a model-family module by the Hugging Face `model_type`. A family module validates the topology, builds the model, builds the unsharded model inputs of a pack, builds the CP sharder of those inputs and the RL token fields, and names the module types whose parameters the rollout engines keep from their own checkpoint. The Qwen3.5 family lives in `models/qwen3_5.py` and the Qwen4 family lives in `models/qwen4.py`. The shared `models.build_model` sets the storage dtypes and the LoRA adapters for every model family, and `peft.py` holds the LoRA support over AutoModel's PEFT modules.

Policy and reference models are created with:

```python
NeMoAutoModelForImageTextToText.from_pretrained(...)
```

Qwen3.5's `BackendConfig` uses an internal `sdpa` attention dispatch. AutoModel's packed CP runtime sends that dispatch to the FlashAttention 2 varlen kernel and falls back to PyTorch SDPA when the varlen kernel is unavailable. Its packed full-attention layers use needed-only halo K/V exchange by default. Documents spanning more than two CP ranks use A2A exchange, while cross-node CP groups fall back to all-gather. Linear, RMS normalization, expert GEMM, and MoE dispatcher backends are selected with `--nemo-linear-backend`, `--nemo-rms-norm-backend`, `--nemo-experts-backend`, and `--nemo-dispatcher`. Their non-FP8 choices follow AutoModel's `BackendConfig`; the defaults are `torch`, `torch_fp32`, `torch_mm`, and `torch`. Other choices retain AutoModel's hardware, dependency, and combination constraints. `FSDP2Config` uses AutoModel's default BF16 parameter and output policy with FP32 gradient reduction. Layer parameters reshard after forward, while prefetch scheduling follows AutoModel's FSDP2 defaults. `--master-weight-dtype` sets the storage dtype of the trained policy and critic. The optimizer updates the stored parameters in place and keeps no master copy, so the default `fp32` makes them the master weights while FSDP2 still computes in BF16; `auto` keeps the checkpoint's dtype. The frozen reference model has no optimizer and keeps the checkpoint's dtype. FSDP2 shards at every world size, including one GPU, so its precision policy always applies; `dependency_patches/nemo_automodel.py` removes AutoModel's single-GPU skip. Qwen3.5's model-defined fp32 Gated DeltaNet holders remain separate dtype-uniform FSDP units that compute in fp32. `DistributedSetup` constructs the FSDP2, CP, and EP meshes before model creation. The optimizer runs over the resulting distributed parameters.

`--optimizer` selects `muon` (default) or `adamw`. `adamw` is AutoModel's `AdamWConfig` with `--weight-decay`, `--adam-beta1`, `--adam-beta2`, and `--adam-eps`. `muon` is Dion's FSDP2 `Muon` with Moonshot's Moonlight recipe: Nesterov momentum `--muon-momentum` (default 0.95), five quintic Newton-Schulz steps with coefficients $(3.4445, -4.7750, 2.0315)$, update scale $0.2\sqrt{\max(m, n)}$ so that the update RMS matches AdamW, and decoupled weight decay `--weight-decay`. The Muon update applies to the 2D matrices and the 3D grouped expert weights. Embeddings, the output embedding, biases, 1D parameters, convolution weights, and the critic value head use Dion's AdamW with the AdamW flags. When experts are FSDP-sharded over `ep_shard > 1` (EP smaller than the FSDP size), their matrix shards are outside Dion's FSDP mesh, so they use AdamW and the backend logs this. Each Slim parameter group splits into a Muon subgroup and an AdamW subgroup with the same `max_lr` and `start_step`. Every grouped expert tensor gets its own Muon subgroup: Dion stacks the same-shape parameters of a subgroup into one Newton-Schulz batch, and stacking the expert tensors of all MoE layers needs tens of GiB of temporaries. Optimizer state from one optimizer does not load into the other; use `--no-load-optim` to switch.

AutoModel applies the parameter freeze configuration before sharding and optimizer construction. Vision and language model parameters are trainable by default. The CLI flags `--freeze-vision-tower` and `--freeze-language-model` explicitly select frozen components. `--freeze-ngram` freezes the PLE n-gram table of Qwen4.

Training constructs the policy without MTP layers. SGLang retains the checkpoint's draft weights through its draft-weight backup while policy updates stream the trained backbone.

The critic uses the same AutoModel backbone and a Slim scalar value head. The value head is FSDP2-sharded with the same precision policy as the backbone.

There is no direct Transformers `AutoModelFor*` training path.

## Topology

Let \(W\) be world size and \(C\) be context-parallel size. The logical data-parallel size used to partition trajectories is:

$$
D = \frac{W}{C}
$$

AutoModel's root mesh contains a size-one `dp_replicate` axis, a `dp_shard` axis of size \(D\), and a `cp` axis of size \(C\). Dense parameters use the flattened `dp_shard_cp` FSDP mesh:

$$
D C = W
$$

Every dense parameter is therefore FSDP-sharded across the complete training world. Packed multimodal branch synchronization uses the same `dp_shard_cp` group.

Expert parallelism is composed over the AutoModel MoE mesh and is not an additional world-size factor. For expert-parallel size \(E\), the MoE mesh is `[ep_shard=W/E, ep=E]`. Expert parameters are excluded from the transformer block's ordinary FSDP unit. EP assigns expert subsets to ranks, while `ep_shard` FSDP-shards each subset.

Slim assigns its Gloo process group to AutoModel's `MeshContext.process_group`. Distributed checkpoint planning and metadata exchange therefore use the control-plane group rather than allocating rank-zero NCCL planning tensors.

Relevant arguments are:

```text
--context-parallel-size
--expert-model-parallel-size
--activation-checkpointing
```

The backend does not expose TP, PP, or sequence-parallel arguments. It rejects dense Qwen3.5 with EP greater than one, Qwen4 with EP of one, invalid world-size factorizations, and MoE expert counts that are not divisible by EP.

## Unified Batch Interface

Dense and MoE training use the same Slim batch, forward, token-sharding, gather, and loss interfaces.

`pack_sequences` creates a CPU-resident physical pack. `build_model_batch` emits:

| Field | Shape | Meaning |
|---|---|---|
| `input_ids` | `[1, S]` | Concatenated document tokens |
| `labels` | `[1, S]` | Next-token labels with `-100` at document ends |
| `position_ids` | `[1, S]`; `[3, 1, S]` for Qwen3.5 mRoPE | Document-local positions |
| `_packed_seq_ids` | `[1, S]` | One-based document IDs (Qwen3.5) |
| RL token fields | `[1, S, ...]` | Source-token-aligned training values |
| `cu_seqlens` | `[N + 1]` | Document boundaries before CP, a model input of Qwen4 |

For document tokens \(x_0, x_1, \ldots, x_{n-1}\):

$$
\operatorname{labels}[t] =
\begin{cases}
x_{t+1}, & t < n - 1 \\
-100, & t = n - 1
\end{cases}
$$

No label or prediction crosses a document boundary.

`prepare_forward` constructs the `ContextParallelSharder` of the model family. Its `shard_token_tensor` and `gather_token_tensor` methods handle every RL token field. Actor, reference, critic, dense, and MoE forwards use this same contract.

Before role-local packing, the Ray `AdvantageEstimator` writes source-token-aligned training targets to each `Trajectory`. GRPO and GSPO produce `advantages`. PPO GAE consumes source-token-aligned critic value payloads and produces `advantages`, `values`, and `value_targets`. The actor consumes `advantages`; the critic consumes `old_values` and `value_targets`.

Actor log-probability precompute and critic value precompute may cache packs before target construction. `update_packed_targets` attaches the completed targets to each role's cached packs immediately before training. Actor and critic do not share physical pack objects.

The rollout/train boundary flattens episodes into trajectories, pads the list to a multiple of `logical_dp * num_steps` with inert two-token documents, then partitions twice: across optimizer steps, then within each step across ranks. Every rank therefore holds the same document count per step, which is what the synchronized pack count in `_packed_data` requires. `--balance-data` is enabled by default and uses Karmarkar-Karp partitioning to balance token counts at both levels; `--no-balance-data` restores round-robin assignment.

The optimizer step count stays denominated in episodes, so the LR schedule length does not move with how many generation calls a rollout made.

`--max-tokens-per-gpu` is the target post-CP token budget for one rank. Dynamic packing therefore targets a physical pack length of:

$$
T_{\mathrm{pack}} \le C T_{\mathrm{GPU}}
$$

For CP2 and `--max-tokens-per-gpu 8192`, each physical pack contains at most 16K tokens before CP sharding. The packer uses budget-aware first-fit decreasing partitions and only splits existing partitions when DP ranks need a synchronized pack count. Individual trajectories are not split, so a trajectory longer than the physical budget is rejected.

## Context Parallel Layouts

Every physical microbatch uses the indexed packed layout, including a pack containing one document.

| Configuration | Physical pack | AutoModel layout |
|---|---|---|
| Qwen3.5 CP1, dense or MoE | One or more documents | Full token order, even-length padding, block-diagonal per-document attention |
| Qwen3.5 CP greater than one, dense or MoE | One or more documents | Contiguous block-diagonal CP |
| Qwen4 CP1 | One or more documents | Full token order without padding; the model reads `cu_seqlens` |
| Qwen4 CP greater than one | One or more documents | AutoModel's model-owned contiguous CP, padded to a multiple of \(4C\) |

For Qwen3.5, full attention exchanges K/V through AutoModel's block-diagonal CP runtime. Gated DeltaNet consumes the same global document boundaries and resets convolution and recurrent state at each boundary.
AutoModel's dense and MoE forwards embed and splice the full sequence, then select the contiguous primary-token shard.
CP1 uses the same packed path with padding to a multiple of two; it is distinct from execution outside the packed context.
For Qwen4, AutoModel's model-owned sharder slices the token inputs into contiguous rank-local shards before the forward, and the CP context carries the global token IDs and document boundaries that QSA, Gated DeltaNet, and PLE read.
Slim's embedding and vision integration aligns image and video branches across FSDP ranks while retaining gradients through the trainable vision tower.

Text positions restart at zero for every document. Mixed-modality packs call the Qwen3.5 mRoPE builder once per document with that document's image and video grids, then concatenate the resulting position axes before CP sharding.

## Loss Computation

The training loop follows AutoModel recipe ordering:

1. Count the global document weight sum and valid tokens over logical DP ranks.
2. Resolve and apply the model family's packed CP sharder.
3. Enter the CP context and NeMo gradient synchronization context.
4. Run the model with filtered forward arguments.
5. Compute CP-local target log probabilities, entropy, policy loss, or value loss.
6. Normalize local contributions by a DP-global denominator.
7. Multiply the local scalar by the DP and CP gradient-reduction group size before backward.
8. Use NeMo gradient scaling, clipping, EP gradient normalization, and optimizer stepping.

Every model family builds the policy with `router_aux_loss_coef` set to zero. The RL loss graph therefore contains only the configured policy, entropy, and reference-KL terms.

For policy token-sum reduction:

$$
L_{\mathrm{token}} = \frac{1}{T}\sum_{d,t} m_{d,t}\ell_{d,t},
\qquad
T = \sum_{d,t} m_{d,t}
$$

For policy sequence-mean reduction, where $w_d$ is the document's `loss_weight`:

$$
L_{\mathrm{sequence}} = \frac{1}{\sum_d w_d}\sum_d w_d
\frac{\sum_t m_{d,t}\ell_{d,t}}
{\max(1,\sum_t m_{d,t})}
$$

A document is one trajectory. `--loss-normalization-unit` sets $w_d$: `episode` writes $1/k$ for an attempt with $k$ trajectories so $\sum_d w_d$ counts attempts, and `trajectory` writes $1$ so it counts total numbers of trajectories. Padding documents carry $w_d = 0$ and drop out of both the numerator and the denominator.

GSPO computes differentiable per-document CP reductions before expanding each sequence statistic back to its local tokens.

`--loss-normalization-unit token` selects policy token-sum reduction. Policy diagnostics (`pg_clipfrac`, `pg_kl_k3`) use the same selected reduction. Entropy, reference KL, score-centering diagnostics, and clipped critic value regression use sequence-mean reduction.

## Routing Replay

Rollout routing choices are source-token aligned with shape:

```text
[num_tokens, num_moe_layers, top_k]
```

The final token row is zero because it has no prediction target. Packing concatenates routing rows with the same token boundaries as the model input, and the selected CP sharder maps them to model-local token order. Router replay remains active through backward so activation-checkpoint recomputation sees the same expert indices. The registry is process owned and clears replay state on context exit.

Routing replay is valid for Qwen3.5 MoE and Qwen4.

## Qwen4

Slim builds the Qwen4 policy with `language_model_only`, so it has no vision tower and no MTP layer. Only a speculative-decoding draft in SGLang loads the checkpoint's MTP weights.
The rollout engines also serve the language model only, through `--sglang-json-model-override-args {"language_model_only":true}`, so they hold no vision tower, which the policy neither trains nor sends.
Packs with images or videos are an error.
The family requires `--freeze-vision-tower` and `--freeze-ngram`, and rejects `--freeze-language-model`.

| Component | Training |
|---|---|
| MoE experts, router, shared expert | Trained; EP greater than one is required |
| Gated DeltaNet and full-attention projections | Trained |
| Hyper-connections | Trained, as ordinary FSDP2 parameters of each decoder block |
| PLE projections, norms, and short convolution | Trained |
| PLE n-gram table, \(320{,}001{,}536 \times 160\) rows by width | Frozen by `--freeze-ngram` and stored in BF16; owner-sharded over the training world, excluded from FSDP2, the optimizer, and the weight sync |
| QSA indexer | Frozen by AutoModel; its top-k is recomputed in every forward and replayed only within activation-checkpoint recomputation |

The n-gram table holds 51.2B parameters, which is 102.4 GB in BF16. Each of \(W\) training ranks holds \(1/W\) of its rows, and lookups use two all-to-all exchanges over the training world. The checkpoint stores the table as 128 row shards, so \(W\) must divide 128.
The table stays in BF16 at every `--master-weight-dtype`, which keeps the checkpoint's values, and the PLE layer casts the rows to the BF16 block activations.
SGLang keeps its copy of the table in pinned host memory with `--sglang-ple-offload-embedding`, outside the GPU weights that a colocated engine releases, so the engines keep the table that they loaded.

QSA uses AutoModel's FlexAttention sparse GQA. Query positions before 2,051 select every complete compressed block, which is dense causal attention; later positions select the 512 top-scoring compressed blocks of four tokens and the tokens of the current incomplete block. Under CP, each rank all-gathers the main K/V with their gradients, the indexer all-gathers the raw index keys and their RoPE values without gradients and compresses every document itself, the Gated DeltaNet passes its state across ranks, and the PLE short convolution reads a nine-token left halo.

On sixteen GPUs, each rank holds 7.86B of the 125.7B trained parameters: 7.55B of expert weights and 0.31B of other weights.
The two-node run uses fp32 master weights and Muon with EP16, so `ep_shard` is one and the Muon update covers the experts.
The trained state is then about 88 GiB per GPU: fp32 weights, fp32 gradients, and one fp32 momentum per matrix, with two AdamW moments only for the small non-matrix parameters.
The BF16 table adds 6.0 GiB per GPU.
With EP8, the experts are FSDP-sharded over `ep_shard` two and use AdamW, which takes about 117 GiB per GPU before activations.
`--master-weight-dtype auto` stores BF16 weights, and an update smaller than half a BF16 ulp of its weight is lost, which at a learning rate of \(10^{-6}\) is nearly every matrix entry.

The two-node GRPO run needs the checkpoint at the same path on both nodes:

```bash
MASTER_ADDR=<head IP> WORKER_HOST=<worker ssh host> \
QWEN4_MODEL_DIR=/path/to/Qwen4 \
bash tests/e2e/run_nemo_qwen4.sh
```

On two p5en.48xlarge nodes, the run completes three rollouts, three optimizer steps, and four weight synchronizations:

| Measurement | Result |
|---|---|
| Rollout | 436 to 643 seconds; mean rewards 0.469, 0.625, and 0.594; 59 to 72 percent of the responses reach the 16K context |
| Actor step | 372 seconds for the first step, which compiles the kernels, then 68 and 52 seconds |
| Weight synchronization | 24.5 to 26.5 seconds for 251.5 GB of BF16 weights |
| Train/rollout mismatch | `pg_kl_k3` of \(5.2 \times 10^{-4}\), \(4.7 \times 10^{-4}\), and \(4.7 \times 10^{-4}\) |
| Gradient norm | 0.344, 0.155, and 0.137 |
| GPU memory | 140 to 142 GB of 143.7 GB in use per GPU at the training peak, including the caching allocator's reserve |

The packed CP harness qualifies Qwen4 with a first document that crosses the sparse QSA boundary and the CP shard boundary.
It runs forward and backward without an optimizer step, so on one node it stores BF16 weights with `--master-weight-dtype auto`.
With `--routing-replay`, the CP1 baseline records its expert selection and the CP2 candidate replays it, as RL training replays the rollout's selection.
Without replay, BF16 differences between layouts flip some of the top-10 expert choices, so the comparison measures those flips instead of CP: CP2 differs from CP1 by up to 6.6 nats at single tokens of this document, and on natural text a CP1 run with EP4 already differs from EP8 by up to 1.9 nats.

```bash
uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen4 \
  --context-parallel-size 1 \
  --expert-parallel-size 8 \
  --master-weight-dtype auto \
  --first-document-length 4096 \
  --routing-replay \
  --output /tmp/qwen4-cp1-ep8.pt

uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen4 \
  --context-parallel-size 2 \
  --expert-parallel-size 8 \
  --master-weight-dtype auto \
  --first-document-length 4096 \
  --routing-replay \
  --output /tmp/qwen4-cp2-ep8.pt \
  --baseline-output /tmp/qwen4-cp1-ep8.pt
```

## Learning Rate Schedule

The LR schedule has three phases: warmup for `--lr-warmup-steps`, constant, and decay for `--lr-decay-steps` with the `--lr-decay-style` shape.
With `--num-rollout`, the decay ends at the last rollout.
With `--num-epoch E`, `E * len(dataset) // rollout_batch_size` rollouts is the upper limit.
When the data source finishes epoch E before the decay start, the decay starts at that rollout, and the training stops when the decay ends.
The decay never starts before all warmups end.

## Checkpoints

AutoModel's `CheckpointingConfig` and `Checkpointer` save model and optimizer state. Slim preserves this directory contract:

```text
<save-root>/
  latest_checkpointed_iteration.txt
  iter_0000001/
    model/
    optim/
    value_head/   # critic
    rng/
    meta.json
```

`model/` holds the backbone of the actor or the critic through the model's state-dict adapter.
Actor checkpoints can export consolidated Hugging Face safetensors according to `--checkpoint-save-consolidated`.
The export also holds the original `config.json` of `--hf-checkpoint` and the processor files, which include the tokenizer. Without a processor, it holds the tokenizer files.

The critic's scalar value head is in `value_head/`, and `optim/` holds the optimizer state of the backbone and the value head. Critic consolidated HF export is disabled.

The reference model never changes, so checkpoints do not hold it, and a resume builds it again from `--ref-load`, by default `--hf-checkpoint`.

Asynchronous save uses AutoModel staging and upload futures. Slim publishes `latest_checkpointed_iteration.txt` only after the pending save completes.

## SGLang Weight Synchronization

Every policy state tensor is converted through the model's `convert_single_tensor_to_hf` adapter before transport. This conversion handles Qwen3.5 native names, Gated DeltaNet fp32 holders, and grouped expert layouts.
Parameters of the family's `ROLLOUT_STATIC_MODULES` are not sent: the engines keep the values they loaded from `--rollout-load`. These parameters must be outside the GPU memory that a colocated engine releases.
`--check-weight-update-equal` is an error for such a family, because SGLang's weight checker resets every engine tensor and the sync does not restore these.
For Qwen4 this is the frozen n-gram table, which SGLang holds in pinned host memory, so the rollout checkpoint must hold the same table as the policy checkpoint.

The configured rollout quantizer runs after conversion. Colocated engines receive flattened CUDA IPC buckets. Separate engines receive tensors through the temporary NCCL update group.

## PEFT

`--actor-peft-config` and `--critic-peft-config` train LoRA adapters on a frozen base model instead of all parameters.
LoRA is the only supported adapter type.
Each value holds NeMo AutoModel `PeftConfig` fields, for example `dim`, `alpha`, `target_modules`, `exclude_modules`, `match_all_linear`, `dropout`, `lora_A_init`, and `moe_rank_scaling`.
The command line takes the value as JSON, and the `--config` YAML takes it as a mapping:

```bash
--actor-peft-config '{"dim": 32, "alpha": 64, "target_modules": ["*_proj"]}'
```

```yaml
actor_peft_config:
  dim: 32
  alpha: 64
  target_modules: ["*_proj", "*.experts"]
```

`target_modules` and `exclude_modules` match module names with wildcards and are mutually exclusive.
Without `target_modules`, `exclude_modules`, and `match_all_linear`, AutoModel adapts the modules that match `*_proj`.
On Qwen3.5, `*_proj` covers the attention and MLP projections and the Gated DeltaNet `out_proj`, but not the Gated DeltaNet inputs `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a`; add `*in_proj_*` to adapt them.
On Qwen3.5 MoE, `*.experts` adds LoRA to the grouped expert weights of every MoE layer, and `moe_rank_scaling` divides their rank by the number of activated experts.
On Qwen4, `*_proj` also matches `index_qk_proj` of the frozen QSA indexer, whose integer top-k passes no gradient, so these adapters stay at zero.
For an adapted weight $W$ with LoRA factors $A$ and $B$ of rank $r$ (`dim`), the effective weight is $W + s B A$ with $s = \alpha / r$.
`match_all_linear` adapts every `nn.Linear`, `lm_head` included, and takes no `exclude_modules`; `exclude_modules: ["*lm_head"]` alone adapts every other `nn.Linear`.
Neither form adapts the grouped experts or the MoE router, which are not `nn.Linear`, so expert LoRA needs `target_modules` with `*.experts`.
With tied word embeddings, as in Qwen3.5-4B, an adapted `lm_head` is an error, because the rollout engine holds one matrix for the embedding and `lm_head`.

### Precision

The frozen base keeps the checkpoint's dtype.
`--master-weight-dtype` sets the storage dtype of the adapters: `fp32` stores them in fp32, and `auto` stores them in the dtype of the weight they adapt.
Each weight computes in the mixed-precision dtype of its FSDP2 unit, as in full-parameter training, so the Qwen3.5 Gated DeltaNet holders `A_log` and `dt_bias` stay fp32.
The critic's scalar value head is a trainable module, not an adapter.
Use `--optimizer adamw` with PEFT.

### Rollout Weights

Rollout engines receive merged weights and no adapter tensors.
Every synchronization sends the frozen weights unchanged and $W + s B A$ in place of each adapted weight $W$, through the conversion, quantization, and transport path of [SGLang Weight Synchronization](#sglang-weight-synchronization).
AutoModel's `materialize_effective_weight` and `materialize_effective_weights` compute the merge from the sharded tensors in the dtype of the adapters, which the default `--master-weight-dtype fp32` makes fp32, and each merged shard takes the compute dtype of $W$.
An actor that offloads merges on the GPU before each offload, at the end of its initialization and of each training step, and keeps the merged shards in pinned CPU memory until the next offload.
An actor that stays on the GPU merges at each synchronization.
SGLang therefore runs unchanged, and an FP8 `--rollout-load` quantizes the merged weights.
The trainer computes the adapter path next to the frozen base weight, while the engine uses the merged weight rounded to the compute dtype.
This rounding adds to the train/rollout mismatch that `pg_kl_k3` and `pg_clipfrac` measure.

### Reference Model

With a nonzero `--kl-loss-coef`, `--ref-load` selects the reference policy:

- Unset: the reference is `--hf-checkpoint`. The actor computes the reference log probabilities with its own model and every LoRA scale at zero, so it builds no second model.
- A NeMo PEFT adapter directory: the actor builds a separate frozen reference model from `--hf-checkpoint` with that adapter and the PEFT config saved beside it. This requires a PEFT actor.
- A Hugging Face checkpoint directory, for example a merged SFT model: the actor builds a separate frozen reference model, as in full-parameter training.

To start from an SFT adapter and keep the KL anchored to it, give the adapter directory to both `--load` and `--ref-load`.
A separate reference model costs the same memory and transfer time as in full-parameter training.
To anchor the KL to an SFT adapter without a second model, merge the adapter into a Hugging Face checkpoint, use that checkpoint as `--hf-checkpoint`, and train new adapters on it.

### Load and Save

The base model is always `--hf-checkpoint`.
With PEFT, `--load` accepts two kinds of directory:

- A Slim PEFT checkpoint root resumes training.
- A NeMo PEFT adapter directory, for example an SFT adapter or `iter_*/model` of a Slim PEFT checkpoint, initializes only the adapter weights and starts a fresh optimizer.

A NeMo PEFT adapter directory holds `adapter_config.json` and `automodel_peft_config.json`.
In both cases the PEFT config comes from the loaded directory and takes precedence over `--actor-peft-config` and `--critic-peft-config`, so these flags are optional.
A Hugging Face full-model checkpoint or a full-parameter Slim checkpoint in `--load` is an error with PEFT.
`--critic-load` accepts the same directories for the critic.
An adapter directory initializes only the critic adapters, and the value head starts from a fresh initialization.

Checkpoints keep the directory layout of [Checkpoints](#checkpoints).
`model/` holds only the backbone adapters: `adapter_model.safetensors` and `adapter_config.json` in Hugging Face PEFT format, and `automodel_peft_config.json`.
The actor and the critic save their adapters in the same format, so an adapter directory of either role can initialize the other.
`--checkpoint-save-consolidated` has no effect with PEFT.
To get a full model, `nemo_automodel.export_merged_peft_checkpoint` merges an adapter directory into an unsharded model that is built from `--hf-checkpoint` with the same PEFT config, and writes Hugging Face safetensors.

## Validation

Hardware-independent checks:

```bash
uv run ruff check slim/backends/nemo tests/unit/backends/nemo tests/e2e/nemo
uv run pytest tests/unit/backends/nemo -q
uv run pytest tests/unit/utils/test_ppo_utils.py -q
uv run python -c "import slim.backends.nemo"
```

### End-to-End RL Qualification

The single-node Qwen3.5 MoE qualification uses a default eight-GPU Ray head:

```bash
uv run ray start --head --num-gpus 8 --disable-usage-stats --block
```

Run GRPO with four TP2 SGLang replicas and one NeMo CP2 and EP8 training world:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync slim-train \
  --num-rollout 1 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --num-steps-per-rollout 1 \
  --max-context-len 16384 \
  --rollout-temperature 1 \
  --prompt-data /tmp/slim-e2e-gsm8k-32.parquet \
  --rm-type math \
  --skip-eval-before-train \
  --rollout-num-gpus-per-replica 2 \
  --sglang-mem-fraction-static 0.7 \
  --sglang-attention-backend fa3 \
  --sglang-mamba-ssm-dtype float32 \
  --sglang-mamba-radix-cache-strategy extra_buffer_lazy \
  --sglang-max-mamba-cache-size 512 \
  --rollout-colocate \
  --actor-num-gpus 8 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 8 \
  --activation-checkpointing \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --advantage-estimator grpo \
  --disable-group-advantage-std-normalization \
  --use-rollout-routing-replay \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --lr 1e-5 \
  --lr-warmup-steps 0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --hf-checkpoint models/Qwen3.5-35B-A3B
```

The run completes 256 episodes, one optimizer step, and the final NeMo-to-SGLang update. The observed results are:

| Measurement | Result |
|---|---|
| Rollout | 242.09 seconds, mean reward 0.94140625 |
| Context lengths | Mean 5,456.49, median 4,422.5, maximum 16,382, truncation ratio 0.07421875 |
| Training topology | World size 8, logical DP4, CP2, EP8 |
| Packed CP | Physical boundary document 16,382 tokens, local CP length 8,192 tokens |
| Routing replay | Enabled through SGLang capture, episode packing, CP sharding, forward, and activation-checkpoint backward |
| Actor step | 252.70 seconds, total training phase 254.99 seconds |
| Actor metrics | Loss \(-3.3276 \times 10^{-6}\), gradient norm 9.70999, clip fraction \(1.2974 \times 10^{-6}\), \(k_3\) KL \(2.3161 \times 10^{-4}\) |
| Weight synchronization | 12.3 seconds before rollout and 12.5 seconds after the optimizer step |
| Sampled GPU memory | Maximum 78,712 MiB on GPU 3 during final synchronization; GPU 0 used 78,146 MiB |

The sampled memory profile does not show a rank-zero memory premium. GPU 3 was the highest-memory rank during the final synchronization sample.

The packed CP harness compares CP candidates against an eight-GPU CP1 baseline:

```bash
uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 1 \
  --output /tmp/qwen35-cp1.pt

uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 2 \
  --output /tmp/qwen35-cp2.pt \
  --baseline-output /tmp/qwen35-cp1.pt
```

For MoE CP and EP composition, hold EP fixed in the baseline and candidate:

```bash
uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 1 \
  --expert-parallel-size 8 \
  --output /tmp/qwen35-cp1-ep8.pt

uv run torchrun --standalone --nproc-per-node 8 tests/e2e/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 2 \
  --expert-parallel-size 8 \
  --output /tmp/qwen35-cp2-ep8.pt \
  --baseline-output /tmp/qwen35-cp1-ep8.pt
```

The harness verifies finite outputs and gradients, exact packed-document isolation, block-diagonal attention execution (Qwen3.5), target log-probability parity, mean-loss parity, aggregate gradient parity, and first-sequence-layer parity. Baseline and candidate artifacts must use the same expert-parallel size.

`tests/e2e/sweep_e2e.sh` trains its replay combos on a fixed fixture in which DP rank 1 holds the vision episodes and every other DP rank holds pure text. The vision tower is trainable in the full-parameter combos and frozen as part of the PEFT base in the LoRA combos. Every physical rank enters the synchronized vision path: text-only ranks use a dummy visual input to preserve FSDP collective ordering, and the dummy result contributes a zero-valued autograd dependency when the vision tower is trainable.

`tests/e2e/sweep_e2e.sh` runs the PEFT qualification: LoRA with merged block-FP8 and bf16 rollout weights, expert LoRA on Qwen3.5 MoE with EP2, save and resume, adapter initialization, PPO with a LoRA actor and critic, and KL with the shared reference, a separate reference model, and an adapter reference. The checkpoint and reference combos replay saved rollouts in which one DP rank holds the vision episodes and the other ranks hold pure text.

GPU qualification covers dense CP, MoE CP and EP composition, heterogeneous multimodal batches, both vision freeze states, routing replay, checkpoint resume, LoRA, and NeMo versus SGLang target log-probability parity.

The eight-GPU qualification results for the pinned revision are:

| Path | Topology | Result |
|---|---|---|
| Dense Qwen3.5 packed parity | CP1 and CP2 | Block-diagonal attention fired in all 12 full-attention calls. Mean losses were 9.479804 and 9.491175. |
| Qwen3.5 MoE actor step | CP2 and EP8 | All 40 routing layers replayed at least twice, including activation-checkpoint recomputation. Loss was -0.1109375 and gradient norm was 329.1560. |
| Dense state-dict conversion | CP2 | Converted 474 tensors totaling 2,214,532,352 bytes. |
| MoE grouped-expert conversion | CP2 and EP8 | Converted 1,026 tensors totaling 70,214,367,712 bytes. |
| Qwen4 packed parity with routing replay | CP1 and CP2, EP8 | Mean losses were 9.860187 and 9.860061. The largest target log-probability delta was 0.350, and the gradient relative L2 was 0.065. |
| Qwen4 state-dict conversion | CP1 and EP8, fp32 master weights | Converted 1,166 tensors totaling 251,487,314,712 bytes, without the n-gram table. |
