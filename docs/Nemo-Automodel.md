# NeMo AutoModel Backend

## Scope

`slim.backends.nemo` is Slim's training backend. It combines Slim's Ray and RL lifecycle with NeMo AutoModel model construction, FSDP2, context parallelism, expert parallelism, optimizer utilities, and distributed checkpointing.

First-party support covers:

| Model | Training topology |
|---|---|
| Dense Qwen3.5 | FSDP2 and context parallelism |
| Qwen3.5 MoE | FSDP2, context parallelism, and expert parallelism |

Tensor parallelism, pipeline parallelism, and sequence parallelism are not supported by this backend.
Training updates model parameters directly; adapter fine-tuning is not part of the backend.

SGLang remains the rollout engine. Slim owns `Episode` processing, RL objectives, reference and critic execution, routing replay, metrics, and policy weight synchronization.

## Environment

[`pyproject.toml`](../pyproject.toml) is the dependency source of truth. The AutoModel dependency is pinned to revision `2de37e53d42e26409f08a8a9fa3a1269c56479a4`.

Create or refresh the environment with:

```bash
uv sync --extra dev
```

The pinned revision supplies Qwen3.5 CP token sharding verbs, block-diagonal attention and Gated DeltaNet runtimes, grouped-expert state-dict conversion, and CP vision sharding. Slim applies these primitives through one packed Qwen3.5 integration for dense and MoE models. The corresponding upstream tracker is [NVIDIA-NeMo/Automodel#2985](https://github.com/NVIDIA-NeMo/Automodel/issues/2985).

## Model Construction

Policy and reference models are created with:

```python
NeMoAutoModelForImageTextToText.from_pretrained(...)
```

`BackendConfig` uses an internal `sdpa` attention dispatch. AutoModel's packed CP runtime sends that dispatch to the FlashAttention 2 varlen kernel and falls back to PyTorch SDPA when the varlen kernel is unavailable. Packed full-attention layers use needed-only halo K/V exchange by default. Documents spanning more than two CP ranks use A2A exchange, while cross-node CP groups fall back to all-gather. Linear, RMS normalization, expert GEMM, and MoE dispatcher backends are selected with `--nemo-linear-backend`, `--nemo-rms-norm-backend`, `--nemo-experts-backend`, and `--nemo-dispatcher`. Their non-FP8 choices follow AutoModel's `BackendConfig`; the defaults `torch`, `torch_fp32`, `torch_mm`, and `torch` are the qualified configuration. Other choices retain AutoModel's hardware, dependency, and combination constraints. `FSDP2Config` uses AutoModel's default BF16 parameter and output policy with FP32 gradient reduction. Layer parameters reshard after forward, and cross-layer backward prefetch is disabled to bound peak memory. Slim exposes no separate parameter-storage or compute-dtype controls and passes no dtype override during model loading. Qwen3.5's model-defined fp32 Gated DeltaNet holders remain separate dtype-uniform FSDP units. `DistributedSetup` constructs the FSDP2, CP, and EP meshes before model creation. The optimizer is AdamW over the resulting distributed parameters.

AutoModel applies the parameter freeze configuration before sharding and optimizer construction. Vision and audio towers are frozen by default, while the language model is trainable. `--no-freeze-vision-tower` and `--no-freeze-audio-tower` opt those towers into training, and `--freeze-language-model` restricts training to enabled non-language towers.

Training constructs the policy without MTP layers. SGLang retains the checkpoint's draft weights through its draft-weight backup while policy updates stream the trained backbone.

The critic uses the same AutoModel backbone and a Slim scalar value head. The value head is FSDP2-sharded with the same precision and offload policies as the backbone.

There is no direct Transformers `AutoModelFor*` training path.

## Topology

Let \(W\) be world size, \(C\) context-parallel size, and \(D_r\) replicated data-parallel size. The sharded data-parallel size is:

$$
D_s = \frac{W}{C D_r}
$$

The logical data-parallel size used to partition episodes is:

$$
D = \frac{W}{C}
$$

The root mesh always contains the replicated data-parallel dimension. AutoModel uses a one-dimensional FSDP mesh when \(D_r = 1\) and a two-dimensional HSDP mesh when \(D_r > 1\). Packed multimodal branch synchronization uses the flattened `dp_shard_cp` group, so ranks that share an FSDP shard group follow the same vision-tower collective path without synchronizing independent HSDP replicas.

Expert parallelism is composed over the AutoModel MoE mesh. It is not an additional world-size factor. Expert parameters are excluded from the transformer block's ordinary FSDP unit. EP assigns expert subsets to ranks, while an orthogonal `ep_shard` mesh may FSDP-shard each subset across its data-parallel copies.

Slim assigns its Gloo process group to AutoModel's `MeshContext.process_group`. Distributed checkpoint planning and metadata exchange therefore use the control-plane group rather than allocating rank-zero NCCL planning tensors.

Relevant arguments are:

```text
--context-parallel-size
--expert-model-parallel-size
--dp-replicate-size
--activation-checkpointing
--nemo-cpu-offload
```

The backend does not expose TP, PP, sequence-parallel, or optimizer selection arguments. It rejects dense Qwen3.5 with EP greater than one, invalid world-size factorizations, and MoE expert counts that are not divisible by EP.

## Unified Batch Interface

Dense and MoE training use the same Slim batch, forward, token-sharding, gather, and loss interfaces.

`pack_sequences` creates a CPU-resident physical pack. `build_model_batch` emits:

| Field | Shape | Meaning |
|---|---|---|
| `input_ids` | `[1, S]` | Concatenated document tokens |
| `labels` | `[1, S]` | Next-token labels with `-100` at document ends |
| `_packed_seq_ids` | `[1, S]` | One-based document IDs |
| RL token fields | `[1, S, ...]` | Edge values placed at source-token positions |
| `cu_seqlens` | `[N + 1]` | Document boundaries before CP |

For document tokens \(x_0, x_1, \ldots, x_{n-1}\):

$$
\operatorname{labels}[t] =
\begin{cases}
x_{t+1}, & t < n - 1 \\
-100, & t = n - 1
\end{cases}
$$

No label or RL edge crosses a document boundary.

`prepare_forward` constructs one contiguous block-diagonal `ContextParallelSharder`. Its `shard_token_tensor` and `gather_token_tensor` methods handle every RL token field. Actor, reference, critic, dense, and MoE forwards use this same contract.

`--max-tokens-per-gpu` is the target post-CP token budget for one rank. Dynamic packing therefore targets a physical pack length of:

$$
T_{\mathrm{pack}} \le C T_{\mathrm{GPU}}
$$

For CP2 and `--max-tokens-per-gpu 8192`, each physical pack contains at most 16K tokens before CP sharding. The packer uses budget-aware first-fit decreasing partitions and only splits existing partitions when DP ranks need a synchronized pack count. Individual episodes are not split, so an episode longer than the physical budget is rejected.

## Context Parallel Layouts

Every physical microbatch uses the indexed packed layout, including a pack containing one document.

| Configuration | Physical pack | AutoModel layout |
|---|---|---|
| CP1, dense or MoE | One or more documents | Identity token layout with block-diagonal per-document attention |
| CP greater than one, dense or MoE | One or more documents | Contiguous block-diagonal CP |

Full attention exchanges K/V through AutoModel's block-diagonal CP runtime. Gated DeltaNet consumes the same global document boundaries and resets convolution and recurrent state at each boundary. One Qwen3.5 adapter embeds and splices the full sequence, takes the contiguous primary-token shard, and dispatches full attention for dense and MoE models.

Text positions restart at zero for every document. Mixed-modality packs call the Qwen3.5 mRoPE builder once per document with that document's image and video grids, then concatenate the resulting position axes before CP sharding.

## Loss Computation

The training loop follows AutoModel recipe ordering:

1. Count global samples and valid tokens over logical DP ranks.
2. Resolve and apply the common packed CP sharder.
3. Enter the CP context and NeMo gradient synchronization context.
4. Run the model with filtered forward arguments.
5. Compute CP-local target log probabilities, entropy, policy loss, or value loss.
6. Normalize local contributions by a DP-global denominator.
7. Multiply the local scalar by the DP and CP gradient-reduction group size before backward.
8. Use NeMo gradient scaling, clipping, EP gradient normalization, and optimizer stepping.

Qwen3.5 policy construction sets `router_aux_loss_coef` to zero. The RL loss graph therefore contains only the configured policy, entropy, and reference-KL terms.

For policy token-sum reduction:

$$
L_{\mathrm{token}} = \frac{1}{T}\sum_{d,t} m_{d,t}\ell_{d,t},
\qquad
T = \sum_{d,t} m_{d,t}
$$

For policy sequence-mean reduction:

$$
L_{\mathrm{sequence}} = \frac{1}{B}\sum_d
\frac{\sum_t m_{d,t}\ell_{d,t}}
{\max(1,\sum_t m_{d,t})}
$$

GSPO computes differentiable per-document CP reductions before expanding each sequence statistic back to its local tokens.

`--calculate-per-token-loss` selects policy token-sum reduction. Policy diagnostics use the same selected reduction. Entropy, reference KL, mismatch diagnostics, and clipped critic value regression use sequence-mean reduction.

## Routing Replay

Rollout routing choices are edge aligned with shape:

```text
[num_edges, num_moe_layers, top_k]
```

Packing places each routing row at its source-token slot. The selected CP sharder maps those rows to model-local token order. Router replay remains active through backward so activation-checkpoint recomputation sees the same expert indices. The registry is process owned and clears replay state on context exit.

Routing replay is valid only for Qwen3.5 MoE.

## Checkpoints

AutoModel's `CheckpointingConfig` and `Checkpointer` save model and optimizer state. Slim preserves this directory contract:

```text
<save-root>/
  latest_checkpointed_iteration.txt
  iter_0000001/
    model/
    optim/
    reference/
    rng/
    meta.json
```

Actor checkpoints use the Qwen3.5 state-dict adapter and can export consolidated Hugging Face safetensors according to `--checkpoint-save-consolidated`.

Critic checkpoints use one native composite module containing the backbone and scalar head. This keeps the value head and all optimizer parameter groups in the same distributed checkpoint. Critic consolidated HF export is disabled because the composite critic is not a Hugging Face causal language model.

Asynchronous save uses AutoModel staging and upload futures. Slim publishes `latest_checkpointed_iteration.txt` only after the pending save completes.

## SGLang Weight Synchronization

Every policy state tensor is converted through the model's `convert_single_tensor_to_hf` adapter before transport. This conversion handles Qwen3.5 native names, Gated DeltaNet fp32 holders, and grouped expert layouts.

The configured rollout quantizer runs after conversion. Colocated engines receive flattened CUDA IPC buckets. Separate engines receive tensors through the temporary NCCL update group.

## Validation

Hardware-independent checks:

```bash
uv run ruff check slim/backends/nemo tests/backends/nemo
uv run pytest tests/backends/nemo -q
uv run pytest tests/utils/test_ppo_utils.py -q
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
  --mamba-radix-cache-strategy extra_buffer \
  --sglang-page-size 64 \
  --sglang-enforce-disable-flashinfer-allreduce-fusion \
  --rollout-colocate \
  --actor-num-gpus 8 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 8 \
  --activation-checkpointing \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --advantage-estimator grpo \
  --disable-rewards-std-normalization \
  --old-logprob-source rollout \
  --use-rollout-routing-replay \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --lr 1e-5 \
  --lr-warmup-iters 0 \
  --lr-decay-style constant \
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
uv run torchrun --standalone --nproc-per-node 8 tests/backends/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 1 \
  --output /tmp/qwen35-cp1.pt

uv run torchrun --standalone --nproc-per-node 8 tests/backends/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 2 \
  --output /tmp/qwen35-cp2.pt \
  --baseline-output /tmp/qwen35-cp1.pt
```

For MoE CP and EP composition, hold EP fixed in the baseline and candidate:

```bash
uv run torchrun --standalone --nproc-per-node 8 tests/backends/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 1 \
  --expert-parallel-size 8 \
  --output /tmp/qwen35-cp1-ep8.pt

uv run torchrun --standalone --nproc-per-node 8 tests/backends/nemo/run_packed_cp_qualification.py \
  --checkpoint /path/to/Qwen3.5-35B-A3B \
  --context-parallel-size 2 \
  --expert-parallel-size 8 \
  --output /tmp/qwen35-cp2-ep8.pt \
  --baseline-output /tmp/qwen35-cp1-ep8.pt
```

The harness verifies finite outputs and gradients, exact packed-document isolation, block-diagonal attention execution, target log-probability parity, mean-loss parity, aggregate gradient parity, and first-sequence-layer parity. Baseline and candidate artifacts must use the same expert-parallel size.

The heterogeneous multimodal harness derives one fixed 256-episode fixture from a captured rollout. Each eight-episode group receives a rotated permutation of rewards from \(0\) through \(1\), so every group has mean reward \(0.5\) and all 256 centered advantages are nonzero. Vision episodes occupy global indices 1, 9, 17, and 25. With CP1 and logical DP8, DP rank 1 receives 28 text and 4 vision episodes, while every other DP rank receives 32 text-only episodes. The harness runs the same serialized fixture at EP4 and EP8 with the vision tower frozen and trainable:

```bash
bash tests/run_nemo_heterogeneous_multimodal.sh
```

Actor-old log probabilities are recomputed on each fixed pack immediately before backward, which makes actor-old versus current \(k_3\) KL zero. The harness requires every physical rank to enter the synchronized vision path. Text-only ranks use a dummy visual input to preserve FSDP collective ordering, and the dummy result contributes a zero-valued autograd dependency when the vision tower is trainable. It also verifies the DP1-only modality layout, unchanged frozen vision parameters, and nonzero trainable vision updates.

GPU qualification covers dense CP, MoE CP and EP composition, heterogeneous multimodal batches, both vision freeze states, routing replay, checkpoint resume, and NeMo versus SGLang target log-probability parity.

The eight-GPU qualification results for the pinned revision are:

| Path | Topology | Result |
|---|---|---|
| Dense Qwen3.5 packed parity | CP1 and CP2 | Block-diagonal attention fired in all 12 full-attention calls. Mean losses were 9.479804 and 9.491175. |
| Qwen3.5 MoE heterogeneous multimodal, frozen vision | CP1 and EP4 | \(k_3\) KL was 0, pre-clip gradient norm was 0.2695210, and the tracked vision parameter update norm was 0. |
| Qwen3.5 MoE heterogeneous multimodal, frozen vision | CP1 and EP8 | \(k_3\) KL was 0, pre-clip gradient norm was 0.2702204, and the tracked vision parameter update norm was 0. |
| Qwen3.5 MoE heterogeneous multimodal, trainable vision | CP1 and EP4 | \(k_3\) KL was 0, pre-clip gradient norm was 0.2700823, and the tracked vision parameter update norm was 13.09052. |
| Qwen3.5 MoE heterogeneous multimodal, trainable vision | CP1 and EP8 | \(k_3\) KL was 0, pre-clip gradient norm was 0.2706632, and the tracked vision parameter update norm was 13.09165. |
| Qwen3.5 MoE actor step | CP2 and EP8 | All 40 routing layers replayed at least twice, including activation-checkpoint recomputation. Loss was -0.1109375 and gradient norm was 329.1560. |
| Dense state-dict conversion | CP2 | Converted 474 tensors totaling 2,214,532,352 bytes. |
| MoE grouped-expert conversion | CP2 and EP8 | Converted 1,026 tensors totaling 70,214,367,712 bytes. |

The EP4 frozen and trainable runs used 48,742 to 48,880 MiB and 52,456 to 52,594 MiB of NVML process memory after the optimizer step. The EP8 frozen and trainable runs used 42,432 to 42,956 MiB and 42,600 to 43,178 MiB. Rank 0 was not the highest-memory process in any case.
