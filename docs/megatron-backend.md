# Megatron Backend

Status: experimental. The actor training path is implemented and covered by CPU contract tests. GPU parity and scale tests are required before production use.

## Scope

The backend keeps slim responsible for rollout and RL semantics while delegating distributed model execution to Megatron:

| Owner | Responsibilities |
|-------|------------------|
| slim | `Episode`, rollout, rewards, advantages, policy loss, Ray actor lifecycle, metrics |
| Megatron Bridge | HuggingFace config and weight conversion, MCore providers, VLM inputs, HuggingFace weight export |
| Megatron Core | TP, SP, CP, EP, non-interleaved PP, DDP, distributed optimizer, schedules, distributed checkpoints |

The backend does not use Megatron datasets, the Megatron pretraining loop, Megatron CLI arguments, or `megatron.training.global_vars`.

## Configuration

Select the backend with:

```bash
--training-backend megatron
```

The parallelism arguments use Bridge provider field names directly:

| Argument | Default | Meaning |
|----------|---------|---------|
| `--tensor-model-parallel-size` | `1` | Tensor parallel size \(T\) |
| `--pipeline-model-parallel-size` | `1` | Pipeline parallel size \(P\) |
| `--context-parallel-size` | `1` | Context parallel size \(C\) |
| `--sequence-parallel` | false | Shard sequence activations across TP ranks |
| `--expert-model-parallel-size` | `1` | Expert parallel size for MoE models |
| `--use-distributed-optimizer` | true | Shard optimizer state across the DP and CP domain |

The optimizer arguments are:

```text
--optimizer adam
--lr
--lr-min
--lr-decay-style
--lr-decay-iters
--lr-warmup-iters
--weight-decay
--adam-beta1
--adam-beta2
--adam-eps
--clip-grad
--gradient-checkpointing
```

Given training world size \(W\), the data parallel size is:

\[
D = \frac{W}{TPC}
\]

The backend requires:

\[
W \bmod (TPC) = 0
\]

For expert tensor parallel size \(E_T\) and expert parallel size \(E\), MCore also requires:

\[
W \bmod (E_T \, E \, P) = 0
\]

SP requires \(T>1\). EP requires an MoE provider. For Qwen3.5 Gated DeltaNet, the backend also requires:

\[
TC \mid \gcd(N_{K,\mathrm{GDN}},N_{V,\mathrm{GDN}})
\]

Qwen3.5-27B has \(N_{K,\mathrm{GDN}}=16\) and \(N_{V,\mathrm{GDN}}=48\), so \(TC\) must divide \(16\).

Virtual pipeline parallelism is unsupported. The CLI rejects VP options, provider validation rejects derived VP fields, and schedule selection always uses `vp_size=None`.

## Implemented Capabilities

The actor backend implements:

- GRPO policy training.
- TP-sharded selected-token log probabilities without gathering the vocabulary.
- TP-sharded entropy.
- Sequence parallelism within TP.
- Packed THD context parallelism.
- Tensor, context, expert, and non-interleaved pipeline parallel composition.
- Dynamic or fixed episode grouping into packed microbatches.
- Qwen3.5-VL batches containing both visual and text-only trajectories.
- R3 rollout routing replay for MCore MoE routers.
- MCore distributed optimizer state.
- MCore distributed checkpoints.
- Streaming Bridge HuggingFace weight export to non-colocated SGLang engines.

The following configurations fail during argument validation:

- Critic training and PPO GAE.
- GSPO.
- Reference models, KL loss, and reference updates.
- Custom policy loss.
- Mismatch correction and mismatch metrics.
- PEFT.
- Parameter include or freeze lists.
- Colocated rollout and trainer sleep or wake.
- Async checkpoint save.
- Checkpoints without optimizer state.
- Virtual pipeline parallelism.

## Packed Trajectories

Each model microbatch is one physical THD stream. Every trajectory is padded independently to:

\[
A = \operatorname{lcm}\left(
\begin{cases}
2C & C>1 \\
1 & C=1
\end{cases},
\begin{cases}
CT & \text{if SP is enabled} \\
1 & \text{otherwise}
\end{cases}
\right)
\]

For tokens \(t_0,\ldots,t_{L-1}\), edge-aligned rollout values become token-aligned rows:

```text
tokens:       t_0, t_1, ..., t_(L-1)
labels:       t_1, t_2, ..., ignore
loss mask:    m_0, m_1, ..., 0
edge values:  e_0, e_1, ..., 0
```

`PackedLayout` carries:

- Tokens, labels, loss masks, and position IDs.
- Advantages, returns, rollout log probabilities, and actor-old log probabilities.
- Logical and physical cumulative sequence lengths.
- Per-trajectory physical token and edge ranges.
- Visual tensor and placeholder ranges.
- Token-aligned R3 expert IDs.

Bridge constructs `PackedSeqParams` from the layout. The same Bridge CP index is used for labels, masks, RL features, sequence IDs, and R3.

Text GPT inputs are sliced by the trainer before model forward. Qwen3.5-VL receives the full `[1,T]` THD input because its Bridge model applies the packed CP index internally to language embeddings, visual masks, mRoPE positions, labels, and loss masks. External RL tensors use the identical index.

## Mixed Vision And Text

One local microbatch can contain:

```python
[
    Episode(tokens=..., multimodal_inputs={...}),
    Episode(tokens=..., multimodal_inputs=None),
    Episode(tokens=..., multimodal_inputs={...}),
]
```

The backend does not run a processor during training. It consumes the processor outputs stored in each `Episode`, concatenates media tensors only for trajectories containing that modality, and preserves trajectory order and placeholder ranges.

The current mixed-modality contract targets the Bridge Qwen3.5-VL provider. Qwen3.5 ignores `mm_token_type_ids`; other VLM providers require separate GPU acceptance before use.

Dynamic packing charges each trajectory's aligned physical length against the token capacity and synchronizes the microbatch count across DP ranks.

## Forward And Backward

The provider sets `calculate_per_token_loss=True` before MCore DDP construction. After optimizer construction, each model config receives:

```python
config.calculate_per_token_loss = True
config.finalize_model_grads_func = partial(
    finalize_model_grads,
    pg_collection=pg_collection,
)
config.grad_scale_func = optimizer.scale_loss
```

Each optimizer step calls the MCore schedule with one model chunk per PP stage. The loss callback returns:

```python
loss_sum, local_active_tokens, metrics
```

MCore sums gradients over CP and DP and divides by the global active-token count.

For trajectory-mean policy reduction:

\[
\mathcal{L}
=
\frac{1}{B}
\sum_{i=1}^{B}
\frac{\sum_j m_{ij}\ell_{ij}}
{\max\left(\sum_j m_{ij},1\right)}
\]

Let \(n_i=\sum_j m_{ij}\) and \(N=\sum_i n_i\). The pre-normalization token weight is:

\[
w_{ij}=\frac{N}{B\max(n_i,1)}
\]

For token-sum policy reduction, the weight is:

\[
w_{ij}=\frac{N}{B}
\]

Entropy and KL helpers retain trajectory-mean weights independently from the policy reduction setting.

## R3

R3 is available only for an MoE provider with MCore routing replay enabled. Dense providers reject `--use-rollout-routing-replay`.

The rollout payload remains:

```text
[num_edges, num_hidden_layers, top_k]
```

Packing appends a valid expert ID \(0\) for each trajectory tail and physical padding row. Before router replay, the trainer applies:

1. The packed CP index.
2. The TP-rank SP slice when SP is enabled.
3. The model-local global layer selection.

Routers are keyed by one-based global `layer_number`, so PP stages do not depend on router construction order. Original forwards use `REPLAY_FORWARD`. A backward hook switches the model-local routers to `REPLAY_BACKWARD` before activation recomputation.

R3 with activation recomputation requires a GPU acceptance test that verifies route equality and empty replay queues after every optimizer step.

## Checkpoints

`--load` accepts an HF checkpoint directory containing `config.json`, a direct slim Megatron distributed checkpoint directory, or a save root containing `rollout_<id>` directories. A save root resolves to its latest complete distributed checkpoint.

HF checkpoints initialize Bridge conversion. Distributed checkpoints restore:

- Model sharded state.
- Distributed optimizer state.
- Optimizer scheduler state.
- Python, NumPy, PyTorch, CUDA, and MCore CUDA RNG state.
- Rollout IDs and global optimizer step.
- World size, TP, PP, CP, EP, expert-TP, and SP metadata.

`save_model(rollout_id)` writes:

```text
<save>/rollout_<rollout_id:08d>
```

World-size and TP, PP, CP, EP, expert-TP, and SP changes are rejected before sharded state is loaded.

## Weight Synchronization

Bridge exports canonical HuggingFace names as a stream:

```python
for name, tensor in bridge.export_hf_weights(model):
    ...
```

The backend groups this stream into bounded buckets and forwards each bucket through Slim's distributed SGLang transport. It does not materialize a complete HuggingFace state dictionary on rank zero.

## Dependencies And Validation

Megatron remains optional. Selecting FSDP does not import Bridge or MCore. Selecting Megatron requires compatible installations of Megatron Bridge, Megatron Core, Transformer Engine, and the model-specific kernels.

The inspected revisions are:

| Component | Revision |
|-----------|----------|
| Megatron Core | `2aa3645b2b31d748d543bd8066fb81cf1f1502d8` |
| Megatron Bridge | `39b316a4f918f31de3d141eefbc2ec2fc2d3cc8b` |

The inspected Bridge revision declares `transformers>=5.8.1,<5.9.0`, while Slim uses a newer Transformers release. The repository does not declare a Megatron optional dependency. The backend requires an explicitly managed environment with a compatible Bridge, MCore, and Transformers stack.

CPU tests cover lazy imports, argument validation, topology checks, schedule selection, packed layout, mixed trajectory metadata, loss normalization, R3 ownership, model construction, checkpoint scaffolding, weight buckets, and Trainer contracts.

GPU validation remains required for:

- Dense GPT parity under TP/SP/CP and \(P=1,2\).
- Qwen3.5 GDN parity under supported \(T,C\) pairs.
- Qwen3.5-VL mixed visual and text training under PP and CP.
- R3 with activation recomputation.
- Distributed optimizer save and resume.
- SGLang logits after streamed weight updates.
