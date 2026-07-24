# slim

**slim** is a lightweight fork of [slime](https://github.com/THUDM/slime) (v0.2.3) that is better suited for VLM and research and prototype.

We love slime — it is well-architected and customizable. 
We also love NeMo AutoModel and FSDP2 — they provide context and expert parallelism for VLM and research.
However, slime [deprecated FSDP backend](https://github.com/THUDM/slime/commit/0d0b974d) in favor of Megatron. 

Thus we forked `slim` — keeping Slime's customizability and efficient RL orchestration while optimizing for VLMs.

### 🏗️ Features
- [x] [Context Parallel](docs/Nemo-Automodel.md#context-parallel-layouts)
- [x] Expert parallel training for MoE models
- [ ] True parallel generate function
- [x] Decouple colocate critic and colocate rollout placement (train PPO on one GPU).
- [x] FP8 inference (per-block with fp32 or UE8M0 scale)
- [x] Rollout Routing Replay for MoE and use processor output format to avoid vision token drifts. 
- [x] Over-sampled groups that are partial complete are not discarded and save to next step.
- [x] Support mixed modality (pure text + vision) training batch.
- [x] Overlap actor forward pass for ref-log-probs and critic forward pass for values.
- [x] Unifiy data layout to token centric.
- [x] Improve dataset loading: datasets now use a finite set of supported columns (`prompt`, `label`, and optional multimodal / control columns) and defer `apply_chat_template` to rollout time. ([upstream discussion](https://github.com/THUDM/slime/issues/1231))
- [x] Remove megatron dependency. No more mbridge converter and docker. It's `uv` friendly now.

## What Changed from Upstream slime

| | slime (v0.2.3+) | slim |
|---|---|---|
| Training backend | Megatron-LM | NeMo AutoModel with FSDP2 |
| Model loading | Megatron checkpoint format (`torch_dist`) | NeMo AutoModel from Hugging Face checkpoints |
| Parallelism | TP, PP, CP, EP, ETP, DP | FSDP2, CP, EP |
| Dependencies | Megatron-LM, mbridge, apex, TransformerEngine | NeMo AutoModel and Hugging Face Transformers |
| Model size target | Several hundred B to 1T | Up to 100B |

## Environment

[`pyproject.toml`](pyproject.toml) is the dependency source of truth.

```bash
uv sync --extra dev
```
