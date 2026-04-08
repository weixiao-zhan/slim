# slim

**slim** is a lightweight fork of [slime](https://github.com/THUDM/slime) (v0.2.3) that is better suited for research and prototype.

We love slime — it is versatile, well-customizable. 
Meanwhile, we also love HF Transformers and FSDP — they are simple and sufficient.
However, slime is [deprecating its FSDP backend](https://github.com/THUDM/slime/commit/0d0b974d) in favor of Megatron. 

Thus we forked `slim` — keeping Slime's customizability and efficient RL orchestration while using a token centric data convention.

### Use case | assumption
1. Dense model ONLY: Same size dense model often perform better than MoE. And we believe SLM on local agentic, automation tasks has much value.
2. DP ONLY: with most open-source dense model cap at 30B (Qwen3, Qwen3.5, Gemma4), data parallel is sufficient. Single H200 can easily support them.

### 🏗️ Features
- [x] Oversampled groups that are partial complete or never generated are not discarded and save to next step
- [x] Support multi-modal mixed dataset
- [x] Add PEFT (Lora / Dora)
- [x] Overlap actor forward pass for ref-log-probs and critic forward pass for values.
- [x] Unifiy data layout to token centric.
- [x] Improve dataset loading: datasets now use a finite set of supported columns (`prompt`, `label`, and optional multimodal / control columns) and defer `apply_chat_template` to rollout time. ([upstream discussion](https://github.com/THUDM/slime/issues/1231))
- [x] Remove megatron dependency. No more mbridge converter and docker. It's `uv` friendly now.

## What Changed from Upstream slime

| | slime (v0.2.3+) | slim |
|---|---|---|
| Training backend | Megatron-LM (FSDP removed) | PyTorch FSDP2 (Megatron removed) |
| Model loading | Megatron checkpoint format (`torch_dist`) | HuggingFace `from_pretrained()` |
| Parallelism | TP, PP, CP, EP, DP | DP only (HYBRID sharding) |
| Dependencies | Megatron-LM, mbridge, apex, TransformerEngine | HuggingFace Transformers (now, its `uv` friendly!) |
| Model size target | Up to 355B+ (multi-node, full parallelism) | Up to ~30B (where most small open-source models tops) |

