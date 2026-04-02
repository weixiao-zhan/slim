# mini-slime

**mini-slime** is a lightweight fork of [slime](https://github.com/THUDM/slime) (v0.2.3) that is better suited for research and prototype.

We love slime — it is versatile, well-customizable. 
Meanwhile, we also love HF Transformers and PyTorch FSDP — they are simple and sufficient.
However, slime is [deprecating its FSDP backend](https://github.com/THUDM/slime/commit/0d0b974d) in favor of Megatron-only. 

Thus we forked `mini-slime`, focusing on small agentic VLMs training — keeping Slime's customizability and efficient RL orchestration while standardizing the dataset contract around a small fixed schema and enhancing concurrency. With FSDP backend, you can directly customize the HF model. 

### 🏗️ Roadmap

- [x] Add PEFT
- [x] Overlap actor forward pass for log-probs and ref-log-probs and critif forward pass for values.
- [x] Unifiy data layout to use tokens.
- [x] Improve dataset loading: datasets now use a finite set of supported columns (`prompt`, `label`, and optional multimodal / control columns) and defer `apply_chat_template` to rollout time. ([upstream discussion](https://github.com/THUDM/slime/issues/1231))
- [x] Remove megatron dependency. No more mbridge converter and docker. It's `uv` friendly now.

### 🚧 Known Broken Features

- **R2 (Routing Replay) and R3 (Rollout Routing Replay)** were implemented for Megatron actor only. HF transformers does not support router replay yet.
- **sglang v0.5.9** release won't load Qwen3-VL vision weight correctly. [fix](https://github.com/sgl-project/sglang/commit/d566816d838ce92d3ae044209f7d67eaa58ce74a)


## What Changed from Upstream slime

| | slime (v0.2.3+) | minislime |
|---|---|---|
| Training backend | Megatron-LM (FSDP removed) | PyTorch FSDP2 (Megatron removed) |
| Model loading | Megatron checkpoint format (`torch_dist`) | HuggingFace `from_pretrained()` |
| Parallelism | TP, PP, CP, EP, DP | DP only (HYBRID sharding) |
| Dependencies | Megatron-LM, mbridge, apex, TransformerEngine | HuggingFace Transformers, accelerate (now, its `uv` friendly!) |
| Model size target | Up to 355B+ (multi-node, full parallelism) | Up to ~30B (where most small open-source models tops) |


## Quick Start
