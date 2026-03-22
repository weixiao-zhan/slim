# mini-slime

**mini-slime** is a lightweight fork of [slime](https://github.com/THUDM/slime) (v0.2.3) that is better suitted for research and prototype.

We love slime — it is versatile, well-customizable. However, slime is [deprecating its FSDP backend](https://github.com/THUDM/slime/commit/0d0b974d) in favor of Megatron-only. 
Meanwhile, we also love HF Transformers and PyTorch FSDP — they are simple and sufficient.

## Roadmap

- [] multi-thread dataloading
- [] Add PEFT

## What Changed from Upstream slime

| | slime (v0.2.3+) | minislime |
|---|---|---|
| Training backend | Megatron-LM (FSDP removed) | PyTorch FSDP2 (Megatron removed) |
| Model loading | Megatron checkpoint format (`torch_dist`) | HuggingFace `from_pretrained()` |
| Parallelism | TP, PP, CP, EP, DP | DP only (HYBRID sharding) |
| Dependencies | Megatron-LM, mbridge, apex, TransformerEngine | HuggingFace Transformers, accelerate (now, its `uv` friendly!) |
| Model size target | Up to 355B+ (multi-node, full parallelism) | Up to ~30B (where most small open-source models tops) |

### Known Broken Features

- **R2 (Routing Replay) and R3 (Rollout Routing Replay)** were implemented for Megatron actor only.

## Quick Start

