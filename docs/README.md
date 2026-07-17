# slim Documentation

slim is an on-policy RL training framework for vision language models.
It orchestrates a **rollout engine** (SGLang) that generates sequences and a **training backend** (FSDP2) that updates the policy.

The basic training loop follows:
1. Rollout engines generate sequences from prompts, recording tokens, and scores each sequence with reward into [Episodes](data-layout.md).
2. The training backend computes advantages and updates the policy via a [policy gradient objective](training-loss.md).
3. Updated weights are [synced back](placement.md) to the rollout engines.

## Docs

| Document | Scope |
|----------|-------|
| [Data Layout](data-layout.md) | Episode lifecycle, dataset columns, edge alignment, VLM processor output, routing replay |
| [SGLang Config](sglang-config.md) | Engine deployment, parameter pass-through, PD disaggregation, speculative decoding, FP8 inference, fault tolerance |
| [Training Loss](training-loss.md) | Advantage estimators (PPO-GAE, GRPO, GSPO), policy surrogates, KL penalty, mismatch correction |
| [Placement](placement.md) | GPU allocation, colocate modes, step timeline, weight update paths |
| [Megatron Backend Design](megatron-backend.md) | Lightweight MCore backend contract for TP, SP, CP, non-interleaved PP, mixed VLM batches, and routing replay |
| [Dev Utils](dev-utils.md) | Profiling, debugging, reproducibility, tests |
| [Customization](customization.md) | All `--*-path` extension points, multi-turn/agentic adaptation |
