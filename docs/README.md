# slim Documentation

slim is an on-policy RL training framework for vision language models.
It orchestrates a **rollout engine** (SGLang) that generates sequences and a **training backend** (NeMo AutoModel with FSDP2, context parallelism, and expert parallelism) that updates the policy.

The basic training loop follows:
1. Rollout engines generate sequences from prompts, recording tokens, and scores each attempt with reward into [Episodes and Trajectories](data-layout.md).
2. The training backend computes advantages and updates the policy via a [policy gradient objective](training-loss.md).
3. Updated weights are [synced back](placement.md) to the rollout engines.

## Docs

| Document | Scope |
|----------|-------|
| [Data Layout](data-layout.md) | Episode/Trajectory hierarchy, dataset columns, source-token alignment, VLM processor output, routing replay, flatten-pad-partition |
| [SGLang Config](sglang-config.md) | Engine deployment, parameter pass-through, PD disaggregation, speculative decoding, FP8 inference, fault tolerance |
| [Training Loss](training-loss.md) | Advantage estimators (PPO-GAE, GRPO, GSPO), policy surrogates, KL penalty, mismatch correction |
| [Placement](placement.md) | GPU allocation, colocate modes, step timeline, weight update paths |
| [Dev Utils](dev-utils.md) | Profiling, debugging, reproducibility, tests |
| [Customization](customization.md) | All `--*-path` extension points, multi-turn/agentic adaptation |
| [NeMo AutoModel Backend](Nemo-Automodel.md) | NeMo training backend scope, packed mixed-modality EP+CP design, loss normalization, checkpoints, and SGLang synchronization |
