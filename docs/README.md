# minislime Documentation

## [Rollout](rollout/README.md)
SGLang setup, parameter pass-through, rollout args, dynamic sampling, partial rollout, evaluation.
- [SGLang Config](rollout/sglang-config.md) -- Multi-model serving, PD disaggregation, YAML deployment
- [Slime Router](rollout/slime-router.md) -- Training-oriented HTTP router
- [Speculative Decoding](rollout/speculative-decoding.md) -- MTP draft model acceleration
- [On-Policy Distillation](rollout/on-policy-distillation.md) -- Teacher-student distillation
- [Fault Tolerance](rollout/fault-tolerance.md) -- Heartbeat-based recovery
- [PD Disaggregation](rollout/pd-disaggregation.md) -- Prefill-decode separation

## [Training](training/README.md)
Installation, GPU allocation, checkpoints, data format, RL algorithms (GRPO/PPO), multi-node, FAQ.
- [Low Precision](training/low-precision.md) -- FP8 inference, INT4 QAT
- [Reproducibility](training/reproducibility.md) -- Deterministic bitwise training
- [Debugging](training/debug.md) -- Precision alignment, separate debugging
- [Profiling](training/profiling.md) -- Rollout performance analysis
- [CI](training/ci.md) -- GitHub Actions workflow

## [Customization](customization/README.md)
All extension points: rollout functions, reward models, filters, loss functions, logging, multi-turn/agentic adaptation.

## [Examples](examples/README.md)
- [Qwen3-30B-A3B (MoE)](examples/qwen3-30B-A3B.md)
- [GLM-4.7-Flash (MoE + MTP)](examples/glm4.7-30B-A3B.md)
- [Qwen3-4B SFT](examples/qwen3-4b-base-openhermes.md)

Also see runnable [examples/](../examples/) for VLM, search, and tool-use workflows.
