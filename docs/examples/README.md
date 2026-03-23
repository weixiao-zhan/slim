# Model Training Examples

End-to-end training configurations for different model architectures.

## Dense Models

- [Qwen3-4B SFT](qwen3-4b-base-openhermes.md) -- Supervised fine-tuning with OpenHermes data

## MoE Models

- [Qwen3-30B-A3B](qwen3-30B-A3B.md) -- MoE RL training on 8xH100 with EP and FP8 inference
- [GLM-4.7-Flash](glm4.7-30B-A3B.md) -- MoE with MTP speculative decoding and online draft training

## Runnable Examples

See the [examples/](../../examples/) directory for complete runnable workflows:

- **[geo3k_vlm](../../examples/geo3k_vlm/)** -- VLM single-turn RL on GEO3K dataset
- **[search-r1](../../examples/search-r1/)** -- Multi-turn search with tool calling
- **[tau-bench](../../examples/tau-bench/)** -- Agentic multi-turn tool use environment
