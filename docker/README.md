# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which based on official sglang release. We will store the patch on those versions.
2. latest version, which aligns to `lmsysorg/sglang:latest`.

current stable version is:
- sglang v0.5.9 (bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2)

The command to build:

```bash
just release
```

Before each update, we will test the following models:

- Qwen3-1.7B FSDP math
- Qwen3-4B FSDP true-on-policy
- Qwen3-VL-4B FSDP
