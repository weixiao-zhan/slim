# Reproducibility

Reproducibility is a bedrock of scientific progress. By combining SGLang's [deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) and FSDP's deterministic mode, slim can provide fully deterministic (bitwise) experiment reproduction.

To enable deterministic training, you need to uninstall flash attention 3 via `pip uninstall flash_attn_3 -y` and set:

```bash
  # sglang config
  --sglang-enable-deterministic-inference
  --sglang-attention-backend flashinfer

  # training config
  --true-on-policy-mode
```

And set the following environment variables:

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

Use `--true-on-policy-mode` with your training script for bitwise reproducible training.
