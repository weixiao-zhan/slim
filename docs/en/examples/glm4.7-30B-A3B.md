# GLM-4.7-Flash with 8×H100


## Environment Preparation

The environment setup and data are the same as for the Qwen3-4B model. You can refer to [Example: Qwen3-4B Model](qwen3-4B.md), replacing mentions of Qwen3-4B with GLM-4.7-Flash. No checkpoint conversion is needed -- FSDP loads HuggingFace checkpoints directly.

### Prerequisites

GLM-4.7-Flash requires **transformers ≥ 5.0** for the `Glm4MoeLiteForCausalLM` architecture. Install or upgrade:

```bash
pip install "transformers>=5.0"
```

### Download Model

```bash
hf download THUDM/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash
```

## Run Training

Execute the training script:

```bash
cd /root/slime
bash scripts/run-glm4.7-30B-A3B-8gpus.sh
```

### Parameter Introduction

Here, we will briefly introduce the key parts in the [run-glm4.7-30B-A3B-8gpus.sh](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B-8gpus.sh) script.

#### MoE Configuration

GLM-4.7-Flash is a Mixture-of-Experts (MoE) model with 64 routed experts (top-4 activation) and 1 shared expert. It has 47 layers: 1 dense layer + 46 MoE layers.

1.  To support running GLM-4.7-Flash on 8×H100, you can enable FSDP CPU offloading and gradient checkpointing to save GPU memory:

    ```bash
    PERF_ARGS=(
       --gradient-checkpointing
       --fsdp-cpu-offload
       --use-dynamic-batch-size
       --max-tokens-per-gpu 4608
    )
    ```

2.  Enable MoE optimization in SGLang with DP attention:

    ```bash
    SGLANG_ARGS=(
       --rollout-num-gpus-per-engine 8
       --sglang-mem-fraction-static 0.7
       --sglang-enable-dp-attention
       --sglang-dp-size 8
       --sglang-enable-dp-lm-head
       --sglang-moe-dense-tp-size 1
       ...
    )
    ```

#### MTP Speculative Decoding (Inference Acceleration)

GLM-4.7-Flash includes 1 MTP (Multi-Token Prediction) layer, which can be used for speculative decoding during inference to speed up rollout generation. To enable this, add the following to `SGLANG_ARGS`:

```bash
SGLANG_ARGS=(
   ...
   # MTP speculative decoding (EAGLE)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
)
```

This enables SGLang to use the model's MTP layer as a draft model for EAGLE-style speculative decoding. The MTP layer predicts multiple future tokens, and SGLang verifies them in parallel, leading to faster generation.

> ⚠️ **Note**: Speculative decoding requires additional GPU memory. If you encounter OOM issues, try reducing `--sglang-mem-fraction-static` or disabling speculative decoding.

#### MTP Training

slime also supports training MTP layers jointly with the main model for models that have MTP weight conversion implemented (e.g., MiMo, GLM-4.5). When enabled, the relevant arguments are:

```bash
# Add MTP layer count to model config
MODEL_ARGS+=(--mtp-num-layers 1)

# Enable MTP training
SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

- `--mtp-num-layers 1`: Tells the training backend to load the MTP layer from the checkpoint.
- `--enable-mtp-training`: Enables gradient computation for MTP layers. Without this flag, the MTP layer is loaded but frozen.
- `--mtp-loss-scaling-factor 0.2`: Weight of the MTP loss relative to the main policy loss. Default is 0.2.

### Multi-Node Support

For multi-node training (e.g., 2×8 H100), use the multi-node script:

```bash
cd /root/slime
export BASE_DIR=/shared/path  # accessible by all nodes
bash scripts/run-glm4.7-30B-A3B.sh
```

Key modifications for multi-node:

  - Place the model and data on a path accessible by all nodes.
  - Set `MASTER_ADDR` to an address accessible by all nodes.
  - Remove `--fsdp-cpu-offload` if not needed. Multi-node FSDP sharding reduces per-GPU memory usage.

When the total number of GPUs is not a multiple or divisor of the total number of experts (64), you can use `--sglang-ep-num-redundant-experts` to add redundant experts. For example, in a 24-GPU scenario:

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 24
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 24
   --sglang-enable-dp-attention
   --sglang-dp-size 3
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-ep-num-redundant-experts 16
)
```
