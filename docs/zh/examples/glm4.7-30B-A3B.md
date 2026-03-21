# 8×H100 训练 GLM-4.7-Flash

## 环境准备

搭建环境和数据均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 GLM-4.7-Flash 即可。无需进行 checkpoint 格式转换，FSDP 可直接加载 HuggingFace checkpoint。

### 前置条件

GLM-4.7-Flash 需要 **transformers ≥ 5.0** 以支持 `Glm4MoeLiteForCausalLM` 架构。请安装或升级：

```bash
pip install "transformers>=5.0"
```

### 下载模型

```bash
hf download THUDM/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash
```

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-glm4.7-30B-A3B-8gpus.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-glm4.7-30B-A3B-8gpus.sh](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B-8gpus.sh) 中的关键部分。

#### MoE 配置

GLM-4.7-Flash 是一个 MoE（混合专家）模型，包含 64 个路由专家（top-4 激活）和 1 个共享专家。共 47 层：1 层 dense 层 + 46 层 MoE 层。

1. 为了支持在 8×H100 环境中运行 GLM-4.7-Flash，可以开启 FSDP CPU offload 和梯度检查点以节省显存：

   ```bash
   PERF_ARGS=(
      --gradient-checkpointing
      --fsdp-cpu-offload
      --use-dynamic-batch-size
      --max-tokens-per-gpu 4608
   )
   ```

2. 开启 SGLang 支持的 MoE 优化，使用 DP attention：

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

#### MTP 投机解码（推理加速）

GLM-4.7-Flash 包含 1 层 MTP（Multi-Token Prediction）层，可用于推理时的投机解码来加速 rollout 生成。要启用此功能，在 `SGLANG_ARGS` 中添加以下配置：

```bash
SGLANG_ARGS=(
   ...
   # MTP 投机解码 (EAGLE)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
)
```

这会让 SGLang 使用模型的 MTP 层作为 EAGLE 风格投机解码的 draft 模型。MTP 层预测多个未来 token，SGLang 并行验证它们，从而加速生成。

> ⚠️ **注意**：投机解码会占用额外的 GPU 显存。如果遇到 OOM 问题，可以尝试降低 `--sglang-mem-fraction-static` 或关闭投机解码。

#### MTP 训练

slime 也支持将 MTP 层与主模型联合训练，适用于已实现 MTP 权重转换的模型（如 MiMo、GLM-4.5）。启用时，相关参数如下：

```bash
# 在模型配置中添加 MTP 层数
MODEL_ARGS+=(--mtp-num-layers 1)

# 启用 MTP 训练
SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

- `--mtp-num-layers 1`：告知训练后端从 checkpoint 中加载 MTP 层。
- `--enable-mtp-training`：启用 MTP 层的梯度计算。不设置此标志时，MTP 层会被加载但冻结。
- `--mtp-loss-scaling-factor 0.2`：MTP loss 相对于主策略 loss 的权重，默认为 0.2。

### 多机支持

对于多机训练（例如 2×8 H100），使用多机脚本：

```bash
cd /root/slime
export BASE_DIR=/shared/path  # 所有节点都可以访问的路径
bash scripts/run-glm4.7-30B-A3B.sh
```

对于多机环境，需要进行如下修改：

- 将训练模型、数据放在所有机器都可以访问到的路径上；
- 设置各台机器都可以访问到的 `MASTER_ADDR`；
- 如果不需要，可以去掉 `--fsdp-cpu-offload`，多机 FSDP 分片会降低每张 GPU 的显存占用。

当总卡数并不能被 expert 总数（64）乘除时，可以使用 `--sglang-ep-num-redundant-experts` 来增加冗余的 expert。例如对于 24 卡的场景：

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
