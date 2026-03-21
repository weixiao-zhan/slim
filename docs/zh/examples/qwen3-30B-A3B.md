# 8xH100 训练 Qwen3-30B-A3B

## 环境准备

搭建环境、下载模型和数据均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 Qwen3-30B-A3B 即可。无需进行 checkpoint 格式转换，FSDP 可直接加载 HuggingFace checkpoint。

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-qwen3-30B-A3B.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-qwen3-30B-A3B.sh](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3-30B-A3B.sh) 中与 MoE 相关的部分。

1. 为了支持在 8xH800 环境中运行 Qwen3-30B-A3B，可以开启 FSDP CPU offload 和梯度检查点以节省显存：

   ```bash
   PERF_ARGS=(
      --gradient-checkpointing
      --fsdp-cpu-offload
      --use-dynamic-batch-size
      --max-tokens-per-gpu 4608
   )
   ```

2. 开启 sglang 支持的 moe 优化，当前配置为 ep8：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.7
      --sglang-ep-size 8
      --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   )
   ```

   类似地，也可以加入 dp attention，例如配置上：

   ```bash
      --sglang-enable-dp-attention
      --sglang-dp-size 8
   ```

### bf16 训练 fp8 推理

slime 还支持 bf16 训练，fp8 推理。对于 Qwen3-30B-A3B 模型，只需要下载如下模型：

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8
```

并将 `--hf-checkpoint` 替换为：

```bash
#--hf-checkpoint /root/Qwen3-30B-A3B
--hf-checkpoint /root/Qwen3-30B-A3B-FP8
```

即可触发 fp8 训练。目前我们会将 bf16 权重直接 cast 为 fp8，后续会逐渐添加对精度影响更小的量化方案。

### 多机支持

对于多机环境，需要进行如下的几点修改：
- 将训练模型，数据放在所有机器都可以访问到的路径上；
- 设置各台机器都可以访问到的 `MASTER_ADDR` 之外；
- 如果不需要，可以去掉 `--fsdp-cpu-offload`，多机 FSDP 分片会降低每张 GPU 的显存占用。

除此之外，还可以进行如下的修改：

- 当总卡数并不能被 expert 总数乘除时，可以使用 `--sglang-ep-num-redundant-experts` 来增加冗余的 expert，例如对于 24 卡的场景，可以配置：

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
