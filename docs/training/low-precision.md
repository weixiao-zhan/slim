# Low Precision Training

- [FP8 rollout and BF16 training](#fp8-rollout-and-bf16-training)
- [INT4 QAT Training](#int4-qat-training)

## FP8 rollout and BF16 training

Use `--hf-checkpoint` for the FP8 model (rollout inference) and `--load` for the BF16 model (training init):

```bash
python3 train_async.py \
    --hf-checkpoint /path/to/Model-FP8 \
    --load /path/to/Model \
    --save /path/to/output \
    ...
```

`--load` auto-detects whether the path is a HuggingFace checkpoint (BF16 init) or a slim DCP checkpoint (training resume). When it detects an HF checkpoint, the actor uses it for `from_pretrained` weight initialization while `--hf-checkpoint` is used only for the sglang rollout engine and tokenizer.

Many models have official FP8 variants on HuggingFace (e.g. `Qwen/Qwen3-VL-4B-Instruct-FP8`). You can also convert a BF16 model to FP8:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_MODEL \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Ensure the FP8 checkpoint's `config.json` contains the correct `quantization_config` so that sglang can automatically re-quantize BF16 weight updates to FP8.

See `examples/skypilot/launch_async_fp8.yaml` for a complete example.

## INT4 QAT Training

This guide provides examples for INT4 STE (Straight-Through Estimator) training and INT4 inference. Utilizing INT4 inference significantly improves throughput, thereby accelerating the training pipeline (specifically during the rollout generation phase).

### Quick Start

1. Convert HuggingFace Weights to INT4
Use the `tools/convert_hf_to_int4_direct.py` script to convert BF16 weights to INT4 format. Ensure that the `--hf-checkpoint` parameter points to a directory where `config.json` contains the correct `quantization_config`. slim will automatically utilize INT4 quantization during weight updates.

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /path/to/your/original/models \
  --save-dir /path/to/your/save/models
```

Note: If you only want INT4 rollout, you only need to set `--hf-checkpoint` to the converted INT4 checkpoint.
