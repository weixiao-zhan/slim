# Low Precision Training

- [FP8 rollout and BF16 training](#fp8-rollout-and-bf16-training)
- [INT4 QAT Training](#int4-qat-training)

## FP8 rollout and BF16 training

You can run FP8 rollout simply by setting `--hf-checkpoint` with a blockwise quantized HuggingFace checkpoint, which can be converted by:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_MODEL \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Please ensure that the converted checkpoint points to a directory where the `config.json` contains the correct `quantization_config` so that minislime can automatically use FP8 quantization during weight updates.

## INT4 QAT Training

This guide provides examples for INT4 STE (Straight-Through Estimator) training and INT4 inference. Utilizing INT4 inference significantly improves throughput, thereby accelerating the training pipeline (specifically during the rollout generation phase).

### Quick Start

1. Convert HuggingFace Weights to INT4
Use the `tools/convert_hf_to_int4_direct.py` script to convert BF16 weights to INT4 format. Ensure that the `--hf-checkpoint` parameter points to a directory where `config.json` contains the correct `quantization_config`. minislime will automatically utilize INT4 quantization during weight updates.

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /path/to/your/original/models \
  --save-dir /path/to/your/save/models
```

Note: If you only want INT4 rollout, you only need to set `--hf-checkpoint` to the converted INT4 checkpoint.
