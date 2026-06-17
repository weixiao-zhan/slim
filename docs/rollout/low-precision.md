# Low Precision Inference


Rollout takes up a large fraction of wall-clock time in RL, and low-precision inference can improve throughput. 
However the cost is added complexity on rollout/training numerical gap and weight-sync path.
Slim can run the rollout engine in low precision while training stays full precision. 
The resulting rollout/training gap is corrected at the loss level by the policy surrogate (`--policy-surrogate`).
The rest of this page covers the parameters, the supported FP8 formats, and how the weight sync recipe.

## Parameters

```bash
python3 train_async.py \
    --hf-checkpoint /path/to/Model-FP8 \   # FP8: rollout engine + tokenizer
    --load          /path/to/Model \       # full precision: training init
    --save          /path/to/output \
    --policy-surrogate cis \               # correct rollout/training mismatch
    --old-logprob-source rollout \
    ...
```

- `--hf-checkpoint` points at the FP8 checkpoint. sglang serves it in FP8, and slim reads its `quantization_config` to re-quantize weight updates to the same format during every weight sync.
- `--load` auto-detects whether the path is a HuggingFace checkpoint (full-precision training init via `from_pretrained`) or a slim DCP checkpoint (training resume). When it is an HF checkpoint, the actor uses it for weight init while `--hf-checkpoint` is used only for the rollout engine and tokenizer.

## Supported quantization recipe

### block-FP8
- e4m3 weights, `128×128` blocks, dynamic activation quantization.
- A multiplicative per-block scale stored as `<weight>.weight_scale_inv` (dequant = `q * scale`).
- Per-tensor and per-channel FP8 are **not** supported.

Two block-scale formats are supported:

- **fp32** (default): a full float32 per-block scale, consumed by sglang's Triton / DeepGEMM block-FP8 GEMM on any supported GPU.
- **ue8m0** (`scale_fmt: "ue8m0"`): DeepSeek-V3.1 / DeepGEMM-on-Blackwell style — each block scale is rounded up to a power of two. On disk it is a fp32 power-of-two; the online sync path additionally packs it into the int32 MN-major TMA-aligned layout DeepGEMM consumes at runtime.

What sglang uses for the FP8 GEMM, and what slim's online weight sync pushes, depends on the GPU's compute capability:

| SM | sgl default GEMM for scale_fmt=fp32 | slim weight sync | sgl default GEMM for scale_fmt=ue8m0 | slim weight sync |
|------|------|------|------|------|
| SM89  | Triton   | fp32 `(n/128, k/128)` | n/a      | — |
| SM90  | DeepGEMM | fp32 `(n/128, k/128)` | n/a      | — |
| SM120 | DeepGEMM | fp32 `(n/128, k/128)` | DeepGEMM | int32 packed `(n, k/128/4)` |

ue8m0 scales only matter where the engine consumes DeepGEMM's packed UE8M0 layout at runtime (Blackwell); on Hopper/Ada the engine keeps fp32 power-of-two scales and slim leaves the scale unpacked. slim detects this automatically.

**Getting an FP8 checkpoint**

To convert a full-precision model yourself, mirror a reference FP8 model's recipe:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir  $BF16_MODEL \
    --save-dir   $FP8_MODEL \
    --ref-config tools/fp8_recipes/qwen35_official.json \
    --max-workers 4
```

- `--ref-config` is a reference FP8 model dir (or its `config.json`) whose `quantization_config` defines the block size, `scale_fmt`, and the exact `modules_to_not_convert` keep-list, so the online weight sync and the engine agree on which modules stay unquantized.
- `--block-size` / `--scale-fmt {ue8m0}` override the corresponding reference-config values; ue8m0 requires `128×128` blocks.
- Ready-made recipes live in `tools/fp8_recipes/` — e.g. `qwen35_official.json` (fp32 scales) and `qwen35_ue8m0.json` (power-of-two scales for Blackwell).
