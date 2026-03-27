# SFT Qwen3-4B-Base


## Environment Preparation

First, install minislime following the [Training Guide](../training/README.md) guide and download the `Qwen3-4B-Base` model:

```bash
hf download Qwen/Qwen3-4B-Base --local-dir /root/Qwen3-4B-Base
```

No checkpoint conversion is needed -- FSDP loads HuggingFace checkpoints directly.

After that, we will process the SFT data. Here, we use the classic [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) as an example. First, we process the data into a format suitable for `minislime` to load. You can use the following script to add a column that conforms to the OpenAI message format and save it to `/root/openhermes2_5.parquet`.

```python
from datasets import load_dataset

ds = load_dataset("teknium/OpenHermes-2.5")["train"]

def convert(sample):
    conversations = sample["conversations"]

    def convert_role(role):
        if role == "human":
            return "user"
        elif role == "gpt":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["from"]),
            "content": turn["value"],
        }
        for turn in conversations
    ]

    return {"messages": messages}

ds = ds.map(convert)
ds.to_parquet("/root/openhermes2_5.parquet")
```

## Execute Training

Execute the training:

```bash
cd /root/minislime
bash script/run-qwen3-4B-base-sft.sh
```

### Parameter Introduction

Compared to a standard RL training script, the main adjustments for SFT are:

1.  Removed `SGLANG_ARGS` and `GRPO_ARGS`. This is because it is not necessary to start SGLang or configure GRPO-related settings during the SFT process.

2.  Renamed `ROLLOUT_ARGS` to `SFT_ARGS` and configured it as follows:

    ```bash
    SFT_ARGS=(
       --rollout-function-path slime.rollout.sft_rollout.generate_rollout
       --prompt-data /root/openhermes2_5.parquet
       --rollout-shuffle
       --num-epoch 3
       --rollout-batch-size 128
       --global-batch-size 128

       --loss-type sft_loss
       --calculate-per-token-loss
       --disable-compute-advantages-and-returns
       --debug-train-only
    )
    ```

    SFT actually reuses the custom rollout functionality of minislime. By using `--rollout-function-path`, the data generation part is switched from the RL rollout that uses `sglang` to the SFT version that reads data from a file, which is `slime.rollout.sft_rollout.generate_rollout`.

    For SFT, it is recommended to set `rollout_batch_size` and `global_batch_size` to the same value and not to configure `n_samples_per_prompt`. This is equivalent to training one batch right after reading one batch.

    `minislime` also supports different loss types, and we configure the SFT loss using `--loss-type sft_loss`.

    As for `--calculate-per-token-loss`, this is because `minislime` defaults to calculating the per-sample mean for GRPO. In general SFT training, the average is taken over all unmasked tokens in a batch, so it is recommended to configure this.

    Finally, `--disable-compute-advantages-and-returns` indicates that there is no need to pre-calculate log probabilities during the SFT process, and `--debug-train-only` means that `sglang` does not need to be initialized.

3.  Used `train_async.py` instead of `train.py`. This is to leverage the asynchronous training process to implement data prefetching.
