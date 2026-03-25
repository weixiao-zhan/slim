"""GRPO on math (GSM8K). 8 actor GPUs colocated with rollout, batch size 512."""

import os

import slime.utils.external_utils.command_utils as U

NUM_GPUS = 8
MODEL_DIR = "/home/ubuntu/models/Qwen3-1.7B-Base"
DATASET_DIR = "/home/ubuntu/datasets"


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/gsm8k/train.parquet "
        "--input-key prompt "
        "--label-key label "
        "--rm-type math "
        "--num-rollout 10 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-context-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
        "--rollout-shuffle "
    )

    eval_args = (
        "--eval-interval 5 "
        f"--eval-prompt-data gsm8k_test {DATASET_DIR}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-context-len 4096 "
    )

    fsdp_args = "--update-weight-buffer-size 536870912 "

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--attn-implementation flash_attention_2 "
    )

    misc_args = (
        f"--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--gradient-checkpointing "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"

    execute()
