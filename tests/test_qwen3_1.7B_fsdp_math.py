import os

import slime.utils.external_utils.command_utils as U

NUM_GPUS = 8
MODEL_NAME = "Qwen3-1.7B"
MODEL_DIR = "/home/ubuntu/models/Qwen3-1.7B"
DATASET_DIR = "/home/ubuntu/datasets"


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 2 "
        f"--eval-prompt-data aime {DATASET_DIR}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 2048 "
        "--eval-top-p 0.7 "
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
        "--sglang-cuda-graph-max-bs 32 "
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

    # Use external Ray (already started with SkyPilot ports)
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"

    execute()
