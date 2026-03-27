"""GRPO on Geo3K VLM. 8 actor GPUs with 8 rollout engines, batch size 512."""

import os

from dotenv import load_dotenv

import slime.utils.external_utils.command_utils as U

load_dotenv()

NUM_GPUS = 8
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", "/home/ubuntu/models/Qwen3-VL-2B-Instruct")
DATASET_DIR = os.environ.get("VLM_DATASET_DIR", "/home/ubuntu/datasets/geo3k")


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/train.parquet "
        "--rm-type math "
        "--num-rollout 200 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-context-len 8192 "
        "--rollout-temperature 1 "
        "--num-steps-per-rollout 1 "
        "--rollout-shuffle "
    )

    eval_args = (
        "--eval-interval 20 "
        "--skip-eval-before-train "
        f"--eval-prompt-data geo3k {DATASET_DIR}/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-context-len 8192 "
    )

    fsdp_args = (
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        "--attn-implementation flash_attention_2 "
    )

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
        "--lr 1e-5 "
        "--lr-warmup-iters 10 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-attention-backend flashinfer "
        "--sglang-mm-enable-dp-encoder "
    )

    wandb_args = (
        "--use-wandb "
        "--wandb-project minislime "
        "--wandb-group grpo-geo3k-qwen3vl2b "
        f"--wandb-key '{os.environ.get('WANDB_API_KEY', '')}' "
        "--disable-wandb-random-suffix "
    )

    load_args = "--load /home/ubuntu/outputs/grpo-geo3k-qwen3vl2b "

    save_args = (
        "--save /home/ubuntu/outputs/grpo-geo3k-qwen3vl2b "
        "--save-interval 20 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--log-pass-ratio "
        "--use-fault-tolerance "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{fsdp_args} "
        f"{eval_args} "
        f"{load_args} "
        f"{save_args} "
        f"{sglang_args} "
        f"{wandb_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        extra_env_vars={"SGLANG_DISABLE_CUDNN_CHECK": "1"},
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"
    os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

    execute()
