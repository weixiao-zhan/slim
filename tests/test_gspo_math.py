"""GSPO on math (GSM8K). 8 actor GPUs colocated with rollout, batch size 512."""

import os

from dotenv import load_dotenv

import slime.utils.external_utils.command_utils as U

load_dotenv()

NUM_GPUS = 8
MODEL_DIR = os.environ.get("LLM_MODEL_DIR", "/home/ubuntu/models/Qwen3-1.7B-Base")
DATASET_DIR = os.environ.get("LLM_DATASET_DIR", "/home/ubuntu/datasets")
SAVE_DIR = os.environ.get("SAVE_DIR", "/home/ubuntu/outputs/gspo-gsm8k-qwen3-1.7b")


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/gsm8k/train.parquet "
        "--rm-type math "
        "--num-rollout 200 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-context-len 2048 "
        "--rollout-temperature 1 "
        "--num-steps-per-rollout 1 "
        "--rollout-shuffle "
        "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--over-sampling-batch-size 48 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--skip-eval-before-train "
        f"--eval-prompt-data gsm8k_test {DATASET_DIR}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-context-len 2048 "
    )

    fsdp_args = (
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        "--attn-implementation flash_attention_2 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    loss_args = (
        "--advantage-estimator gspo "
        "--disable-grpo-std-normalization "
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
        "--sglang-attention-backend fa3 "
        "--sglang-mm-enable-dp-encoder "
        "--use-fault-tolerance "
    )

    save_args = (
        f"--save {SAVE_DIR} "
        "--save-interval 20 "
    )

    wandb_args = (
        "--use-wandb "
        "--wandb-project minislime "
        "--wandb-group gspo-gsm8k-qwen3-1.7b "
        f"--wandb-key '{os.environ.get('WANDB_API_KEY', '')}' "
        "--disable-wandb-random-suffix "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--log-passrate "
        ""
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{loss_args} "
        f"{fsdp_args} "
        f"{save_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{wandb_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        train_script=str(U.repo_base_dir / "train.py"),
    )


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"

    execute()
