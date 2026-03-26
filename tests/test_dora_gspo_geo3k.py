"""LoRA (PEFT) + GSPO on Geo3K VLM with Qwen3-VL-2B-Thinking. 8xH100 colocated.

LoRA r=128 on all-linear (language model only), GSPO with dynamic sampling filter,
checkpoints saved every 16 rollouts.

Setup & run::

    # Install
    uv sync && uv pip install peft

    # Download model & prepare dataset
    uv run huggingface-cli download Qwen/Qwen3-VL-2B-Thinking \
        --local-dir /home/ubuntu/models/Qwen3-VL-2B-Thinking
    uv run python tests/prepare_geo3k_processor_ready.py

    # Start Ray and launch training
    uv run ray start --head --num-gpus 8 --disable-usage-stats
    WANDB_API_KEY=<key> SLIME_SCRIPT_EXTERNAL_RAY=1 \
        uv run python tests/test_dora_gspo_geo3k.py

    # Follow logs
    uv run ray job logs --follow $(uv run ray job list 2>&1 | \
        grep -oP "submission_id='[^']*'" | head -1 | grep -oP "'[^']*'" | tr -d "'")

    # Stop
    uv run ray stop --force
"""

import os

import slime.utils.external_utils.command_utils as U

NUM_GPUS = 8
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", "/home/ubuntu/models/Qwen3-VL-2B-Thinking")
DATASET_DIR = os.environ.get("VLM_DATASET_DIR", "/home/ubuntu/datasets/geo3k")
SAVE_DIR = os.environ.get("SAVE_DIR", "/home/ubuntu/checkpoints/lora-gspo-geo3k-qwen3vl2b")


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/train.parquet "
        "--input-key prompt "
        "--label-key label "
        "--rm-type math "
        "--num-rollout 500 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-context-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
        "--rollout-shuffle "
    )

    multimodal_args = '--multimodal-keys \'{"image": "images"}\' '

    eval_args = (
        "--eval-interval 16 "
        "--skip-eval-before-train "
        f"--eval-prompt-data geo3k {DATASET_DIR}/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-context-len 8192 "
    )

    fsdp_args = "--update-weight-buffer-size 536870912 " "--gradient-checkpointing "

    peft_args = (
        "--use-peft "
        '--peft-config \'{"r": 128, "lora_alpha": 256, "target_modules": "all-linear"}\' '
    )

    filter_args = (
        "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--over-sampling-batch-size 48 "
    )

    gspo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 5e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-attention-backend flashinfer "
        "--attn-implementation flash_attention_2 "
    )

    save_args = (
        f"--save {SAVE_DIR} "
        "--save-interval 16 "
    )

    wandb_args = (
        "--use-wandb "
        "--wandb-project minislime "
        "--wandb-group lora-gspo-geo3k-qwen3vl2b "
        f"--wandb-key '{os.environ.get('WANDB_API_KEY', '')}' "
        "--disable-wandb-random-suffix "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
        f"{optimizer_args} "
        f"{gspo_args} "
        f"{filter_args} "
        f"{peft_args} "
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

    # Use external Ray (already started) and skip ray job submit
    # (run train.py directly so it uses the venv's python + packages)
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "1"

    execute()
