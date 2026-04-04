"""PPO on Geo3K VLM. 4 actor + 4 critic GPUs colocated with 8 rollout, batch size 512."""

import os

from dotenv import load_dotenv

import slime.utils.external_utils.command_utils as U

load_dotenv()

NUM_GPUS = 8
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", "/home/ubuntu/models/Qwen3-VL-2B-Instruct")
DATASET_DIR = os.environ.get("VLM_DATASET_DIR", "/home/ubuntu/datasets/geo3k")
SAVE_DIR = os.environ.get("SAVE_DIR", "/home/ubuntu/outputs/ppo-geo3k-qwen3vl2b")


def execute():
    orch_args = (
        "--num-rollout 200 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        "--rollout-max-context-len 2048 "
        "--rollout-temperature 1 "
        "--num-steps-per-rollout 1 "
    )
    data_args = (
        f"--prompt-data {DATASET_DIR}/train.parquet "
        "--rm-type math "
        "--rollout-shuffle "
        "--log-passrate "
        # "--skip-eval-before-train "
        "--eval-interval 20 "
        f"--eval-prompt-data geo3k {DATASET_DIR}/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-context-len 2048 "
    )
    rollout_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-attention-backend fa3 "
        "--sglang-mm-enable-dp-encoder "
        "--use-fault-tolerance "
    )
    actor_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--critic-num-nodes 1 "
        "--critic-num-gpus-per-node 4 "
        "--attn-implementation flash_attention_3 "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        "--colocate "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )
    loss_args = (
        "--advantage-estimator ppo "
        "--gamma 1.0 "
        "--lambd 0.95 "
        "--value-clip 0.2 "
        "--kl-coef 0.0 "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--critic-lr 2e-5 "
        "--num-critic-only-steps 20"
        "--lr-warmup-iters 10 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    ckpt_args = (
        f"--hf-checkpoint {MODEL_DIR} "
        f"--save {SAVE_DIR} "
        "--save-interval 20 "
    )
    wandb_args = (
        "--use-wandb "
        "--wandb-project minislime "
        "--wandb-group ppo-geo3k-qwen3vl2b "
        f"--wandb-key '{os.environ.get('WANDB_API_KEY', '')}' "
        "--disable-wandb-random-suffix "
    )

    train_args = (
        f"{orch_args} "
        f"{data_args} "
        f"{rollout_args} "
        f"{actor_args} "
        f"{loss_args} "
        f"{optimizer_args} "
        f"{ckpt_args} "
        f"{wandb_args} "
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
