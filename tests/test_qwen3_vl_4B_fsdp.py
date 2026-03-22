import os
import slime.utils.external_utils.command_utils as U

ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))
NUM_GPUS = 8

MODEL_NAME = "Qwen3.5-4B"
MODEL_DIR = os.environ.get("VLM_MODEL_DIR", "/home/ubuntu/models/Qwen3.5-4B")
DATASET_DIR = os.environ.get("VLM_DATASET_DIR", "/home/ubuntu/datasets/vlm_test")


def prepare():
    pass


def execute():
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} "

    rollout_args = (
        f"--prompt-data {DATASET_DIR}/train.parquet "
        "--input-key problem "
        "--label-key answer "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    # multimodal keys: maps type name -> column name in the dataset
    multimodal_args = '--multimodal-keys \'{"image": "images"}\' '

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        f"--eval-prompt-data geo3k {DATASET_DIR}/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
    )

    fsdp_args = "--gradient-checkpointing " "--update-weight-buffer-size 536870912 "

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
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        # "--sglang-enable-deterministic-inference "
        # "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend flashinfer "
        "--attn-implementation sdpa "
        "--sglang-cuda-graph-max-bs 32 "
        # "--deterministic-mode "
        # "--true-on-policy-mode "
    )

    ci_args = ""

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    extra_env_vars = {
        # "NCCL_ALGO": "allreduce:tree",
        # "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        # "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "SGLANG_DISABLE_CUDNN_CHECK": "1",
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
