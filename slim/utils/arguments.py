# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs

from slim.backends.sglang_utils.arguments import sglang_parse_args
from slim.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slim.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slim.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of an argument or add it if missing.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slim_extra_args_provider(add_custom_arguments=None):
    def add_slim_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument(
                "--actor-num-gpus",
                type=int,
                default=8,
                help="Total number of GPUs for the training actor.",
            )
            parser.add_argument(
                "--critic-num-gpus",
                type=int,
                default=None,
                help="Total number of GPUs for the critic. Must equal actor_num_gpus. Defaults to actor_num_gpus.",
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --rollout-colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set "
                    "to the number of training GPUs (actor, plus critic when disaggregated)."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-replica",
                type=int,
                default=1,
                help="Number of GPUs per inference engine (a rollout replica), just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Physical number of GPUs per node, used for packing / NUMA / port layout. "
                    "Notice: If you are going to use less than 8 gpus per node, you should set this number."
                ),
            )
            parser.add_argument(
                "--rollout-colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines on the training GPUs. "
                ),
            )
            parser.add_argument(
                "--critic-colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the critic on the actor GPUs (time-shared) instead of giving it "
                    "its own GPUs. Requires critic_num_gpus == actor_num_gpus."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of the training backend, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slim.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `loss_mask`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-sampling-params",
                type=json.loads,
                default={},
                help=(
                    "Extra SGLang sampling params as a JSON dict, e.g. "
                    "'{\"top_p\":0.95,\"top_k\":50,\"stop\":[\"<|im_end|>\"]}'. "
                    "Merged on top of {temperature, no_stop_trim=True, "
                    "spaces_between_special_tokens=False}. Use this for top_p, "
                    "top_k, stop, stop_token_ids, skip_special_tokens, "
                    "min_new_tokens, repetition_penalty, ignore_eos, etc."
                ),
            )
            parser.add_argument(
                "--max-context-len",
                type=int,
                default=None,
                help=(
                    "Single source of truth for max context length. Drives train rollout budget, "
                    "eval rollout budget (per-dataset YAML override allowed), and SGLang server "
                    "context_length. Must not exceed `max_position_embeddings` in the HF model config."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--rollout-concurrency-per-replica",
                type=int,
                default=128,
                help=(
                    "The total in-flight generate concurrency is sized to "
                    "rollout_concurrency_per_replica * num_rollout_engines. "
                    "The router decides how to distribute these requests across engines."
                ),
            )
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int, 
                default=None,
                help=(
                    "Target size of the in-flight rollout group pool. The rollout loop tops up to this "
                    "number on every iteration. Set larger than rollout_batch_size to oversample, or smaller "
                    "to cap rollout concurrency. If None and no rollout-group-filter-path: dispatch exactly "
                    "rollout_batch_size groups once and drain them all (no re-top-up). "
                ),
            )
            parser.add_argument(
                "--rollout-group-filter-path",
                type=str,
                default=None,
                help=(
                    "Group-level dynamic sampling filter: decides whether to KEEP or DROP an entire "
                    "sample group during rollout, e.g. drop all-correct / all-wrong groups with zero std"
                    "You could use `slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(state, episode)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--rollout-disable-fault-tolerance",
                action="store_false",
                dest="rollout_fault_tolerance",
                help="Disable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slim.rollout.data_source.RolloutDataSource",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "By convention each row should contain `prompt` and `label`, "
                    "and may also contain `images`, `videos`, `audio`, `tools`, and `metadata`. "
                    "The input can be a plain string or a list of chat messages "
                    "(e.g. [{'role': 'user', 'content': 'blabla'}]). "
                    "Chat-format prompts are automatically processed via apply_chat_template. "
                ),
            )
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slim will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note: this value should typically be close to `max_response_len`."
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            parser.add_argument(
                "--eval-n-samples-per-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            reset_arg(
                parser,
                "--load",
                type=str,
                default=None,
                help=(
                    "Path to load training weights from. Accepts either: "
                    "(1) a slim DCP checkpoint directory (with latest_checkpointed_iteration.txt) for resuming training, or "
                    "(2) a HuggingFace checkpoint directory (with config.json) for BF16 weight initialization. "
                    "When an HF checkpoint is given, it overrides --hf-checkpoint for training init only, "
                    "allowing --hf-checkpoint to point to a quantized model (e.g. FP8) for the rollout engine."
                ),
            )
            parser.add_argument(
                "--ckpt-step",
                type=int,
                default=None,
                help=(
                    "The checkpoint step for actor/critic training resumption. "
                    "When unset, the loader uses `latest_checkpointed_iteration.txt` under --load."
                ),
            )
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument(
                "--lr-actor",
                type=float,
                default=None,
                help="Max LR for the actor (RL policy). Defaults to --lr.",
            )
            parser.add_argument(
                "--lr-actor-start-step",
                type=int,
                default=0,
                help="Rollout step at which the actor (RL policy) starts training.",
            )
            parser.add_argument(
                "--lr-critic",
                type=float,
                default=None,
                help="Max LR for the critic backbone. Defaults to --lr.",
            )
            parser.add_argument(
                "--lr-critic-start-step",
                type=int,
                default=0,
                help="Rollout step at which the critic backbone learning rate leaves 0.",
            )
            parser.add_argument(
                "--lr-critic-value-head",
                type=float,
                default=None,
                help="Max LR for the critic value head. Defaults to --lr-critic.",
            )
            parser.add_argument(
                "--lr-critic-value-head-start-step",
                type=int,
                default=0,
                help="Rollout step at which the critic value head learning rate leaves 0.",
            )
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-train-only", action="store_true", default=False, help="Only train critic")

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type. `policy_loss` is the built-in policy-gradient loss. "
                    "If `custom_loss` is set, the function from `--custom-loss-function-path` is used."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help="Path to the custom loss function, used when `--loss-type custom_loss`.",
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="low_var_kl",
                help="Choose KL loss type: k1, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--policy-surrogate",
                type=str,
                choices=["ppo_clip", "is", "tis", "cis"],
                default="ppo_clip",
                help="Policy-gradient surrogate objective.",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "ppo_gae",
                ],
                default="grpo",
                help="Advantage estimator to use.",
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final loss. ",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-group-advantage-normalization",
                action="store_false",
                dest="group_advantage_normalization",
                help="Use raw rewards directly as GRPO or GSPO advantages.",
            )
            parser.add_argument(
                "--disable-group-advantage-std-normalization",
                action="store_false",
                dest="group_advantage_std_normalization",
                help="Disable standard-deviation scaling after GRPO or GSPO group mean centering.",
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help=(
                    "Force an old actor forward pass for train/mismatch/*"
                ),
            )
            parser.add_argument(
                "--old-logprob-source",
                type=str,
                choices=["actor", "rollout"],
                default="actor",
                help="Baseline policy for policy-surrogate ratios.",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help=(
                    "Replay rollout-time MoE expert routing during training. Captures top-k expert "
                    "indices from sglang via enable_return_routed_experts and forces the actor's "
                    "router to gather scores at those same indices. Eliminates train/inference "
                    "expert-selection mismatch when supported by the selected training model family."
                ),
            )
            parser.add_argument(
                "--mismatch-correction",
                type=str,
                choices=["none", "custom"],
                default="none",
                help="Optional actor-old / rollout-old importance correction.",
            )
            parser.add_argument(
                "--custom-mismatch-correction-function-path",
                type=str,
                default=None,
                help="Dotted path to a custom mismatch correction function.",
            )
            return parser

        def add_router_arguments(parser):
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            parser.set_defaults(router_balance_abs_threshold=10, router_balance_rel_threshold=1.2)
            # slim-specific router connection args (RouterArgs excludes host/port).
            parser.add_argument(
                "--router-ip",
                type=str,
                default=None,
                help="IP address of the SGLang router",
            )
            parser.add_argument(
                "--router-port",
                type=int,
                default=None,
                help="Port of the SGLang router",
            )
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--eval-log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--eval-save-rollout",
                type=str,
                default=None,
                help=(
                    "Save eval rollout tokens and rewards to this path template. "
                    "Use {rollout_id} as placeholder. Saved as .pt with per-dataset "
                    "flat rewards and tokens, ordered same as dataset * eval_n_samples_per_prompt."
                ),
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # --- Performance profiler (torch.profiler for train_* targets; VizTracer for `rollout`) ---
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["rollout", "train_log_probs", "train_pg", "train_overall"],
                default=[],
                nargs="+",
                help="What to profile (empty = off). "
                "`rollout` uses VizTracer to trace whole step's async generate_rm calls."
                "`train_log_probs`, `train_pg`, and `train_overall` use torch.profiler. "
            )
            parser.add_argument("--profile-step-start", type=int, default=10)
            parser.add_argument("--profile-step-end", type=int, default=12)
            parser.add_argument(
                "--profile-dir",
                type=str,
                default="./profiles",
                help="Output dir for all profiler artifacts (chrome traces, memory snapshots).",
            )
            # --- Memory recorder ---
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default=[],
                nargs="+",
                help="Which memory recorder(s) to run (empty = off); may pass both. torch = GPU/CUDA "
                "allocator history (CUDA OOM); memray = host/CPU RAM allocations (host OOM).",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
                help="Stop recording and dump the snapshot after this many rollouts (required for memray).",
            )
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help="Path to a custom function that shapes episode rewards before advantage estimation.",
            )
            return parser

        def add_rollout_filter_arguments(parser):
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and sample groups as input. "
                    "To mask a sample from training, set its loss_mask to all zeros. "
                    "Note: This does not affect advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="Disable trim samples when converting samples to train data",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="Enable dynamic global batch size, disable trim samples when converting samples to train data",
            )
            return parser


        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slim arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_filter_arguments(parser)
        parser = add_ci_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slim_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slim_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_slim_arguments = get_slim_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse NeMo + slim args.
    from slim.backends.nemo.arguments import nemo_parse_args

    args = nemo_parse_args(extra_args_provider=add_slim_arguments, ignore_unknown_args=True)

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slim_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def slim_validate_args(args):
    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    args.eval_datasets = _resolve_eval_datasets(args)

    if args.kl_loss_coef != 0:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    if args.mismatch_correction != "none" and args.old_logprob_source != "actor":
        raise ValueError("--mismatch-correction requires --old-logprob-source actor.")

    if args.mismatch_correction == "custom" and args.custom_mismatch_correction_function_path is None:
        raise ValueError("--mismatch-correction custom requires --custom-mismatch-correction-function-path.")

    if args.get_mismatch_metrics and args.old_logprob_source == "rollout":
        logger.info(
            "get_mismatch_metrics is set; actor-old log probs will be computed by the training engine."
        )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.use_critic = args.advantage_estimator == "ppo_gae"
    if args.critic_train_only:
        if not args.use_critic:
            raise ValueError("--critic-train-only requires --use-critic (or --advantage-estimator ppo_gae).")
        if args.actor_num_gpus != 0:
            raise ValueError(
                f"--critic-train-only requires --actor-num-gpus 0, but got actor_num_gpus={args.actor_num_gpus}."
            )
    if args.critic_num_gpus is None:
        args.critic_num_gpus = args.actor_num_gpus
    if args.critic_load is None:
        args.critic_load = args.load
    if args.lr_actor is None:
        args.lr_actor = args.lr
    if args.lr_critic is None:
        args.lr_critic = args.lr
    if args.lr_critic_value_head is None:
        args.lr_critic_value_head = args.lr_critic

    # Actor and critic share one rollout-data split and pair rank-wise, so their
    # GPU totals must match (this also satisfies the --critic-colocate C == A rule).
    if args.use_critic and not args.critic_train_only:
        assert args.critic_num_gpus == args.actor_num_gpus, (
            f"critic_num_gpus ({args.critic_num_gpus}) must equal actor_num_gpus ({args.actor_num_gpus})."
        )
    if args.critic_colocate:
        if not args.use_critic:
            raise ValueError("--critic-colocate requires --advantage-estimator ppo_gae.")
        if args.critic_train_only:
            raise ValueError("--critic-colocate is incompatible with --critic-train-only.")
        assert args.critic_num_gpus == args.actor_num_gpus, (
            f"--critic-colocate requires critic_num_gpus == actor_num_gpus, "
            f"got {args.critic_num_gpus} vs {args.actor_num_gpus}."
        )

    if args.debug_rollout_only:
        if args.rollout_colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus
        else:
            args.actor_num_gpus = args.rollout_num_gpus
        args.rollout_colocate = False
        args.critic_colocate = False

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # Rollout colocate time-shares the training GPUs: size R to the training span.
    if args.rollout_colocate:
        if args.critic_train_only:
            train_span = args.critic_num_gpus
        elif args.critic_colocate or not args.use_critic:
            train_span = args.actor_num_gpus
        else:
            train_span = args.actor_num_gpus + args.critic_num_gpus
        if args.rollout_num_gpus != train_span:
            logger.info(
                f"rollout_colocate set: overriding rollout_num_gpus {args.rollout_num_gpus} -> {train_span} "
                "to fill the training span."
            )
            args.rollout_num_gpus = train_span

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.group_advantage_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, group advantage std normalization will be disabled.")

    if args.rollout_group_filter_path and not args.over_sampling_batch_size:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size is None or args.over_sampling_batch_size > 0, (
        f"over_sampling_batch_size must be a positive int or None, got {args.over_sampling_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.loss_type == "custom_loss" and args.custom_loss_function_path is None:
        raise ValueError("--loss-type custom_loss requires --custom-loss-function-path.")
