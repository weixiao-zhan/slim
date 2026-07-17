"""CLI arguments and capability validation for the Megatron backend."""

from __future__ import annotations

import argparse


_VIRTUAL_PIPELINE_OPTIONS = {
    "--virtual-pipeline-model-parallel-size",
    "--num-layers-per-virtual-pipeline-stage",
    "--num-virtual-stages-per-pipeline-rank",
}
_VIRTUAL_PIPELINE_FIELDS = tuple(option[2:].replace("-", "_") for option in _VIRTUAL_PIPELINE_OPTIONS)
_SHORT_PARALLELISM_OPTIONS = {
    "--tp-size": "--tensor-model-parallel-size",
    "--pp-size": "--pipeline-model-parallel-size",
    "--cp-size": "--context-parallel-size",
    "--sp": "--sequence-parallel",
    "--ep-size": "--expert-model-parallel-size",
}


def _option_name(value: str) -> str:
    return value.partition("=")[0]


def _reject_unsupported_cli(unknown: list[str]) -> None:
    for value in unknown:
        option = _option_name(value)
        if option in _VIRTUAL_PIPELINE_OPTIONS:
            raise ValueError(
                f"{option} is not supported; the slim Megatron backend does not "
                "support virtual pipeline parallelism."
            )
        if option in _SHORT_PARALLELISM_OPTIONS:
            raise ValueError(f"{option} is not supported; use {_SHORT_PARALLELISM_OPTIONS[option]}.")


def _parse_megatron_cli(extra_args_provider=None, ignore_unknown_args: bool = False):
    parser = argparse.ArgumentParser("Megatron Training (slim)", allow_abbrev=False)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true", default=False)
    parser.add_argument("--expert-model-parallel-size", type=int, default=1)
    parser.add_argument(
        "--use-distributed-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--optimizer", choices=("adam",), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument(
        "--lr-decay-style",
        choices=("constant", "linear", "cosine", "inverse-square-root", "WSD"),
        default="constant",
    )
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--lr-warmup-iters", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    if ignore_unknown_args:
        args, unknown = parser.parse_known_args()
        _reject_unsupported_cli(unknown)
        return args
    return parser.parse_args()


def _positive_int(args, field: str) -> int:
    value = getattr(args, field)
    option = f"--{field.replace('_', '-')}"
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{option} must be a positive integer, got {value!r}.")
    return value


def _validate_optimizer_args(args) -> None:
    for field in ("weight_decay", "lr_min"):
        value = getattr(args, field)
        if value < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be nonnegative, got {value}.")
    for field in ("adam_beta1", "adam_beta2"):
        value = getattr(args, field)
        if not 0 <= value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be in [0, 1), got {value}.")
    if args.adam_eps <= 0:
        raise ValueError(f"--adam-eps must be positive, got {args.adam_eps}.")
    if args.lr_warmup_iters < 0:
        raise ValueError(
            f"--lr-warmup-iters must be nonnegative, got {args.lr_warmup_iters}."
        )
    if args.lr_decay_iters is not None and args.lr_decay_iters <= 0:
        raise ValueError(
            f"--lr-decay-iters must be positive when set, got {args.lr_decay_iters}."
        )
    if hasattr(args, "clip_grad") and args.clip_grad <= 0:
        raise ValueError(f"--clip-grad must be positive, got {args.clip_grad}.")
    if hasattr(args, "lr_actor") and args.lr_actor is not None and args.lr_actor <= 0:
        raise ValueError(f"--lr-actor must be positive, got {args.lr_actor}.")


def validate_args(args) -> None:
    """Reject unsupported features before Ray allocates trainer workers."""

    for field in _VIRTUAL_PIPELINE_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            option = f"--{field.replace('_', '-')}"
            raise ValueError(
                f"{option} is not supported; the slim Megatron backend does not "
                "support virtual pipeline parallelism."
            )

    use_critic = getattr(args, "use_critic", False) or getattr(args, "advantage_estimator", None) == "ppo_gae"
    if use_critic or getattr(args, "critic_train_only", False):
        raise ValueError("The slim Megatron backend currently supports only the actor role.")

    unsupported_flags = {
        "use_peft": "--use-peft",
        "async_save": "--async-save",
        "no_save_optim": "--no-save-optim",
        "only_train_params_name_list": "--only-train-params-name-list",
        "freeze_params_name_list": "--freeze-params-name-list",
        "rollout_colocate": "--rollout-colocate",
    }
    for field, option in unsupported_flags.items():
        if getattr(args, field, None):
            raise ValueError(f"{option} is not supported by the slim Megatron backend.")

    if getattr(args, "loss_type", "policy_loss") != "policy_loss":
        raise ValueError("--loss-type custom_loss is not supported by the slim Megatron backend.")
    if getattr(args, "advantage_estimator", "grpo") == "gspo":
        raise ValueError("--advantage-estimator gspo is not supported by the slim Megatron backend.")
    if getattr(args, "kl_loss_coef", 0.0) != 0.0:
        raise ValueError("--kl-loss-coef is not supported by the slim Megatron backend.")
    if getattr(args, "ref_update_interval", None) is not None:
        raise ValueError("--ref-update-interval is not supported by the slim Megatron backend.")
    if getattr(args, "mismatch_correction", "none") != "none":
        raise ValueError("--mismatch-correction is not supported by the slim Megatron backend.")
    if getattr(args, "get_mismatch_metrics", False):
        raise ValueError("--get-mismatch-metrics is not supported by the slim Megatron backend.")
    _validate_optimizer_args(args)

    tensor_parallel_size = _positive_int(args, "tensor_model_parallel_size")
    pipeline_parallel_size = _positive_int(args, "pipeline_model_parallel_size")
    context_parallel_size = _positive_int(args, "context_parallel_size")
    _positive_int(args, "expert_model_parallel_size")

    if getattr(args, "sequence_parallel", False) and tensor_parallel_size == 1:
        raise ValueError("--sequence-parallel requires --tensor-model-parallel-size greater than 1.")

    world_size = getattr(args, "actor_num_gpus", None)
    if world_size is None:
        world_size = getattr(args, "world_size", None)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Megatron actor world size must be a positive integer, got {world_size!r}.")

    model_parallel_size = tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            "Megatron actor world size must be divisible by tensor_model_parallel_size * "
            "pipeline_model_parallel_size * context_parallel_size; "
            f"got {world_size} and {model_parallel_size}."
        )
    expert_tensor_parallel_size = getattr(args, "expert_tensor_parallel_size", None)
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = (
            1 if args.expert_model_parallel_size > 1 else tensor_parallel_size
        )
    if (
        isinstance(expert_tensor_parallel_size, bool)
        or not isinstance(expert_tensor_parallel_size, int)
        or expert_tensor_parallel_size <= 0
    ):
        raise ValueError(
            "--expert-tensor-parallel-size must be a positive integer when set, "
            f"got {expert_tensor_parallel_size!r}."
        )
    expert_parallel_size = (
        expert_tensor_parallel_size
        * args.expert_model_parallel_size
        * pipeline_parallel_size
    )
    if world_size % expert_parallel_size != 0:
        raise ValueError(
            "Megatron actor world size must be divisible by expert_tensor_parallel_size * "
            "expert_model_parallel_size * pipeline_model_parallel_size; "
            f"got {world_size} and {expert_parallel_size}."
        )

    replica_size = getattr(args, "actor_num_gpus_per_replica", None)
    if replica_size is not None and replica_size != world_size:
        raise ValueError(
            "--actor-num-gpus-per-replica is an FSDP sharding option and must equal "
            "--actor-num-gpus with the Megatron backend."
        )

    args.rank = 0
    args.world_size = world_size


def megatron_parse_args(extra_args_provider=None, ignore_unknown_args: bool = False):
    args = _parse_megatron_cli(
        extra_args_provider=extra_args_provider,
        ignore_unknown_args=ignore_unknown_args,
    )
    validate_args(args)
    return args


__all__ = ["megatron_parse_args", "validate_args"]
