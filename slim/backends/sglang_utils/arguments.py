import argparse
import contextlib

from sglang.srt.server_args import ServerArgs
from slim.utils.http_utils import _wrap_ipv6

# ServerArgs fields that managed by slim
# NOT be overwritten by `--sglang-*` CLI flags
_SLIM_MANAGED_SERVER_ARGS = frozenset(
    {
        "model_path",
        "config",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
    }
)


def _resolve_dest(name_or_flags, kwargs):
    """Derive the argparse ``dest`` an add_argument() call would produce.

    Explicit ``dest=`` wins, 
    Otherwise derive from the first ``--long-flag``.
    Returns ``None`` when neither is available.
    """
    if "dest" in kwargs:
        return kwargs["dest"]
    for flag in name_or_flags:
        if isinstance(flag, str) and flag.startswith("--"):
            return flag[2:].replace("-", "_")
    return None


@contextlib.contextmanager
def _sglang_prefixed_add_argument(parser):
    """Within the ``with`` block, arguments registered on ``parser`` are:

    - flag overwrite: ``--log-level`` -> ``--sglang-log-level``
    - dest overwrite: ``args.log_level`` -> ``args.sglang_log_level``
    - skipped if in ``_SLIM_MANAGED_SERVER_ARGS``
    """
    add_argument = parser.add_argument

    def add_prefixed_argument(*name_or_flags, **kwargs):
        dest = _resolve_dest(name_or_flags, kwargs)
        if dest in _SLIM_MANAGED_SERVER_ARGS:
            return  # slim sets this itself; don't expose a CLI flag for it.

        prefixed_flags = [
            f"--sglang-{flag.lstrip('-')}" if isinstance(flag, str) and flag.startswith("-") else flag
            for flag in name_or_flags
        ]

        kwargs = kwargs.copy()
        # Prefix an explicit dest; otherwise argparse derives it from the
        # already-prefixed flags (-> "sglang_foo_bar"), so nothing to do.
        if isinstance(kwargs.get("dest"), str) and not kwargs["dest"].startswith("sglang_"):
            kwargs["dest"] = f"sglang_{kwargs['dest']}"

        add_argument(*prefixed_flags, **kwargs)

    parser.add_argument = add_prefixed_argument
    try:
        yield
    finally:
        parser.add_argument = add_argument


def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    # Register every ServerArgs flag under the `--sglang-*` namespace.
    with _sglang_prefixed_add_argument(parser):
        ServerArgs.add_cli_args(parser)

    # Default the engine console log level to "warning"
    parser.set_defaults(sglang_log_level="warning")

    # PD disaggregation / multi-group config
    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for disaggregation.",
    )
    parser.add_argument(
        "--sglang-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config for SGLang engine deployment. "
            "Defines server_groups with worker_type (regular/prefill/decode/placeholder), "
            "num_gpus per group, and optional per-group 'overrides' dict of "
            "ServerArgs field names that override the base --sglang-* CLI args. "
            "Placeholder groups reserve GPU slots without creating engines. "
            "Mutually exclusive with --prefill-num-servers."
        ),
    )

    return parser


def validate_args(args):
    args.sglang_dp_size = args.sglang_data_parallel_size
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    args.sglang_ep_size = args.sglang_expert_parallel_size

    # Compute effective TP size considering PP size
    if args.sglang_pp_size > 1:
        assert args.rollout_num_gpus_per_replica % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_replica ({args.rollout_num_gpus_per_replica}) must be divisible by "
            f"sglang_pipeline_parallel_size ({args.sglang_pp_size})"
        )
        args.sglang_tp_size = args.rollout_num_gpus_per_replica // args.sglang_pp_size
    else:
        args.sglang_tp_size = args.rollout_num_gpus_per_replica

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if getattr(args, "router_ip", None):
        args.router_ip = _wrap_ipv6(args.router_ip)

    # Mutual-exclusion checks for PD disaggregation / sglang-config.
    assert not (
        getattr(args, "prefill_num_servers", None) is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and args.rollout_external
    ), "sglang_config cannot be set when rollout_external is set."

    assert not (
        getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."


def sglang_parse_args():
    """
    Parse sglang server arguments independently using a separate ArgumentParser.
    Uses parse_known_args() to only consume sglang-related arguments from sys.argv,
    allowing the remaining arguments to be parsed by the training backend separately.

    Returns:
        argparse.Namespace: Parsed sglang arguments (all attributes prefixed with sglang_).
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_sglang_arguments(parser)

    # Compute default sglang_tensor_parallel_size from CLI args
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-replica", type=int, default=1)
    temp_parser.add_argument("--sglang-pp-size", type=int, default=1)
    temp_parser.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.sglang_pp_size if temp_args.sglang_pp_size != 1 else temp_args.sglang_pipeline_parallel_size
    sglang_tp_size = temp_args.rollout_num_gpus_per_replica // pp_size
    parser.set_defaults(sglang_tensor_parallel_size=sglang_tp_size)

    args, _ = parser.parse_known_args()
    return args
