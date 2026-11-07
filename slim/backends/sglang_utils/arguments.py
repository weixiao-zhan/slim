# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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

    return parser


def validate_args(args):
    if args.sglang_pp_size > 1:
        assert args.rollout_num_gpus_per_replica % args.sglang_pp_size == 0, (
            f"rollout_num_gpus_per_replica ({args.rollout_num_gpus_per_replica}) must be divisible by "
            f"sglang_pp_size ({args.sglang_pp_size})"
        )

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if args.router_ip:
        args.router_ip = _wrap_ipv6(args.router_ip)
