import argparse
import dataclasses
import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"  # Optimizer type: "adam" (AdamW)
    lr: float = 2e-5
    lr_warmup_init: float = 0.0
    min_lr: float = 0.0
    lr_decay_style: str = "constant"
    lr_decay_iters: int | None = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: float | None = None
    lr_wsd_decay_iters: int | None = None
    lr_wsd_decay_style: str | None = None
    no_load_lr_scheduler: bool = False
    lr_scheduler_start_step: int | None = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    attn_implementation: str = "sdpa"

    # Logging
    wandb_project: str = "slim-fsdp"
    wandb_run_name: str | None = None

    # Precision
    gradient_checkpointing: bool = False
    # Master-weight (storage) dtype passed to from_pretrained. 
    # None preserves the checkpoint's per-tensor dtypes. 
    # "fp32" force-promotes everything to fp32 (Mixed precision standard practice)
    master_weight_dtype: str | None = None  # None | "fp32"
    # FSDP2 compute dtype (MixedPrecisionPolicy.param_dtype) for forward/backward.
    # None lets compute follow storage dtype; 
    # "bf16"/"fp16" force a uniform compute dtype. 
    compute_dtype: str | None = None  # None | "bf16" | "fp16"

    # FSDP configuration
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.
    fsdp_cpu_offload: bool = (
        False  # If True, offload parameters, gradients, and optimizer states to CPU (optimizer runs on CPU)
    )
    fsdp_cpu_backend: str | None = (
        "gloo"  # CPU backend for FSDP CPU offload (e.g., "gloo"). Set to None to disable hybrid backend.
    )

    deterministic_mode: bool = False

    # YAML bookkeeping
    config: str | None = None


def _parse_fsdp_cli(extra_args_provider=None, ignore_unknown_args=False):
    parser = argparse.ArgumentParser("FSDP Training (slim)", allow_abbrev=False)
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue

        # Handle union types like int | None, str | None, etc.
        if hasattr(f.type, "__args__"):  # Check if it's a Union type
            # For T | None, use T as the type
            non_none_types = [t for t in f.type.__args__ if t is not type(None)]
            arg_type = non_none_types[0] if non_none_types else str
        else:
            arg_type = f.type

        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def fsdp_parse_args(extra_args_provider=None, ignore_unknown_args=False):
    args = _parse_fsdp_cli(extra_args_provider, ignore_unknown_args=ignore_unknown_args)
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    args.rank = 0  # Primary process rank for wandb initialization
    args.world_size = args.actor_num_gpus

    valid_master = (None, "fp32")
    if args.master_weight_dtype not in valid_master:
        raise ValueError(
            f"--master-weight-dtype must be one of {valid_master}, got {args.master_weight_dtype!r}"
        )
    valid_compute = (None, "bf16", "fp16")
    if args.compute_dtype not in valid_compute:
        raise ValueError(
            f"--compute-dtype must be one of {valid_compute}, got {args.compute_dtype!r}"
        )
    if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "use_peft", False):
        logger.warning(
            "--fsdp-cpu-offload and --use-peft would cause "
            "extremely slow weight sync due to merge/unmerge "
            "adaptor in CPU DTensors."
        )

    return args
