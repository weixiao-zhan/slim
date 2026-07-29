# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel checkpoint orchestration."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

logger = logging.getLogger(__name__)
_ITERATION_DIR_PATTERN = re.compile(r"iter_(\d+)$")


class RNGState:
    """Rank-local PyTorch RNG state."""

    def state_dict(self) -> dict[str, torch.Tensor]:
        state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state()
        return state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        torch.set_rng_state(state_dict["torch"])
        if torch.cuda.is_available() and "cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["cuda"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Failed to parse checkpoint metadata at %s", path)
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def is_hf_checkpoint(path: str | Path) -> bool:
    """Return whether a path is a Hugging Face model checkpoint."""
    checkpoint = Path(path).expanduser()
    return (
        checkpoint.is_dir()
        and (checkpoint / "config.json").exists()
        and not (checkpoint / "latest_checkpointed_iteration.txt").exists()
    )


def resolve_checkpoint_dir(load_root: str | None, args: Any) -> Path | None:
    """Resolve a Slim iteration directory from a checkpoint root."""
    if load_root is None:
        return None
    root = Path(load_root).expanduser()
    if not root.exists() or is_hf_checkpoint(root):
        return None
    if _ITERATION_DIR_PATTERN.fullmatch(root.name):
        return root

    target_step = args.ckpt_step
    if target_step is None:
        tracker = root / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            return None
        target_step = int(tracker.read_text().strip())
    checkpoint_dir = root / f"iter_{target_step:07d}"
    return checkpoint_dir if checkpoint_dir.exists() else None


def _checkpoint_root(trainer: Any) -> str:
    save_dir = getattr(trainer, "_checkpoint_save_dir", None)
    load_dir = getattr(trainer, "_checkpoint_load_dir", None)
    return str(Path(save_dir or load_dir or trainer.args.hf_checkpoint).expanduser())


def build_checkpointer(trainer: Any):
    """Build the pinned AutoModel checkpointer for a trainer."""
    is_actor = trainer.role == "actor"
    config = CheckpointingConfig(
        checkpoint_dir=_checkpoint_root(trainer),
        model_save_format="safetensors" if is_actor else "torch_save",
        model_repo_id=trainer.args.hf_checkpoint,
        save_consolidated=trainer.args.checkpoint_save_consolidated if is_actor else "false",
        is_async=trainer.args.async_save,
        wait_for_staging=trainer.args.async_save,
        cpu_offload=trainer.args.checkpoint_cpu_offload,
    )
    process_group = getattr(trainer.mesh_context, "process_group", None)
    return config.build(
        dp_rank=dist.get_rank(),
        tp_rank=0,
        pp_rank=0,
        moe_mesh=trainer.moe_mesh,
        process_group=process_group,
    )


def _has_rank_state(checkpoint_dir: Path, state_name: str) -> bool:
    path = checkpoint_dir / state_name / f"{state_name}_dp_rank_{dist.get_rank()}.pt"
    return path.exists()


def load(trainer: Any) -> dict[str, Any] | None:
    """Restore model and optional optimizer state from a Slim checkpoint."""
    load_root = getattr(trainer, "_checkpoint_load_dir", None)
    checkpoint_dir = resolve_checkpoint_dir(load_root, trainer.args)
    if checkpoint_dir is None:
        logger.info("No NeMo checkpoint found at %s", load_root)
        return None

    model_dir = checkpoint_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_dir}")

    logger.info("Loading NeMo model checkpoint from %s", model_dir)
    trainer.checkpointer.load_model(trainer.checkpoint_model, str(model_dir))

    optim_dir = checkpoint_dir / "optim"
    if not trainer.args.no_load_optim and optim_dir.exists():
        logger.info("Loading NeMo optimizer checkpoint from %s", optim_dir)
        trainer.checkpointer.load_optimizer(
            trainer.optimizer,
            trainer.checkpoint_model,
            str(checkpoint_dir),
            None if trainer.args.no_load_lr_scheduler else trainer.lr_scheduler,
        )
    elif not trainer.args.no_load_optim:
        logger.warning("Optimizer checkpoint not found at %s; optimizer state is fresh", optim_dir)
    if trainer.args.no_load_lr_scheduler:
        trainer.lr_scheduler.reset()
        logger.info("LR scheduler reset to step 0")

    match = _ITERATION_DIR_PATTERN.fullmatch(checkpoint_dir.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint_dir.name}")
    return {
        "checkpoint_dir": checkpoint_dir,
        "metadata": _read_checkpoint_metadata(checkpoint_dir / "meta.json"),
        "iteration": int(match.group(1)),
    }


def finalize_load(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Restore auxiliary state after role-specific models are initialized."""
    if payload is None:
        dist.barrier()
        return

    checkpoint_dir = payload["checkpoint_dir"]
    if (
        trainer.role == "actor"
        and trainer.ref_model is not None
        and (checkpoint_dir / "reference").exists()
    ):
        trainer.checkpointer.load_distributed_state(
            ModelState(trainer.ref_model),
            "reference",
            str(checkpoint_dir),
        )

    if not trainer.args.no_load_rng and _has_rank_state(checkpoint_dir, "rng"):
        trainer.checkpointer.load_on_dp_ranks(RNGState(), "rng", str(checkpoint_dir))

    metadata = payload["metadata"]
    iteration = payload["iteration"]
    if metadata:
        trainer.global_step = int(metadata.get("global_step", trainer.global_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            trainer.args.start_rollout_id = int(next_rollout)
    elif trainer.args.start_rollout_id is None:
        trainer.args.start_rollout_id = iteration

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()


def _complete_pending_save(trainer: Any) -> None:
    pending = getattr(trainer, "_pending_checkpoint", None)
    if pending is None:
        return

    trainer.checkpointer.async_wait()
    dist.barrier()
    if dist.get_rank() == 0:
        base_dir, checkpoint_dir, step_id = pending
        tracker = base_dir / "latest_checkpointed_iteration.txt"
        tracker.write_text(str(step_id))
        logger.info("Saved NeMo checkpoint to %s", checkpoint_dir)
    dist.barrier()
    trainer._pending_checkpoint = None


def _initialize_missing_adamw_state(optimizer: torch.optim.Optimizer) -> None:
    """Give every optimized parameter a checkpointable AdamW state."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError(f"NeMo checkpointing expects AdamW, got {type(optimizer).__name__}")
    with torch.no_grad():
        for group in optimizer.param_groups:
            step_on_parameter_device = group.get("capturable", False) or group.get("fused", False)
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if "step" not in state:
                    state["step"] = torch.zeros(
                        (),
                        dtype=torch.float32,
                        device=parameter.device if step_on_parameter_device else "cpu",
                    )
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                if group.get("amsgrad", False) and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)


def _save_consolidated_processor(trainer: Any, checkpoint_dir: Path, *, is_final: bool) -> None:
    if trainer.role != "actor" or trainer.processor is None:
        return
    mode = trainer.args.checkpoint_save_consolidated
    if mode == "false" or (mode == "final" and not is_final):
        return
    if dist.get_rank() == 0:
        trainer.processor.save_pretrained(checkpoint_dir / "model" / "consolidated")
    dist.barrier()


def save(trainer: Any, rollout_id: int, *, force_sync: bool = False) -> None:
    """Save one Slim iteration with NeMo checkpoint components."""
    _complete_pending_save(trainer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    base_dir = Path(trainer._checkpoint_save_dir).expanduser()
    step_id = rollout_id + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    if dist.get_rank() == 0:
        if checkpoint_dir.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True)
    dist.barrier()

    trainer.checkpointer.save_model(
        trainer.checkpoint_model,
        str(checkpoint_dir),
        tokenizer=trainer.tokenizer if trainer.role == "actor" else None,
        is_final_checkpoint=force_sync,
    )
    _save_consolidated_processor(trainer, checkpoint_dir, is_final=force_sync)
    if not trainer.args.no_save_optim:
        _initialize_missing_adamw_state(trainer.optimizer)
        trainer.checkpointer.save_optimizer(
            trainer.optimizer,
            trainer.checkpoint_model,
            str(checkpoint_dir),
            trainer.lr_scheduler,
        )

    if trainer.role == "actor" and trainer.ref_model is not None:
        trainer.checkpointer.save_distributed_state(
            ModelState(trainer.ref_model),
            "reference",
            str(checkpoint_dir),
        )
    trainer.checkpointer.save_on_dp_ranks(RNGState(), "rng", str(checkpoint_dir))

    if dist.get_rank() == 0:
        _write_checkpoint_metadata(
            checkpoint_dir / "meta.json",
            {
                "iteration": step_id,
                "rollout_id": rollout_id,
                "next_rollout_id": rollout_id + 1,
                "global_step": trainer.global_step,
                "world_size": dist.get_world_size(),
                "timestamp": time.time(),
            },
        )
    dist.barrier()
    trainer._pending_checkpoint = (base_dir, checkpoint_dir, step_id)
    if force_sync or not trainer.checkpointer.config.is_async:
        _complete_pending_save(trainer)
