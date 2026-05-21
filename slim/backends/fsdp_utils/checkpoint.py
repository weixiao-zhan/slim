from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)
_ITERATION_DIR_PATTERN = re.compile(r"iter_(\d+)$")


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(self.model, optimizers=[], model_state_dict=state_dict["model"], optim_state_dict=None)


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def is_hf_checkpoint(path: str | Path) -> bool:
    """Return True if *path* looks like a HuggingFace checkpoint directory.

    A HuggingFace checkpoint has ``config.json`` at its root but no
    ``latest_checkpointed_iteration.txt`` (which marks a slim DCP checkpoint).
    """
    p = Path(path).expanduser()
    return p.is_dir() and (p / "config.json").exists() and not (p / "latest_checkpointed_iteration.txt").exists()


def resolve_checkpoint_dir(load_root: str | None, args: Any) -> Path | None:
    """Resolve the latest checkpoint directory from a load root and args.

    Returns the ``iter_NNNNNNN`` directory Path, or None if not found.
    Skips HF checkpoint directories (handled separately by the actor).
    """
    if load_root is None:
        return None
    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        return None
    # HF checkpoints are not DCP — the actor handles them via from_pretrained.
    if is_hf_checkpoint(root_path):
        return None
    direct_match = _ITERATION_DIR_PATTERN.fullmatch(root_path.name)
    if direct_match is not None:
        return root_path
    target_step = getattr(args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            return None
        target_step = int(tracker_file.read_text().strip())
    return root_path / f"iter_{target_step:07d}"


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor, "_checkpoint_load_dir", None) or getattr(actor.args, "load", None)
    checkpoint_dir = resolve_checkpoint_dir(load_root, actor.args)
    if checkpoint_dir is None:
        logger.info(f"[FSDP] Checkpoint not found at {load_root}; skipping load.")
        return None
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    # Load model weights (always — includes PEFT adapter weights if present)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        logger.info(f"[FSDP] Loading model from {model_dir} ...")
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(model_dir),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
        return None

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            logger.info(f"[FSDP] Loading optimizer from {optimizer_dir} ...")
            dcp.load(
                state_dict=optim_state_dict,
                checkpoint_id=str(optimizer_dir),
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            optimizer_state_count = len(getattr(actor.optimizer, "state", {}))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir} ({optimizer_state_count} state entries)")
            if optimizer_state_count == 0:
                logger.warning(
                    "[FSDP] Optimizer checkpoint load completed but optimizer.state is empty. "
                    "Training will continue with fresh optimizer moments."
                )
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    load_lr_scheduler = hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            logger.info(f"[FSDP] Loading LR scheduler from {lr_scheduler_dir} ...")
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(
                f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir} (last_epoch={actor.lr_scheduler.last_epoch})"
            )
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[FSDP] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    # Extract step from directory name (e.g., "iter_0000001" → 1)
    iteration = int(checkpoint_dir.name.split("_")[1])

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": iteration,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    torch.cuda.synchronize()
    dist.barrier()


def save(actor: Any, iteration: int) -> None:
    """Save checkpoint to disk.

    Saves model weights and optimizer state to separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    torch.cuda.synchronize()

    save_dir = getattr(actor, "_checkpoint_save_dir", None) or actor.args.save
    base_dir = Path(save_dir).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if dist.get_rank() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # Note: PEFT adapter weights are included in the DCP model checkpoint above.
    # No separate adapter save needed — checkpoint.load() restores everything.

    # Save optimizer state (skip if --no-save-optim is set)
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # Save LR scheduler state (skip if --no-save-optim is set)
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if dist.get_rank() == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, checkpoint_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(checkpoint_dir / "meta.json", metadata)

        tracker_file = base_dir / "latest_checkpointed_iteration.txt"
        tracker_file.write_text(str(step_id))
        logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()
