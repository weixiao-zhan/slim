# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learning rate scheduler for NeMo AutoModel training."""

import logging
import math
from typing import override

import torch
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


_DECAY_COEFFICIENTS = {
    "linear": lambda ratio: 1.0 - ratio,
    "cosine": lambda ratio: 0.5 * (math.cos(math.pi * ratio) + 1.0),
    "exponential": lambda ratio: 2.0 * math.pow(0.5, ratio) - 1.0,
    "minus_sqrt": lambda ratio: 1.0 - math.sqrt(ratio),
}


class NeMoLRScheduler(LRScheduler):
    """Warmup, constant, and decay learning-rate schedule for NeMo training.

    Each parameter group warms up from ``init_lr`` to its ``max_lr`` over ``lr_warmup_steps``
    steps from its ``start_step``. All groups keep ``max_lr`` until ``decay_start``, then decay
    to ``min_lr`` over ``lr_decay_steps`` steps with the ``lr_decay_style`` shape.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        init_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        lr_warmup_steps (int): Number of warmup steps.
        lr_decay_steps (int): Number of decay steps. Zero keeps ``max_lr`` after warmup.
        lr_decay_style (str): Decay shape, one of ``_DECAY_COEFFICIENTS``.
        decay_start (int): Step at which the decay starts. ``start_lr_decay`` can move it earlier.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        decay_start: int,
        last_epoch: int = -1,
    ) -> None:
        # Store our custom parameters
        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_style = lr_decay_style
        self.decay_start = decay_start

        self.warmup_end = 0
        for index, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"param_group_{index}")
            warmup_end = group.get("start_step", 0) + self.lr_warmup_steps
            if warmup_end > self.decay_start:
                raise ValueError(f"{name} warmup ends at step {warmup_end}, after the decay start {self.decay_start}")
            self.warmup_end = max(self.warmup_end, warmup_end)

        super().__init__(optimizer, last_epoch)

        logger.info(
            f"> learning rate: warmup {self.lr_warmup_steps} steps, "
            f"{self.lr_decay_style} decay {self.lr_decay_steps} steps from step {self.decay_start}"
        )

    def _get_lr_for_group(self, param_group: dict) -> float:
        """Compute learning rate for a specific parameter group.

        Args:
            param_group (dict): parameter group from the optimizer.

        Returns:
            float: learning rate for this parameter group.
        """
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)
        start_step = param_group.get("start_step", 0)
        if self.last_epoch < start_step:
            return 0.0

        component_step = self.last_epoch - start_step

        if self.lr_warmup_steps > 0 and component_step <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr)
                * float(component_step)
                / float(self.lr_warmup_steps)
            )

        if self.lr_decay_steps == 0 or self.last_epoch <= self.decay_start:
            return max_lr

        decay_ratio = min(1.0, float(self.last_epoch - self.decay_start) / float(self.lr_decay_steps))
        coeff = _DECAY_COEFFICIENTS[self.lr_decay_style](decay_ratio)
        return min_lr + coeff * (max_lr - min_lr)

    @override
    def get_lr(self) -> list[float]:
        """Compute the learning rates for each parameter group.

        Returns:
            list[float]: A list of learning rates, one for each parameter group.
        """
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]

    def start_lr_decay(self) -> int:
        """Start the decay at the current step and return the remaining decay steps.

        The decay starts after all warmups end. The decay start never moves later, so the
        decay start saved in the scheduler state stays the same after resume.
        """
        decay_start = min(self.decay_start, max(self.last_epoch, self.warmup_end))
        if decay_start != self.decay_start:
            self.decay_start = decay_start
            logger.info(f"> learning rate decay starts at step {self.decay_start}")
        return max(self.decay_start + self.lr_decay_steps - self.last_epoch, 0)

    def reset(self) -> None:
        """Reset the learning-rate timeline to step zero."""
        self.last_epoch = 0
        self._step_count = 1
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            param_group["lr"] = lr


def get_lr_scheduler(args, optimizer: torch.optim.Optimizer) -> NeMoLRScheduler:
    """Create and configure the learning-rate scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args: Training/runtime arguments (namespace).
        optimizer (torch.optim.Optimizer): Optimizer bound to the model.

    Returns:
        NeMoLRScheduler: Initialized scheduler bound to ``optimizer``.
    """
    train_steps = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_warmup_steps + args.lr_decay_steps > train_steps:
        raise ValueError(
            f"lr_warmup_steps {args.lr_warmup_steps} + lr_decay_steps {args.lr_decay_steps} "
            f"exceed the {train_steps} train steps"
        )
    lr_scheduler = NeMoLRScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.lr_min,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        decay_start=train_steps - args.lr_decay_steps,
    )

    return lr_scheduler
