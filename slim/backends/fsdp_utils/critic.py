"""FSDP trainer for the PPO value function (critic).

Wraps a pretrained causal LM by replacing its lm_head with a scalar
value head (Linear(hidden_size, 1)), producing per-token value estimates,
and trains it with the clipped value loss.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm

from slim.utils.data import process_rollout_data
from slim.utils.ppo_utils import compute_value_loss, sum_of_sample_mean
from slim.utils.timer import timer

from .base import FSDPTrainer
from .data_packing import (
    init_dummy_advantages,
    strip_cross_boundary,
    unpack_sequences,
)

logger = logging.getLogger(__name__)


class CriticFSDPTrainer(FSDPTrainer):
    """FSDP trainer for the PPO value function (critic)."""

    _train_log_prefix = "train/critic"

    def _resolve_checkpoint_paths(self) -> str:
        # Critic resumes from and saves to its own checkpoints; init weights from
        # --critic-load (falling back to the shared HF checkpoint).
        self._checkpoint_load_dir = self.args.critic_save
        self._checkpoint_save_dir = self.args.critic_save
        return self.args.critic_load or self.args.hf_checkpoint

    def _create_model(self, hf_checkpoint: str, init_context):
        """Build the critic by swapping the backbone's lm_head for a value head.

        Loads the base causal LM and replaces its language-modeling head with a
        scalar value head (``Linear(hidden_size, 1)``) that outputs a single
        value estimate per token position.
        """
        with init_context():
            model = self.get_model_cls().from_pretrained(hf_checkpoint, **self._load_kwargs)

        # Replace the lm_head with a value head.
        # Zero-init for stable PPO startup (V(s) ≈ 0 before critic warmup).
        old_lm_head = model.lm_head
        hidden_size = old_lm_head.in_features
        backbone_dtype = old_lm_head.weight.dtype
        value_head = nn.Linear(hidden_size, 1, bias=False, dtype=backbone_dtype)
        nn.init.zeros_(value_head.weight)
        model.lm_head = value_head

        logger.info(
            f"[Rank {dist.get_rank()}] Critic model created from {hf_checkpoint}; "
            f"replaced lm_head [{old_lm_head.in_features}x{old_lm_head.out_features}] "
            f"with value_head [{hidden_size}x1]"
        )

        return model

    def _build_optimizer_param_groups(self) -> list[dict]:
        """Two groups so the value head and backbone can warm up independently.

        The value head is the swapped-in lm_head (``lm_head.weight``); FSDP2
        (fully_shard) preserves the parameter FQN, so name-suffix selection is
        reliable post-wrap. Each group carries its own ``max_lr`` and
        ``start_step`` (in optimizer steps) consumed by FSDPLRScheduler.
        """
        steps_per_rollout = self._steps_per_rollout()
        value_head_params, backbone_params = [], []
        for name, p in self.model.named_parameters():
            if name.endswith("lm_head.weight"):
                value_head_params.append(p)
            else:
                backbone_params.append(p)
        assert len(value_head_params) == 1, (
            f"expected exactly one value-head param (lm_head.weight), got {len(value_head_params)}"
        )
        return [
            {
                "params": backbone_params,
                "max_lr": self.args.critic_lr,
                "start_step": self.args.lr_critic_start_step * steps_per_rollout,
            },
            {
                "params": value_head_params,
                "max_lr": self.args.critic_value_head_lr,
                "start_step": self.args.lr_critic_value_head_start_step * steps_per_rollout,
            },
        ]

    def compute_values(self, rollout_id: int, rollout_data_ref: list) -> list[torch.Tensor]:
        """Compute per-token value predictions for all episodes (critic only).

        Returns:
            List of CPU tensors. values[i] has shape [episodes[i].response_length].
        """

        episodes = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
        init_dummy_advantages(episodes)
        packed_batches, grad_accum = self._packed_data(episodes)

        self.wake_up()
        self.model.eval()
        with timer("critic_compute_values"), torch.no_grad():
            for batch in tqdm(packed_batches, desc="critic_values", disable=dist.get_rank() != 0):
                model_args = self._get_model_inputs_args(batch)
                values = self.model(**model_args).logits.squeeze(-1).squeeze(0).float()
                batch["cur_values"] = strip_cross_boundary(values[:-1], batch["cu_seqlens"])

        self.model.train()

        # Unpack and reorder values back to original episode order
        all_values = [None] * len(episodes)
        for batch in packed_batches:
            ep_indices = batch.get("_episode_indices", [])
            unpacked = unpack_sequences(batch)
            for j, ub in enumerate(unpacked):
                all_values[ep_indices[j]] = ub["cur_values"].detach().cpu()

        # Cache packed data for reuse in train()
        self._pending_episodes = episodes
        self._pending_packed_batches = packed_batches
        self._pending_grad_accum = grad_accum

        return all_values

    def _run_train_loop(self, rollout_id: int, packed_batches: list, grad_accum: list) -> None:
        """Training loop for the critic model (value loss)."""
        self._log_train_metrics(packed_batches)
        with timer("critic_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in enumerate(
                tqdm(packed_batches, desc="critic_train", disable=dist.get_rank() != 0)
            ):
                self._train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )


    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        """Single training step for the critic (value loss with clipping)."""
        model_args = self._get_model_inputs_args(packed_batch)
        # Value head outputs [..., 1]; shift by 1 to align with response tokens (same as log_probs)
        values_output = self.model(**model_args).logits.squeeze(-1).squeeze(0).float()
        packed_batch["cur_values"] = strip_cross_boundary(values_output[:-1], packed_batch["cu_seqlens"])

        unpacked_batches = unpack_sequences(packed_batch)

        cur_values_list = [batch["cur_values"] for batch in unpacked_batches]
        old_values_list = [batch["old_values"].to(device=cur_values_list[0].device) for batch in unpacked_batches]
        returns_list = [batch["returns"].to(device=cur_values_list[0].device) for batch in unpacked_batches]
        loss_masks = [batch["loss_masks"].to(device=cur_values_list[0].device) for batch in unpacked_batches]
        edge_lengths = [batch["edge_lengths"] for batch in unpacked_batches]

        cur_values = torch.cat(cur_values_list, dim=0)
        old_values = torch.cat(old_values_list, dim=0)
        returns = torch.cat(returns_list, dim=0)

        value_loss = compute_value_loss(cur_values, old_values, returns, self.args.value_clip)
        value_loss = sum_of_sample_mean(value_loss, edge_lengths, loss_masks)

        reported = {"value_loss": value_loss.detach()}

        loss = value_loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        self._accumulate_and_step(reported, reported_accum, mbs_id, grad_accum)

    def _log_train_metrics(self, packed_batches):
        self._log_packed_metrics(packed_batches, ["old_values", "returns"])
