# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel PPO value trainer."""

from __future__ import annotations

import logging

import torch

from slim.utils.data import process_rollout_data
from slim.utils.ppo_utils import compute_value_loss

from .base import NeMoTrainer
from .data_packing import fill_document_terminal_slots, unpack_sequences
from .loss import reduce_weighted_sequence_mean
from .model import CriticModel, build_optimizer, build_value_head
from .models import build_model

logger = logging.getLogger(__name__)


class CriticNeMoTrainer(NeMoTrainer):
    """Train a scalar value head over a NeMo AutoModel backbone."""

    role = "critic"
    _train_log_prefix = "train/critic"
    bulk_fields = ("multimodal_inputs",)
    pack_fields = ("values", "value_targets")

    @property
    def model_parts(self) -> list[torch.nn.Module]:
        return [self.model, self.value_head]

    @property
    def checkpoint_model(self) -> torch.nn.Module:
        return self.critic_model

    def _resolve_checkpoint_paths(self) -> str:
        self._checkpoint_load_dir = self.args.critic_load
        self._checkpoint_save_dir = self.args.critic_save
        return self._resolve_model_checkpoint(self.args.critic_peft_config)

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        self.model = build_model(
            self.args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=False,
            peft_config=self.peft_config,
        )
        self.value_head = build_value_head(
            self.model,
            self.distributed_setup,
            self.args,
        )
        self.critic_model = CriticModel(self.model, self.value_head)

        backbone_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in self.value_head.parameters() if parameter.requires_grad
        ]
        groups = [
            {
                "params": backbone_parameters,
                "name": "critic",
                "max_lr": self.args.lr_critic,
                "start_step": self.args.lr_critic_start_step * self.args.num_steps_per_rollout,
            },
            {
                "params": head_parameters,
                "name": "critic_value_head",
                "max_lr": self.args.lr_critic_value_head,
                "start_step": self.args.lr_critic_value_head_start_step
                * self.args.num_steps_per_rollout,
            },
        ]
        self.optimizer = build_optimizer(
            self.args,
            self.device_mesh,
            self.model,
            param_groups=groups,
        )
        self.model.train()
        self.value_head.train()

    def compute_values(self, rollout_data_ref: list) -> dict:
        batch = process_rollout_data(rollout_data_ref, self.dp_rank, self.bulk_fields)
        packed_batches, grad_accum = self._packed_data(batch)

        self.wake_up()
        self._forward_only(
            self.model,
            packed_batches,
            "critic_compute_values",
            store_key="cur_values",
            forward=lambda prepared: self.critic_model(**prepared.model_batch),
        )

        all_values: list[torch.Tensor | None] = [None] * len(batch)
        for pack in packed_batches:
            pack["cur_values"] = fill_document_terminal_slots(pack["cur_values"], pack["cu_seqlens"])
            for document_index, document in zip(
                pack["_document_indices"],
                unpack_sequences(pack),
                strict=True,
            ):
                all_values[document_index] = document["cur_values"]

        self._cache_packed_data(packed_batches, grad_accum)
        # The advantage estimator reads the values of CP rank 0; every CP rank holds the same values.
        return {
            "dp_rank": self.dp_rank,
            "cp_rank": self.cp_rank,
            "values": all_values if self.cp_rank == 0 else None,
        }

    def _microbatch_loss(
        self,
        pack: dict,
        prepared,
        *,
        num_documents: int,
        global_sequences: torch.Tensor,
        global_tokens: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = self.critic_model(**prepared.model_batch)
        mask = prepared.fields["loss_masks"].to(values.dtype)
        old_values = prepared.fields["old_values"].to(values.dtype)
        value_targets = prepared.fields["value_targets"].to(values.dtype)
        value_values, clip_values = compute_value_loss(
            values,
            old_values,
            value_targets,
            self.args.value_clip,
        )
        document_ids = prepared.fields["document_ids"]
        value_loss = reduce_weighted_sequence_mean(
            value_values,
            mask,
            document_ids,
            num_documents,
            global_sequences,
            self.cp_group,
            loss_weights,
        )
        value_clipfrac = reduce_weighted_sequence_mean(
            clip_values,
            mask,
            document_ids,
            num_documents,
            global_sequences,
            self.cp_group,
            loss_weights,
        )
        return value_loss, {
            "value_loss": value_loss.detach(),
            "value_clipfrac": value_clipfrac.detach(),
        }

    def _run_train_loop(self, packed_batches: list, grad_accum: list[int]) -> None:
        self._log_packed_metrics(packed_batches, ["old_values", "value_targets"])
        self._run_optimizer_steps(packed_batches, grad_accum, "critic_train")
