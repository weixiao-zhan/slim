# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel PPO value trainer."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states

from slim.utils.data import process_rollout_data
from slim.utils.ppo_utils import compute_value_loss

from .base import NeMoTrainer
from .checkpoint import resolve_model_source
from .data_packing import fill_document_terminal_slots, unpack_sequences
from .loss import reduce_weighted_sequence_mean
from .models import build_model
from .optimizer import build_optimizer

logger = logging.getLogger(__name__)


class ScalarValueHead(nn.Module):
    """Token-level scalar projection for the PPO critic."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).squeeze(-1).float()


class CriticModel(nn.Module):
    """AutoModel backbone and scalar value head."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, **kwargs) -> torch.Tensor:
        output = self.backbone(logits_to_keep=1, output_hidden_states=True, **kwargs)
        return self.value_head(get_final_hidden_states(output))


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
    def optimizer_model(self) -> torch.nn.Module:
        return self.critic_model

    def checkpoint_states(self) -> dict[str, torch.nn.Module]:
        return {"value_head": self.value_head}

    def _resolve_checkpoint_paths(self) -> str:
        self._checkpoint_load_dir = self.args.critic_load
        self._checkpoint_save_dir = self.args.critic_save
        checkpoint_path, self.peft_config = resolve_model_source(
            self.args.critic_load,
            self.args,
            self.args.critic_peft_config,
        )
        return checkpoint_path

    def _build_value_head(self) -> nn.Module:
        output_embeddings = self.model.get_output_embeddings()
        output_embeddings.requires_grad_(False)
        # The trained head is stored in --master-weight-dtype; `auto` follows the backbone's output embedding.
        head = ScalarValueHead(
            hidden_size=self.model.config.get_text_config().hidden_size,
            dtype=torch.float32 if self.args.master_weight_dtype == "fp32" else output_embeddings.weight.dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        strategy = self.distributed_setup.strategy_config
        fully_shard_by_dtype(
            head,
            get_fsdp_dp_mesh(self.device_mesh),
            strategy.mp_policy,
            strategy.offload_policy,
            reshard_after_forward=False,
        )
        return head

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        self.model = build_model(
            self.args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=False,
            peft_config=self.peft_config,
        )
        self.value_head = self._build_value_head()
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
