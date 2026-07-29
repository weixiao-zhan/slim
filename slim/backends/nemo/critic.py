# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel PPO value trainer."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from tqdm import tqdm

from slim.utils.data import process_rollout_data
from slim.utils.ppo_utils import compute_value_loss
from slim.utils.timer import timer

from .base import NeMoTrainer
from .checkpoint import is_hf_checkpoint
from .data_packing import init_dummy_advantages, token_slots_to_edges, unpack_sequences
from .forward import model_forward, prepare_forward
from .loss import count_global_denominators, normalize_sequence_values
from .model import CriticModel, build_optimizer, build_value_head, final_hidden_state
from .models import build_model

logger = logging.getLogger(__name__)


class CriticNeMoTrainer(NeMoTrainer):
    """Train a scalar value head over a NeMo AutoModel backbone."""

    role = "critic"
    _train_log_prefix = "train/critic"

    @property
    def model_parts(self) -> list[torch.nn.Module]:
        return [self.model, self.value_head]

    def _resolve_checkpoint_paths(self) -> str:
        self._checkpoint_load_dir = self.args.critic_load
        self._checkpoint_save_dir = self.args.critic_save
        if self.args.critic_load and is_hf_checkpoint(self.args.critic_load):
            return self.args.critic_load
        return self.args.hf_checkpoint

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        self.model = build_model(
            self.args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=False,
        )
        self.value_head = build_value_head(
            self.model,
            self.distributed_setup,
        )
        self._checkpoint_model = CriticModel(self.model, self.value_head)

        backbone_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in self.value_head.parameters() if parameter.requires_grad
        ]
        groups = [
            {
                "params": backbone_parameters,
                "max_lr": self.args.lr_critic,
            },
            {
                "params": head_parameters,
                "max_lr": self.args.lr_critic_value_head,
            },
        ]
        self.optimizer = build_optimizer(
            self.args,
            self.device_mesh,
            param_groups=groups,
        )
        self.model.train()
        self.value_head.train()

    def _padding_token_id(self) -> int:
        return self.tokenizer.pad_token_id or 0

    def _forward_values(self, prepared) -> torch.Tensor:
        model_batch = dict(prepared.model_batch)
        model_batch["output_hidden_states"] = True
        model_batch["logits_to_keep"] = 1
        output = model_forward(self.model, model_batch)
        return self.value_head(final_hidden_state(output))

    def compute_values(self, rollout_data_ref: list) -> list[torch.Tensor]:
        episodes = process_rollout_data(rollout_data_ref, self.dp_rank, self.dp_size)
        init_dummy_advantages(episodes)
        packed_batches, grad_accum = self._packed_data(episodes)

        self.wake_up()
        self.model.eval()
        self.value_head.eval()
        with timer("critic_compute_values"), torch.no_grad():
            for pack in tqdm(packed_batches, desc="critic_values", disable=dist.get_rank() != 0):
                prepared = prepare_forward(
                    self.model,
                    self.device_mesh,
                    pack,
                    padding_token_id=self._padding_token_id(),
                )
                with prepared.context_factory():
                    local_values = self._forward_values(prepared)
                full_values = prepared.gather(local_values, fill=0)
                pack["cur_values"] = token_slots_to_edges(
                    full_values.squeeze(0),
                    pack["cu_seqlens"],
                ).detach().cpu()
        self.model.train()
        self.value_head.train()

        all_values: list[torch.Tensor | None] = [None] * len(episodes)
        for pack in packed_batches:
            for episode_index, batch in zip(
                pack["_episode_indices"],
                unpack_sequences(pack),
                strict=True,
            ):
                all_values[episode_index] = batch["cur_values"]
        if any(value is None for value in all_values):
            raise RuntimeError("critic value reconstruction omitted an episode")

        self._pending_episodes = episodes
        self._pending_packed_batches = packed_batches
        self._pending_grad_accum = grad_accum
        return [value for value in all_values if value is not None]

    def _train_microbatch(
        self,
        pack: dict,
        *,
        is_final: bool,
        global_sequences: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prepared = prepare_forward(
            self.model,
            self.device_mesh,
            pack,
            padding_token_id=self._padding_token_id(),
        )
        if is_final:
            self._prepare_final_backward()
        with self._sync_context(is_final), prepared.context_factory():
            values = self._forward_values(prepared)
            mask = prepared.fields["loss_masks"].to(values.dtype)
            old_values = prepared.fields["old_values"].to(values.dtype)
            returns = prepared.fields["returns"].to(values.dtype)
            value_values, clip_values = compute_value_loss(
                values,
                old_values,
                returns,
                self.args.value_clip,
            )
            document_ids = prepared.fields["document_ids"]
            num_documents = len(pack["edge_lengths"])
            value_loss = normalize_sequence_values(
                value_values,
                mask,
                document_ids,
                num_documents,
                global_sequences,
                self.cp_group,
            )
            value_clipfrac = normalize_sequence_values(
                clip_values,
                mask,
                document_ids,
                num_documents,
                global_sequences,
                self.cp_group,
            )
            (value_loss * self.backward_group_size).backward()
        return {
            "value_loss": value_loss.detach(),
            "value_clipfrac": value_clipfrac.detach(),
        }

    def _run_train_loop(self, packed_batches: list, grad_accum: list[int]) -> None:
        self._log_packed_metrics(packed_batches, ["old_values", "returns"])
        progress = tqdm(packed_batches, desc="critic_train", disable=dist.get_rank() != 0)
        profiled = iter(self.prof.iterate_train_pg(enumerate(progress)))
        self.optimizer.zero_grad(set_to_none=True)
        with timer("critic_train"):
            for step_batches in self._optimizer_step_batches(packed_batches, grad_accum):
                global_sequences, _ = count_global_denominators(
                    step_batches,
                    self.dp_group,
                    torch.device("cuda", torch.cuda.current_device()),
                )
                self._begin_gradient_accumulation()
                metric_sums: dict[str, torch.Tensor] = {}
                for index in range(len(step_batches)):
                    _, pack = next(profiled)
                    metrics = self._train_microbatch(
                        pack,
                        is_final=index == len(step_batches) - 1,
                        global_sequences=global_sequences,
                    )
                    for name, value in metrics.items():
                        metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
                    if index == 0:
                        self._after_first_microbatch()
                grad_norm = self._optimizer_step()
                self._log_step(self._reduce_metrics(metric_sums), grad_norm)
