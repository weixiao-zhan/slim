# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel policy trainer."""

from __future__ import annotations

import contextlib
import logging
import random
from functools import partial

import ray
import torch
import torch.distributed as dist
from nemo_automodel.components.moe.router_replay import RouterReplay

from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group
from slim.utils.memory_utils import clear_memory
from slim.utils.misc import load_function
from slim.utils.ppo_utils import compute_approx_kl, compute_score_centering_loss, compute_surrogate_weight
from slim.utils.quant import Quantizer
from slim.utils.timer import Timer, timer

from .base import NeMoTrainer, _move_module
from .checkpoint import is_hf_checkpoint, is_peft_adapter_dir, resolve_model_source
from .data_packing import (
    SEQUENCE_FIELDS,
    unpack_sequences,
)
from .loss import (
    entropy_from_logits,
    reduce_token_mean,
    reduce_weighted_sequence_mean,
    selective_log_probs,
    sequence_mean_at_tokens,
)
from .models import build_model
from .optimizer import build_optimizer
from .peft import lora_disabled
from .routing_replay import replay_router_targets
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class ActorNeMoTrainer(NeMoTrainer):
    """Train the policy and own its optional frozen reference model."""

    role = "actor"
    _train_log_prefix = "train/actor"
    pack_fields = (
        "advantages",
        "rollout_log_probs",
        "rollout_topk_log_probs",
        "rollout_topk_token_ids",
        "rollout_routed_experts",
    )

    def _resolve_checkpoint_paths(self) -> str:
        self._checkpoint_load_dir = self.args.load
        self._checkpoint_save_dir = self.args.save
        checkpoint_path, self.peft_config = resolve_model_source(
            self.args.load,
            self.args,
            self.args.actor_peft_config,
        )
        return checkpoint_path

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        RouterReplay.clear_registry()
        self.model = build_model(
            self.args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=self.args.use_rollout_routing_replay,
            peft_config=self.peft_config,
        )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = build_optimizer(
            self.args,
            self.device_mesh,
            self.model,
            param_groups=[{"params": parameters, "max_lr": self.args.lr_actor}],
        )
        self.model.train()

    @contextlib.contextmanager
    def _policy_offloaded(self, ref_model: torch.nn.Module | None = None):
        """Hold the policy on CPU for the block, and `ref_model` on GPU if given."""
        device = torch.device("cuda", torch.cuda.current_device())
        _move_module(self.model, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        if ref_model is not None:
            _move_module(ref_model, device)
            dist.barrier(group=get_gloo_group())
        try:
            yield
        finally:
            if ref_model is not None:
                _move_module(ref_model, "cpu")
            _move_module(self.model, device)
            dist.barrier(group=get_gloo_group())

    def _create_ref_model(self, checkpoint_path: str) -> torch.nn.Module:
        if not (is_hf_checkpoint(checkpoint_path) or is_peft_adapter_dir(checkpoint_path)):
            raise ValueError(f"reference must be a Hugging Face checkpoint or a NeMo PEFT adapter: {checkpoint_path}")
        # A NeMo PEFT adapter makes the reference --hf-checkpoint with that adapter.
        model_checkpoint, peft_config = resolve_model_source(checkpoint_path, self.args, None)
        if peft_config is not None and self.peft_config is None:
            raise ValueError("a NeMo PEFT adapter as the reference requires a PEFT actor")

        with self._policy_offloaded():
            ref_model = build_model(
                self.args,
                model_checkpoint,
                self.distributed_setup,
                routing_replay=False,
                peft_config=peft_config,
                frozen=True,
            )
            if peft_config is not None:
                self.checkpointer.load_model(ref_model, checkpoint_path)
            ref_model.eval()
            ref_model.requires_grad_(False)
            _move_module(ref_model, "cpu")
        return ref_model

    def _post_model_setup(self) -> None:
        self.ref_model = None
        # Without --ref-load the reference is --hf-checkpoint, which a PEFT policy computes with its adapters disabled.
        if self.args.kl_loss_coef != 0 and (self.args.ref_load is not None or self.peft_config is None):
            self.ref_model = self._create_ref_model(self.args.ref_load or self.args.hf_checkpoint)
        quantizer = Quantizer.maybe_from_checkpoint(self.args.rollout_load, self.args)
        if quantizer is not None:
            logger.info(
                "Rollout weight quantization enabled with %s from %s",
                type(quantizer).__name__,
                self.args.rollout_load,
            )
        updater_type = UpdateWeightFromTensor if self.args.rollout_colocate else UpdateWeightFromDistributed
        self.weight_updater = updater_type(self.args, self.model, quantizer)

    def _compute_ref_log_probs(self, packed_batches: list[dict]) -> None:
        if self.ref_model is None:
            ref_model, context = self.model, lora_disabled(self.model)
        else:
            ref_model, context = self.ref_model, self._policy_offloaded(self.ref_model)

        def forward(prepared):
            output = ref_model(**prepared.model_batch, logits_to_keep=prepared.loss_positions)
            local_log_probs = selective_log_probs(
                output.logits,
                prepared.fields["labels"][:, prepared.loss_positions],
                temperature=self.args.rollout_temperature,
            )
            return prepared.scatter_loss_values(local_log_probs)

        with context:
            self._forward_only(
                ref_model,
                packed_batches,
                "ref_log_probs",
                store_key="ref_log_probs",
                forward=forward,
            )

    def compute_log_probs(self, rollout_data_ref: list) -> None:
        batch = process_rollout_data(rollout_data_ref, self.dp_rank, self.bulk_fields)
        packed_batches, grad_accum = self._packed_data(batch)
        if self.args.kl_loss_coef != 0:
            self.wake_up()
            self._compute_ref_log_probs(packed_batches)

        self._cache_packed_data(packed_batches, grad_accum)

    def _policy_loss(
        self,
        logits: torch.Tensor,
        fields: dict[str, torch.Tensor],
        *,
        num_documents: int,
        global_sequences: torch.Tensor,
        global_tokens: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = fields["loss_masks"].to(torch.float32)
        document_ids = fields["document_ids"]

        def sequence_mean(values, value_mask=mask):
            return reduce_weighted_sequence_mean(
                values, value_mask, document_ids, num_documents, global_sequences, self.cp_group, loss_weights
            )

        def policy_mean(values, value_mask=mask):
            # `--loss-normalization-unit token` denominates the policy terms in tokens.
            if self.args.loss_normalization_unit == "token":
                return reduce_token_mean(values, value_mask, global_tokens)
            return sequence_mean(values, value_mask)

        # Read everything the loss needs from the logits in one fused pass, after the entropy:
        # the kernel overwrites the logits with their gradient.
        local_entropy = entropy_from_logits(logits) if self.args.entropy_coef != 0 else None
        scored_token_ids = fields["labels"].unsqueeze(-1)
        if self.args.use_score_centering is not None:
            scored_token_ids = torch.cat((scored_token_ids, fields["rollout_topk_token_ids"].long()), dim=-1)
        scored_log_probs = selective_log_probs(logits, scored_token_ids, temperature=self.args.rollout_temperature)
        log_probs, topk_log_probs = scored_log_probs[..., 0], scored_log_probs[..., 1:]
        rollout_log_probs = fields["rollout_log_probs"].to(log_probs.dtype)
        advantages = fields["advantages"].to(log_probs.dtype)

        score, rollout_score = log_probs, rollout_log_probs
        if self.args.advantage_estimator == "gspo":
            # GSPO weighs the sequence-mean score by the sequence-level ratio.
            score = sequence_mean_at_tokens(log_probs, mask, document_ids, num_documents, self.cp_group)
            rollout_score = sequence_mean_at_tokens(
                rollout_log_probs, mask, document_ids, num_documents, self.cp_group
            )
        policy_log_ratio = rollout_score - score

        # The detached surrogate, $-A \operatorname{sg}[w_x] \log p_x$.
        surrogate_weight = partial(
            compute_surrogate_weight,
            eps_clip=self.args.eps_clip,
            eps_clip_high=self.args.eps_clip_high,
            policy_surrogate=self.args.policy_surrogate,
            eps_clip_c=self.args.eps_clip_c,
        )
        weights, clipped = surrogate_weight((-policy_log_ratio).exp(), advantages)
        pg_values = -advantages * weights * score
        # Per-token diagnostics, logged as their weighted sequence mean.
        sequence_metrics = {}
        if self.args.use_score_centering is not None:
            # Score centering adds $A \sum_{v \in H} \operatorname{sg}[q_v w_v - \alpha p_v] \log p_v$.
            correction, topk_mass, rho = compute_score_centering_loss(
                topk_log_probs,
                fields["rollout_topk_log_probs"].to(topk_log_probs.dtype),
                advantages,
                surrogate_weight,
            )
            pg_values = pg_values + correction
            sequence_metrics["score_centering/topk_mass"] = topk_mass
            sequence_metrics["score_centering/rho"] = rho
        pg_loss = policy_mean(pg_values)

        zero = logits.new_zeros((), dtype=torch.float32)
        entropy_loss = zero if local_entropy is None else sequence_mean(local_entropy)
        kl_loss = zero
        if self.args.kl_loss_coef != 0:
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - rollout_log_probs)
            kl_values = compute_approx_kl(
                log_probs,
                fields["ref_log_probs"].to(log_probs.dtype),
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = sequence_mean(kl_values)

        loss = pg_loss - self.args.entropy_coef * entropy_loss + self.args.kl_loss_coef * kl_loss
        metrics = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": policy_mean(clipped).detach(),
            "pg_kl_k3": policy_mean(compute_approx_kl(rollout_score, score, kl_loss_type="k3")).detach(),
            "entropy_loss": entropy_loss.detach(),
        }
        if self.args.kl_loss_coef != 0:
            metrics["kl_loss"] = kl_loss.detach()
        for name, values in sequence_metrics.items():
            metrics[name] = sequence_mean(values).detach()
        return loss, metrics

    def _custom_loss(
        self,
        pack: dict,
        prepared,
        local_log_probs: torch.Tensor,
        local_entropy: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        full_pack = dict(pack)
        full_pack["cur_log_probs"] = prepared.gather(prepared.scatter_loss_values(local_log_probs)).squeeze(0)
        if local_entropy is not None:
            full_pack["entropy"] = prepared.gather(prepared.scatter_loss_values(local_entropy)).squeeze(0)
        batches = unpack_sequences(full_pack)
        device = local_log_probs.device
        for batch in batches:
            for name, value in list(batch.items()):
                if name in SEQUENCE_FIELDS and isinstance(value, torch.Tensor):
                    batch[name] = value.to(device)
        custom_loss = load_function(self.args.custom_loss_function_path)
        raw_loss, raw_metrics = custom_loss(self.args, batches)
        normalization = self.args.global_batch_size * self.cp_size
        loss = raw_loss / normalization
        metrics = {name: value.detach() / normalization for name, value in raw_metrics.items()}
        return loss, metrics

    def _forward_context(self, prepared):
        replay = self.args.use_rollout_routing_replay
        return replay_router_targets(prepared.fields["rollout_routed_experts"] if replay else None)

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
        output = self.model(**prepared.model_batch, logits_to_keep=prepared.loss_positions)
        if self.args.loss_type == "custom_loss":
            local_entropy = None
            if self.args.entropy_coef != 0:
                local_entropy = entropy_from_logits(output.logits)
            local_log_probs = selective_log_probs(
                output.logits,
                prepared.fields["labels"][:, prepared.loss_positions],
                temperature=self.args.rollout_temperature,
            )
            return self._custom_loss(pack, prepared, local_log_probs, local_entropy)
        # The logits cover only the loss positions, so the loss reads its fields there too.
        return self._policy_loss(
            output.logits,
            {name: value[:, prepared.loss_positions] for name, value in prepared.fields.items()},
            num_documents=num_documents,
            global_sequences=global_sequences,
            global_tokens=global_tokens,
            loss_weights=loss_weights,
        )

    def _run_train_loop(self, packed_batches: list, grad_accum: list[int]) -> None:
        if self.args.kl_loss_coef != 0 and "ref_log_probs" not in packed_batches[0]:
            self._compute_ref_log_probs(packed_batches)

        self._log_packed_metrics(packed_batches, ["ref_log_probs", "advantages"])
        self._run_optimizer_steps(packed_batches, grad_accum, "actor_train")

    def sleep(self) -> None:
        if self._need_offload:
            # The engines read the policy after it offloads, so its adapters merge while it is on the GPU.
            self.weight_updater.stage_merged_weights()
        super().sleep()

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only:
            return

        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.recover_engines.remote())
        dist.barrier(group=get_gloo_group())

        rollout_engines, num_new_engines, engine_actor_ranks = ray.get(self.rollout_manager.get_engines.remote())
        if dist.get_rank() == 0:
            Timer().add_count("restart_engines", num_new_engines)
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, engine_actor_ranks)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()

        if self.args.ci_test and rollout_engines:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"weight version mismatch: engine={engine_version}, updater={self.weight_updater.weight_version}"
                )
        clear_memory()
