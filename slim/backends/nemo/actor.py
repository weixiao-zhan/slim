# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel policy trainer."""

from __future__ import annotations

import copy
import logging
import os
import random
from functools import partial

import ray
import torch
import torch.distributed as dist
from nemo_automodel.components.moe.router_replay import RouterReplay
from tqdm import tqdm

from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group
from slim.utils.memory_utils import clear_memory
from slim.utils.misc import load_function
from slim.utils.ppo_utils import compute_approx_kl, compute_score_centering_loss, compute_surrogate_weight
from slim.utils.quant import Quantizer
from slim.utils.timer import Timer, timer

from .base import NeMoTrainer, _move_module
from .checkpoint import is_hf_checkpoint
from .data_packing import (
    SEQUENCE_FIELDS,
    unpack_sequences,
)
from .packed_cp_forward import prepare_forward
from .loss import (
    entropy_from_logits,
    reduce_token_mean,
    reduce_weighted_sequence_mean,
    selective_log_probs,
    sequence_mean_at_tokens,
)
from .model import build_optimizer
from .models import build_model
from .routing_replay import replay_router_targets
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class ActorNeMoTrainer(NeMoTrainer):
    """Train the policy and own its optional frozen reference model."""

    role = "actor"
    _train_log_prefix = "train/actor"

    def _resolve_checkpoint_paths(self) -> str:
        self._checkpoint_load_dir = self.args.load
        self._checkpoint_save_dir = self.args.save
        if self.args.load and is_hf_checkpoint(self.args.load):
            return self.args.load
        return self.args.hf_checkpoint

    def _create_model_and_optimizer(self, checkpoint_path: str) -> None:
        RouterReplay.clear_registry()
        self.model = build_model(
            self.args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=self.args.use_rollout_routing_replay,
        )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = build_optimizer(
            self.args,
            self.device_mesh,
            self.model,
            param_groups=[{"params": parameters, "max_lr": self.args.lr_actor}],
        )
        self.model.train()

    def _create_ref_model(self, checkpoint_path: str) -> torch.nn.Module:
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"reference checkpoint must be a local directory: {checkpoint_path}")

        _move_module(self.model, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())

        # The frozen reference has no optimizer, so no master weights: it keeps the checkpoint's dtype.
        ref_args = copy.copy(self.args)
        ref_args.master_weight_dtype = "auto"
        ref_model = build_model(
            ref_args,
            checkpoint_path,
            self.distributed_setup,
            routing_replay=False,
        )
        ref_model.eval()
        ref_model.requires_grad_(False)
        _move_module(ref_model, "cpu")
        _move_module(self.model, torch.device("cuda", torch.cuda.current_device()))
        dist.barrier(group=get_gloo_group())
        return ref_model

    def _post_model_setup(self) -> None:
        self.ref_model = self._create_ref_model(self.args.ref_load) if self.args.kl_loss_coef != 0 else None
        quantizer = Quantizer.maybe_from_checkpoint(self.args.rollout_load)
        if quantizer is not None:
            logger.info(
                "Rollout weight quantization enabled with %s from %s",
                type(quantizer).__name__,
                self.args.rollout_load,
            )
        updater_type = UpdateWeightFromTensor if self.args.rollout_colocate else UpdateWeightFromDistributed
        self.weight_updater = updater_type(self.args, self.model, quantizer)

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict],
        *,
        store_key: str,
    ) -> None:
        is_reference = model_tag == "ref"
        active_model = self.ref_model if is_reference else self.model

        if is_reference:
            _move_module(self.model, "cpu")
            clear_memory()
            dist.barrier(group=get_gloo_group())
            _move_module(active_model, torch.device("cuda", torch.cuda.current_device()))
            dist.barrier(group=get_gloo_group())

        active_model.eval()
        try:
            with timer(store_key), torch.no_grad():
                iterator = tqdm(packed_batches, desc=store_key, disable=dist.get_rank() != 0)
                for pack in self.prof.iterate_train_log_probs(iterator):
                    prepared = prepare_forward(
                        active_model,
                        self.device_mesh,
                        pack,
                        padding_token_id=self.tokenizer.pad_token_id,
                    )
                    replay_enabled = not is_reference and self.args.use_rollout_routing_replay
                    targets = prepared.fields["rollout_routed_experts"] if replay_enabled else None
                    with prepared.context_factory(), replay_router_targets(targets):
                        output = active_model(**prepared.model_batch, logits_to_keep=prepared.loss_positions)
                        local_log_probs = selective_log_probs(
                            output.logits,
                            prepared.fields["labels"][:, prepared.loss_positions],
                            temperature=self.args.rollout_temperature,
                        )
                    pack[store_key] = prepared.gather(prepared.scatter_loss_values(local_log_probs)).squeeze(0).cpu()
        finally:
            if is_reference:
                active_model.eval()
                _move_module(active_model, "cpu")
                _move_module(self.model, torch.device("cuda", torch.cuda.current_device()))
                dist.barrier(group=get_gloo_group())
            else:
                active_model.train()

    def compute_log_probs(self, rollout_data_ref: list) -> None:
        batch = process_rollout_data(rollout_data_ref, self.dp_rank, self.dp_size)
        packed_batches, grad_accum = self._packed_data(batch)
        if self.ref_model is not None or self._needs_actor_old_log_probs():
            self.wake_up()
            if self.ref_model is not None:
                self._compute_log_prob("ref", packed_batches, store_key="ref_log_probs")
            if self._needs_actor_old_log_probs():
                self._compute_log_prob("actor", packed_batches, store_key="actor_old_log_probs")

        self._cache_packed_data(packed_batches, grad_accum)

    def _prepare_mismatch(self, pack: dict) -> None:
        if not (self.args.mismatch_correction != "none" or self.args.get_mismatch_metrics):
            return
        if self.args.mismatch_correction == "none":
            return

        batches = unpack_sequences(pack)
        correction = load_function(self.args.custom_mismatch_correction_function_path)
        weights, masks, custom_metrics = correction(
            args=self.args,
            train_log_probs=[batch["actor_old_log_probs"] for batch in batches],
            rollout_log_probs=[batch["rollout_log_probs"] for batch in batches],
            loss_masks=[batch["loss_masks"] for batch in batches],
        )
        if weights is not None:
            pack["mismatch_weights"] = torch.cat([torch.as_tensor(value) for value in weights])
        if masks is not None:
            pack["mismatch_masks"] = torch.cat([torch.as_tensor(value) for value in masks])
        if custom_metrics:
            pack["_mismatch_metrics"] = {
                name: torch.cat([torch.as_tensor(value) for value in values])
                for name, values in custom_metrics.items()
            }

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
        old_key = "actor_old_log_probs" if self.args.old_logprob_source == "actor" else "rollout_log_probs"
        old_log_probs = fields[old_key].to(log_probs.dtype)
        advantages = fields["advantages"].to(log_probs.dtype)

        score, old_score = log_probs, old_log_probs
        if self.args.advantage_estimator == "gspo":
            # GSPO weighs the sequence-mean score by the sequence-level ratio.
            score = sequence_mean_at_tokens(log_probs, mask, document_ids, num_documents, self.cp_group)
            old_score = sequence_mean_at_tokens(old_log_probs, mask, document_ids, num_documents, self.cp_group)
        policy_log_ratio = old_score - score

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
        effective_mask = fields.get("mismatch_masks", mask).to(mask.dtype)
        if "mismatch_weights" in fields:
            pg_values = pg_values * fields["mismatch_weights"].to(pg_values.dtype)
        pg_loss = policy_mean(pg_values, effective_mask)

        zero = logits.new_zeros((), dtype=torch.float32)
        entropy_loss = zero if local_entropy is None else sequence_mean(local_entropy)
        kl_loss = zero
        if self.args.kl_loss_coef != 0:
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
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
            "pg_kl_k3": policy_mean(torch.exp(-policy_log_ratio) + policy_log_ratio - 1).detach(),
            "entropy_loss": entropy_loss.detach(),
        }
        if self.args.kl_loss_coef != 0:
            metrics["kl_loss"] = kl_loss.detach()
        if "actor_old_log_probs" in fields and "rollout_log_probs" in fields:
            mismatch_ratio = fields["actor_old_log_probs"] - fields["rollout_log_probs"]
            sequence_metrics["mismatch/kl_k3"] = torch.exp(mismatch_ratio) - mismatch_ratio - 1
            sequence_metrics["mismatch/log_prob_abs_diff"] = mismatch_ratio.abs()
        sequence_metrics.update((name, values) for name, values in fields.items() if name.startswith("mismatch/"))
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

    def _train_microbatch(
        self,
        pack: dict,
        *,
        is_final: bool,
        global_sequences: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_weights = torch.tensor(
            pack["loss_weights"],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        prepared = prepare_forward(
            self.model,
            self.device_mesh,
            pack,
            padding_token_id=self.tokenizer.pad_token_id,
        )
        if is_final:
            self._prepare_final_backward()
        replay_enabled = self.args.use_rollout_routing_replay
        targets = prepared.fields["rollout_routed_experts"] if replay_enabled else None
        with (
            self._sync_context(is_final),
            prepared.context_factory(),
            replay_router_targets(targets),
        ):
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
                loss, metrics = self._custom_loss(pack, prepared, local_log_probs, local_entropy)
            else:
                # The logits cover only the loss positions, so the loss reads its fields there too.
                loss, metrics = self._policy_loss(
                    output.logits,
                    {name: value[:, prepared.loss_positions] for name, value in prepared.fields.items()},
                    num_documents=pack["cu_seqlens"].numel() - 1,
                    global_sequences=global_sequences,
                    global_tokens=global_tokens,
                    loss_weights=loss_weights,
                )
            (loss * self.backward_group_size).backward()
        return metrics

    def _run_train_loop(self, packed_batches: list, grad_accum: list[int]) -> None:
        if self.ref_model is not None and "ref_log_probs" not in packed_batches[0]:
            self._compute_log_prob("ref", packed_batches, store_key="ref_log_probs")
        if self._needs_actor_old_log_probs() and "actor_old_log_probs" not in packed_batches[0]:
            self._compute_log_prob("actor", packed_batches, store_key="actor_old_log_probs")
        for pack in packed_batches:
            self._prepare_mismatch(pack)

        self._log_packed_metrics(packed_batches, ["actor_old_log_probs", "ref_log_probs", "advantages"])
        self._run_optimizer_steps(packed_batches, grad_accum, "actor_train")

    def _maybe_update_ref_model(self, rollout_id: int) -> None:
        if (
            self.ref_model is None
            or self.args.ref_update_interval is None
            or (rollout_id + 1) % self.args.ref_update_interval
        ):
            return
        self.ref_model.load_state_dict(self.model.state_dict())

    def _needs_actor_old_log_probs(self) -> bool:
        return (
            self.args.old_logprob_source == "actor"
            or self.args.mismatch_correction != "none"
            or self.args.get_mismatch_metrics
        )

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.recover_updatable_engines.remote())
        dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )
        if dist.get_rank() == 0:
            Timer().add_count("restart_engines", num_new_engines)
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        self.weight_updater.update_weights()

        if self.args.ci_test and rollout_engines:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"weight version mismatch: engine={engine_version}, updater={self.weight_updater.weight_version}"
                )
        clear_memory()
