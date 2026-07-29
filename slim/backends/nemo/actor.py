# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo AutoModel policy trainer."""

from __future__ import annotations

import logging
import os
import random

import ray
import torch
import torch.distributed as dist
from nemo_automodel.components.moe.router_replay import RouterReplay
from tqdm import tqdm

from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group
from slim.utils.memory_utils import clear_memory
from slim.utils.misc import load_function
from slim.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slim.utils.quant import Quantizer
from slim.utils.timer import Timer, timer

from .base import NeMoTrainer, _move_module
from .checkpoint import is_hf_checkpoint
from .data_packing import (
    SEQUENCE_FIELDS,
    fill_document_terminal_slots,
    unpack_sequences,
)
from .forward import model_forward, prepare_forward
from .loss import (
    count_global_denominators,
    entropy_from_logits,
    normalize_policy_values,
    normalize_sequence_values,
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
            param_groups=[{"params": parameters, "max_lr": self.args.lr_actor}],
        )
        self.model.train()

    def _create_ref_model(self, checkpoint_path: str) -> torch.nn.Module:
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"reference checkpoint must be a local directory: {checkpoint_path}")

        _move_module(self.model, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())

        ref_model = build_model(
            self.args,
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
        quantizer = Quantizer.maybe_from_checkpoint(self.args.hf_checkpoint)
        if quantizer is not None:
            logger.info(
                "Rollout weight quantization enabled with %s from %s",
                type(quantizer).__name__,
                self.args.hf_checkpoint,
            )
        updater_type = UpdateWeightFromTensor if self.args.rollout_colocate else UpdateWeightFromDistributed
        self.weight_updater = updater_type(self.args, self.model, quantizer)

    def _padding_token_id(self) -> int:
        return self.tokenizer.pad_token_id or 0

    def _routing_targets(self, prepared, *, required: bool):
        targets = prepared.fields.get("rollout_routed_experts")
        if required and targets is None:
            raise KeyError("rollout_routed_experts is required when rollout routing replay is enabled")
        return targets

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict],
        *,
        store_key: str,
    ) -> None:
        is_reference = model_tag == "ref"
        active_model = self.ref_model if is_reference else self.model
        if active_model is None:
            raise RuntimeError("reference log probabilities requested without a reference model")

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
                        padding_token_id=self._padding_token_id(),
                    )
                    replay_enabled = not is_reference and self.args.use_rollout_routing_replay
                    targets = self._routing_targets(prepared, required=replay_enabled)
                    with prepared.context_factory(), replay_router_targets(targets if replay_enabled else None):
                        output = model_forward(active_model, prepared.model_batch)
                        local_log_probs = selective_log_probs(
                            output.logits,
                            prepared.fields["labels"],
                            temperature=self.args.rollout_temperature,
                        )
                    full_log_probs = prepared.gather(local_log_probs, fill=0)
                    pack[store_key] = fill_document_terminal_slots(
                        full_log_probs.squeeze(0),
                        pack["cu_seqlens"],
                    ).detach().cpu()
        finally:
            if is_reference:
                active_model.eval()
                _move_module(active_model, "cpu")
                _move_module(self.model, torch.device("cuda", torch.cuda.current_device()))
                dist.barrier(group=get_gloo_group())
            else:
                active_model.train()

    def compute_log_probs(self, rollout_data_ref: list) -> None:
        episodes = process_rollout_data(rollout_data_ref, self.dp_rank, self.dp_size)
        packed_batches, grad_accum = self._packed_data(episodes)
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
        if "actor_old_log_probs" not in pack or "rollout_log_probs" not in pack:
            raise KeyError("mismatch correction requires actor_old_log_probs and rollout_log_probs")
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

    def _normalize_policy(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        document_ids: torch.Tensor,
        num_documents: int,
        global_sequences: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return normalize_policy_values(
            values,
            mask,
            document_ids,
            num_documents,
            sum_tokens=self.args.calculate_per_token_loss,
            global_sequences=global_sequences,
            global_tokens=global_tokens,
            cp_group=self.cp_group,
        )

    def _normalize_sequence(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        document_ids: torch.Tensor,
        num_documents: int,
        global_sequences: torch.Tensor,
    ) -> torch.Tensor:
        return normalize_sequence_values(
            values,
            mask,
            document_ids,
            num_documents,
            global_sequences,
            self.cp_group,
        )

    def _policy_loss(
        self,
        logits: torch.Tensor,
        fields: dict[str, torch.Tensor],
        *,
        num_documents: int,
        global_sequences: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = fields["labels"]
        mask = fields["loss_masks"].to(torch.float32)
        document_ids = fields["document_ids"]
        local_entropy = entropy_from_logits(logits) if self.args.entropy_coef != 0 else None
        log_probs = selective_log_probs(logits, labels, temperature=self.args.rollout_temperature)
        old_key = "actor_old_log_probs" if self.args.old_logprob_source == "actor" else "rollout_log_probs"
        if old_key not in fields:
            raise KeyError(f"{old_key} is required for the policy ratio")
        old_log_probs = fields[old_key].to(log_probs.dtype)
        advantages = fields["advantages"].to(log_probs.dtype)

        policy_log_ratio = old_log_probs - log_probs
        if self.args.advantage_estimator == "gspo":
            policy_log_ratio = sequence_mean_at_tokens(
                policy_log_ratio,
                mask,
                document_ids,
                num_documents,
                self.cp_group,
            )

        pg_values, clip_values = compute_policy_loss(
            policy_log_ratio,
            log_probs,
            advantages,
            self.args.eps_clip,
            self.args.eps_clip_high,
            self.args.policy_surrogate,
            self.args.eps_clip_c,
        )
        effective_mask = fields.get("mismatch_masks", mask).to(mask.dtype)
        if "mismatch_weights" in fields:
            pg_values = pg_values * fields["mismatch_weights"].to(pg_values.dtype)

        pg_loss = self._normalize_policy(
            pg_values,
            effective_mask,
            document_ids,
            num_documents,
            global_sequences,
            global_tokens,
        )
        pg_clipfrac = self._normalize_policy(
            clip_values,
            mask,
            document_ids,
            num_documents,
            global_sequences,
            global_tokens,
        )
        pg_kl_k3 = self._normalize_policy(
            torch.exp(-policy_log_ratio) + policy_log_ratio - 1,
            mask,
            document_ids,
            num_documents,
            global_sequences,
            global_tokens,
        )

        zero = logits.new_zeros((), dtype=torch.float32)
        entropy_loss = zero
        if local_entropy is not None:
            entropy_loss = self._normalize_sequence(
                local_entropy,
                mask,
                document_ids,
                num_documents,
                global_sequences,
            )

        kl_loss = zero
        if self.args.kl_loss_coef != 0:
            if "ref_log_probs" not in fields:
                raise KeyError("ref_log_probs is required when kl_loss_coef is nonzero")
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
            kl_values = compute_approx_kl(
                log_probs,
                fields["ref_log_probs"].to(log_probs.dtype),
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = self._normalize_sequence(
                kl_values,
                mask,
                document_ids,
                num_documents,
                global_sequences,
            )

        loss = pg_loss - self.args.entropy_coef * entropy_loss + self.args.kl_loss_coef * kl_loss
        metrics = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "pg_kl_k3": pg_kl_k3.detach(),
            "entropy_loss": entropy_loss.detach(),
        }
        if self.args.kl_loss_coef != 0:
            metrics["kl_loss"] = kl_loss.detach()
        if "actor_old_log_probs" in fields and "rollout_log_probs" in fields:
            mismatch_ratio = fields["actor_old_log_probs"] - fields["rollout_log_probs"]
            metrics["mismatch/kl_k3"] = self._normalize_sequence(
                torch.exp(mismatch_ratio) - mismatch_ratio - 1,
                mask,
                document_ids,
                num_documents,
                global_sequences,
            ).detach()
            metrics["mismatch/log_prob_abs_diff"] = self._normalize_sequence(
                mismatch_ratio.abs(),
                mask,
                document_ids,
                num_documents,
                global_sequences,
            ).detach()
        for name, values in fields.items():
            if not name.startswith("mismatch/"):
                continue
            metrics[name] = self._normalize_sequence(
                values,
                mask,
                document_ids,
                num_documents,
                global_sequences,
            ).detach()
        return loss, metrics

    def _custom_loss(
        self,
        pack: dict,
        prepared,
        local_log_probs: torch.Tensor,
        local_entropy: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        full_pack = dict(pack)
        full_pack["cur_log_probs"] = fill_document_terminal_slots(
            prepared.gather(local_log_probs, fill=0).squeeze(0),
            pack["cu_seqlens"],
        )
        if local_entropy is not None:
            full_pack["entropy"] = fill_document_terminal_slots(
                prepared.gather(local_entropy, fill=0).squeeze(0),
                pack["cu_seqlens"],
            )
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
        prepared = prepare_forward(
            self.model,
            self.device_mesh,
            pack,
            padding_token_id=self._padding_token_id(),
        )
        if is_final:
            self._prepare_final_backward()
        replay_enabled = self.args.use_rollout_routing_replay
        targets = self._routing_targets(prepared, required=replay_enabled)
        with (
            self._sync_context(is_final),
            prepared.context_factory(),
            replay_router_targets(targets if replay_enabled else None),
        ):
            output = model_forward(self.model, prepared.model_batch)
            if self.args.loss_type == "custom_loss":
                local_entropy = None
                if self.args.entropy_coef != 0:
                    local_entropy = entropy_from_logits(output.logits)
                local_log_probs = selective_log_probs(
                    output.logits,
                    prepared.fields["labels"],
                    temperature=self.args.rollout_temperature,
                )
                loss, metrics = self._custom_loss(pack, prepared, local_log_probs, local_entropy)
            else:
                loss, metrics = self._policy_loss(
                    output.logits,
                    prepared.fields,
                    num_documents=pack["cu_seqlens"].numel() - 1,
                    global_sequences=global_sequences,
                    global_tokens=global_tokens,
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
        progress = tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0)
        profiled = iter(self.prof.iterate_train_pg(enumerate(progress)))
        self.optimizer.zero_grad(set_to_none=True)
        with timer("actor_train"):
            for step_batches in self._optimizer_step_batches(packed_batches, grad_accum):
                global_sequences, global_tokens = count_global_denominators(
                    step_batches,
                    self.dp_group,
                    torch.cuda.current_device(),
                )
                self._begin_gradient_accumulation()
                metric_sums: dict[str, torch.Tensor] = {}
                for index in range(len(step_batches)):
                    _, pack = next(profiled)
                    metrics = self._train_microbatch(
                        pack,
                        is_final=index == len(step_batches) - 1,
                        global_sequences=global_sequences,
                        global_tokens=global_tokens,
                    )
                    for name, value in metrics.items():
                        metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
                    if index == 0:
                        self._after_first_microbatch()
                grad_norm = self._optimizer_step()
                self._log_step(self._reduce_metrics(metric_sums), grad_norm)

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

        if self.args.rollout_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_and_get_updatable_engines.remote())
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
