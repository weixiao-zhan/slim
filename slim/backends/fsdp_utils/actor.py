import logging
import os
import random

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

from slim.utils.data import process_rollout_data
from slim.utils.distributed_utils import get_gloo_group
from slim.utils.quant import Quantizer
from slim.utils.memory_utils import clear_memory
from slim.utils.ppo_utils import (
    compute_approx_kl,
    compute_gspo_kl,
    compute_policy_loss,
    sum_of_sample_mean,
    sum_of_token,
)
from slim.utils.misc import load_function
from slim.utils.timer import Timer, timer

from .base import FSDPTrainer
from .data_packing import (
    init_dummy_advantages,
    strip_cross_boundary,
    unpack_sequences,
)
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class ActorFSDPTrainer(FSDPTrainer):
    """FSDP trainer for the RL policy (actor + optional frozen reference model)."""

    _train_log_prefix = "train/actor"

    def _resolve_checkpoint_paths(self) -> str:
        from .checkpoint import is_hf_checkpoint

        args = self.args
        self._checkpoint_load_dir = args.load
        self._checkpoint_save_dir = args.save
        # When --load points to an HF checkpoint (e.g. BF16 model), use it for
        # training weight init so that --hf-checkpoint can independently point to
        # a quantized model (e.g. FP8) for the rollout engine.
        if args.load and is_hf_checkpoint(args.load):
            logger.info(
                f"Detected HF checkpoint at --load={args.load}; "
                f"using it for training init (--hf-checkpoint={args.hf_checkpoint} used for rollout/tokenizer)."
            )
            return args.load
        return args.hf_checkpoint

    def _create_model(self, hf_checkpoint: str, init_context):
        with init_context():
            model = self.get_model_cls().from_pretrained(hf_checkpoint, **self._load_kwargs)
        logger.info(f"[Rank {dist.get_rank()}] Actor model created from {hf_checkpoint}")
        return model

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model using actor-same CPU offload strategy.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if not os.path.isdir(ref_load_path):
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

        init_context = self._get_init_weight_context_manager()

        def _build_raw_ref():
            with init_context():
                return self.get_model_cls().from_pretrained(ref_load_path, **self._load_kwargs)

        ref_model = self._build_fsdp_model(_build_raw_ref, trainable=False)

        # The ref model shares policy trainer with actor model.
        # Need to explicit offload
        if not self.fsdp_cpu_offload:
            ref_model.cpu()

        logger.info(f"[Rank {dist.get_rank()}] Reference model created from {ref_load_path}")
        return ref_model

    def _build_optimizer_param_groups(self) -> list[dict]:
        return [{"params": list(self.model.parameters()), "max_lr": self.args.lr_actor}]

    def _post_model_setup(self) -> None:
        args = self.args
        self.ref_model = None
        if self.with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        quantizer = Quantizer.maybe_from_checkpoint(args.hf_checkpoint)
        if quantizer is not None:
            logger.info(
                f"Rollout weight quantization enabled ({type(quantizer).__name__}) from {args.hf_checkpoint}"
            )
        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model, quantizer)
            if self.args.rollout_colocate
            else UpdateWeightFromDistributed(self.args, self.model, quantizer)
        )

    def _maybe_update_ref_model(self, rollout_id: int) -> None:
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            if not self.fsdp_cpu_offload:
                self.ref_model.cpu()

    def compute_log_probs(self, rollout_id: int, rollout_data_ref: list) -> None:
        """Pre-compute log-probs and cache packed batches for the upcoming train() call.

        Packs episodes and computes ref/actor-old log-probs as needed. The
        packed batches are stored on self for reuse in train(), avoiding
        redundant packing and forward passes.
        """

        episodes = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
        init_dummy_advantages(episodes)
        packed_batches, grad_accum = self._packed_data(episodes)

        needs_forward = self.ref_model is not None or self._needs_actor_old_log_probs()
        if needs_forward:
            self.wake_up()

            if self.ref_model is not None:
                self._compute_log_prob("ref", packed_batches, store_prefix="ref_")
            if self._needs_actor_old_log_probs():
                self._compute_log_prob("actor", packed_batches, store_prefix="actor_old_")
            self._deactivate_routing_replay()
            # Stay resident; the immediately-following train() reuses the model on GPU.

        # Cache for train()
        self._pending_episodes = episodes
        self._pending_packed_batches = packed_batches
        self._pending_grad_accum = grad_accum

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict[str, torch.Tensor]],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities for a list of packed batches.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            packed_batches: A list of packed batch dictionaries produced by
                `pack_sequences`, each containing at least `tokens` and
                `position_ids`; may also include multimodal keys like `pixel_values`.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs". The
            actual per-sequence results are written in-place into each element of
            `packed_batches` under the same key and can be read back by callers.

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                self.ref_model.cuda()
                dist.barrier(group=get_gloo_group())
            active_model = self.ref_model.eval()
        else:
            # old log-prob is a snapshot of the sampling policy (merged-LoRA,
            # dropout off in sglang); match it by forwarding self.model in eval.
            active_model = self.model.eval()

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    model_args = self._get_model_inputs_args(batch)
                    # Replay only on actor forward; ref model uses its own router.
                    with self._maybe_routing_replay(batch, enabled=model_tag != "ref"):
                        logits = active_model(**model_args).logits.squeeze(0)
                    log_probs_result, entropy_result = get_logprob_and_entropy(
                        logits=logits,
                        target_tokens=batch["tokens"],
                        allow_compile=True,
                        temperature=self.args.rollout_temperature,
                        need_full_log_probs=False,
                        cu_seqlens=batch["cu_seqlens"],
                    )
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if entropy_result is not None:
                        batch["entropy"] = entropy_result
            return rollout_data

        finally:
            if model_tag == "ref" and self.ref_model is not None and not self.fsdp_cpu_offload:
                # Restore actor model if it was offloaded
                self.ref_model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                self.model.cuda()
                dist.barrier(group=get_gloo_group())
            else:
                # Restore train mode after the eval-mode old-log-prob forward.
                self.model.train()

    def _run_train_loop(self, rollout_id: int, packed_batches: list, grad_accum: list) -> None:
        # Compute log-probs if not pre-computed
        if self.ref_model is not None and "ref_log_probs" not in packed_batches[0]:
            self._compute_log_prob("ref", packed_batches, store_prefix="ref_")
        if self._needs_actor_old_log_probs() and "actor_old_log_probs" not in packed_batches[0]:
            self._compute_log_prob("actor", packed_batches, store_prefix="actor_old_")

        self._log_train_metrics(packed_batches)

        with timer("actor_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in self.prof.iterate_train_pg(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                self._train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        model_args = self._get_model_inputs_args(packed_batch)
        with self._maybe_routing_replay(packed_batch, enabled=True):
            logits = self.model(**model_args).logits.squeeze(0)

        # Compute log probs and entropy
        log_probs, entropy_result = get_logprob_and_entropy(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            allow_compile=True,
            temperature=self.args.rollout_temperature,
            need_full_log_probs=self.args.entropy_coef != 0.0,
            cu_seqlens=packed_batch["cu_seqlens"],
        )
        packed_batch["cur_log_probs"] = log_probs
        if entropy_result is not None:
            packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)

        if self.args.loss_type == "custom_loss":
            custom_loss_func = load_function(self.args.custom_loss_function_path)
            loss, reported = custom_loss_func(self.args, unpacked_batches)
        else:
            loss, reported = self._policy_loss(unpacked_batches)

        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        self._accumulate_and_step(reported, reported_accum, mbs_id, grad_accum)


    def _policy_loss(self, unpacked_batches):
        """policy-gradient loss (pg + entropy + optional KL).

        Returns ``(loss, reported)`` where ``loss`` is the summed-microbatch loss
        and ``reported`` is a dict of detached scalar metrics logged under train/.
        """
        need_full_log_probs = self.args.entropy_coef != 0.0
        old_log_prob_key = "actor_old_log_probs" if self.args.old_logprob_source == "actor" else "rollout_log_probs"
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if (
                old_log_prob_key not in batch
                or not isinstance(batch[old_log_prob_key], torch.Tensor)
                or batch[old_log_prob_key].numel() == 0
            )
        ]
        if missing_old_log_probs:
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"old_logprob_source={self.args.old_logprob_source}. Missing in batches: {missing_old_log_probs}"
            )
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        edge_lengths = [batch["edge_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        pg_kl = old_log_probs - log_probs

        if self.args.advantage_estimator == "gspo":
            pg_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        pg_kl_k3 = torch.exp(-pg_kl) + pg_kl - 1

        pg_loss, pg_clipfrac = compute_policy_loss(
            pg_kl,
            log_probs,
            advantages,
            self.args.eps_clip,
            self.args.eps_clip_high,
            self.args.policy_surrogate,
            self.args.eps_clip_c,
        )

        def _has_nonempty_tensor(batch, key: str) -> bool:
            tensor = batch.get(key)
            return isinstance(tensor, torch.Tensor) and tensor.numel() > 0

        has_rollout_log_probs = all(_has_nonempty_tensor(batch, "rollout_log_probs") for batch in unpacked_batches)
        has_actor_old_log_probs = all(_has_nonempty_tensor(batch, "actor_old_log_probs") for batch in unpacked_batches)

        mismatch_loss_masks = loss_masks
        mismatch_weights = None
        mismatch_modified_masks = loss_masks
        mismatch_metrics = {}
        has_mismatch_log_probs = has_rollout_log_probs and has_actor_old_log_probs
        if has_mismatch_log_probs:
            from slim.utils.mismatch import compute_mismatch_metrics

            _, _, mismatch_metrics = compute_mismatch_metrics(
                args=self.args,
                train_log_probs=[batch["actor_old_log_probs"] for batch in unpacked_batches],
                rollout_log_probs=[batch["rollout_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        run_mismatch = self.args.mismatch_correction != "none" or self.args.get_mismatch_metrics
        if run_mismatch:
            if not has_mismatch_log_probs:
                raise KeyError("mismatch correction requires rollout_log_probs and actor_old_log_probs.")

            if self.args.custom_mismatch_correction_function_path is not None:
                mismatch_func = load_function(self.args.custom_mismatch_correction_function_path)
                mismatch_weights, mismatch_modified_masks, custom_mismatch_metrics = mismatch_func(
                    args=self.args,
                    train_log_probs=[batch["actor_old_log_probs"] for batch in unpacked_batches],
                    rollout_log_probs=[batch["rollout_log_probs"] for batch in unpacked_batches],
                    loss_masks=loss_masks,
                )
                if custom_mismatch_metrics:
                    mismatch_metrics |= custom_mismatch_metrics
            if self.args.mismatch_correction != "none" and mismatch_weights is not None:
                flat_weights = torch.cat(mismatch_weights, dim=0).to(device=pg_loss.device)
                pg_loss = pg_loss * flat_weights
            if self.args.mismatch_correction != "none":
                mismatch_loss_masks = mismatch_modified_masks

        if self.args.calculate_per_token_loss:
            pg_loss = sum_of_token(pg_loss, edge_lengths, mismatch_loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, edge_lengths, loss_masks)
            pg_kl_k3 = sum_of_token(pg_kl_k3, edge_lengths, loss_masks)
        else:
            pg_loss = sum_of_sample_mean(pg_loss, edge_lengths, mismatch_loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, edge_lengths, loss_masks)
            pg_kl_k3 = sum_of_sample_mean(pg_kl_k3, edge_lengths, loss_masks)

        if need_full_log_probs:
            entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
            entropy_loss = sum_of_sample_mean(entropy, edge_lengths, loss_masks)
        else:
            entropy_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.kl_loss_coef != 0:
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = sum_of_sample_mean(kl, edge_lengths, loss_masks)

            loss = loss + self.args.kl_loss_coef * kl_loss

        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "pg_kl_k3": pg_kl_k3.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if mismatch_metrics:
            for key, values in mismatch_metrics.items():
                flat_v = torch.cat(values, dim=0)
                reported[f"mismatch/{key}"] = sum_of_sample_mean(flat_v, edge_lengths, loss_masks).detach()

        if self.args.kl_loss_coef != 0:
            reported["kl_loss"] = kl_loss.detach()

        return loss, reported

    def _needs_actor_old_log_probs(self) -> bool:
        return (
            self.args.old_logprob_source == "actor"
            or self.args.mismatch_correction != "none"
            or self.args.get_mismatch_metrics
        )
    
    def _log_train_metrics(self, packed_batches):
        self._log_packed_metrics(packed_batches, ["actor_old_log_probs", "ref_log_probs", "advantages"])

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        # Two-phase recovery:
        # Phase 1: only rank 0 triggers recovery so start_engines() sets
        #          num_new_engines exactly once (subsequent calls would reset it to 0).
        if self.args.rollout_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_and_get_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        # Phase 2: all ranks read the now-stable engine handles.
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

        # PEFT weight sync: state_dict() must return merged (base + adapter) weights
        # colocate: sleep() already merged before moving to CPU, wake_up() will unmerge.
        # otherwise: model is on GPU, merge/sync/unmerge in place.
        is_peft = hasattr(self.model, "peft_config")
        if is_peft:
            if self._need_offload:
                # Already merged by sleep(), just sync.
                self.weight_updater.update_weights(peft_remap=True)
            else:
                self.model.merge_adapter()
                try:
                    self.weight_updater.update_weights(peft_remap=True)
                finally:
                    self.model.unmerge_adapter()
        else:
            self.weight_updater.update_weights()

        if self.args.ci_test and len(rollout_engines) > 0:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        clear_memory()


def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.

    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.

    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.

    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


# Liger-kernel path: log p(target|ctx) = -CE(logits, target). Skips the [T, V] log_softmax
# intermediate and does FP32 reductions per-tile internally, so BF16 logits are safe even
# at vocab=248K.
_LIGER_CE = LigerCrossEntropyLoss(ignore_index=-100, reduction="none")
def selective_log_softmax_liger(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Liger-fused equivalent of selective_log_softmax, without log_softmax materialization."""
    shape = input_ids.shape
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_ids = input_ids.reshape(-1).to(torch.long)
    return (-_LIGER_CE(flat_logits, flat_ids)).reshape(shape)


def gather_log_probs_packed(
    shifted_logits: torch.Tensor,
    input_ids: torch.Tensor,
    allow_compile: bool,
    cu_seqlens: torch.Tensor | float | None = None,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather next-token log probabilities for packed sequences.

    Parameters:
        logits: Model logits of shape [B, T, V] or [T, V].
        input_ids: Token ids of shape [B, T] or [T].
        cu_seqlens: Optional cumulative sequence lengths (unused here). Present
            for API compatibility with callers.

    Returns:
        A tensor of shape [T-1] (or [B, T-1]) with log-probabilities of targets.
    """
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    if shifted_logits.dim() == 3:
        # Remove batch dimension for packed sequences
        shifted_logits = shifted_logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        shifted_logits = shifted_logits.div(temperature)

    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    # selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax_liger(shifted_logits, targets)


def get_logprob_and_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
    need_full_log_probs: bool = True,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute log probabilities and entropy.

    Parameters:
        logits: Model output logits with shape [seq_len, vocab_size]
        target_tokens: Target tokens with shape [seq_len]
        allow_compile: Whether to allow compilation
        temperature: Temperature parameter (optional)
        cu_seqlens: Cumulative sequence lengths for packed sequences.
            When provided, cross-boundary entries between packed sequences
            are stripped so the output is edge-aligned (sum of per-sequence
            num_tokens - 1).

    Returns:
        log_probs: Edge-aligned log probabilities
        entropy: Optional edge-aligned entropy
    """
    shifted_logits = logits[:-1, :]
    log_probs = gather_log_probs_packed(
        shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
    )
    entropy = None
    if need_full_log_probs:
        shifted_logits_f32 = shifted_logits.float()
        log_probs_full = torch.log_softmax(shifted_logits_f32, dim=-1)
        probs = torch.softmax(shifted_logits_f32, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
    if cu_seqlens is not None:
        log_probs = strip_cross_boundary(log_probs, cu_seqlens)
        if entropy is not None:
            entropy = strip_cross_boundary(entropy, cu_seqlens)
    return log_probs, entropy

