import logging
import os
import random
from argparse import Namespace
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig

from slime.ray.train_actor import TrainRayActor
from slime.utils import logging_utils, train_dump_utils, train_metric_utils
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.ppo_utils import (
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    compute_value_loss,
    vanilla_gae,
)
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.profile_utils import TrainProfiler
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import Episode

from . import checkpoint
from .data_packing import pack_sequences, unpack_sequences
from .lr_scheduler import get_lr_scheduler
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False, with_opd_teacher: bool = False) -> int:  # type: ignore[override]
        if with_opd_teacher:
            raise NotImplementedError(
                "On-policy distillation (OPD) with a local teacher model is not supported. "
                "Please use --opd-type=sglang with an external teacher server."
            )
        super().init(args, role, with_ref, with_opd_teacher)

        # Setup device mesh for data parallelism
        self._setup_device_mesh()
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.dp_size,
            "fsdp_strategy": self.args.fsdp_strategy,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self._apply_moe_patch()
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        # Determine checkpoint paths based on role
        self._is_critic = role == "critic"
        if self._is_critic:
            self._checkpoint_load_dir = args.critic_save  # resume from critic checkpoints
            self._checkpoint_save_dir = args.critic_save
            hf_checkpoint = args.critic_load or args.hf_checkpoint
            lr = args.critic_lr
        else:
            self._checkpoint_load_dir = args.load
            self._checkpoint_save_dir = args.save
            hf_checkpoint = args.hf_checkpoint
            lr = args.lr

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        init_context = self._get_init_weight_context_manager()

        # Downcast to bf16 if model config doesn't specify a dtype and flash attention is requested,
        # since FA2/FA3 require float16/bfloat16 (e.g. Gemma-3's Siglip vision encoder defaults to float32).
        load_dtype = getattr(self.hf_config, "torch_dtype", None)
        if load_dtype is None and self.args.attn_implementation in ("flash_attention_2", "flash_attention_3"):
            load_dtype = torch.bfloat16

        # Shared kwargs for from_pretrained — reused by _create_ref_model
        self._load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=self.args.attn_implementation,
        )
        if load_dtype is not None:
            self._load_kwargs["dtype"] = load_dtype

        if self._is_critic:
            from .models.critic import create_critic_model

            model = create_critic_model(
                hf_checkpoint,
                init_context=init_context,
                model_cls=self.get_model_cls(),
                **self._load_kwargs,
            )
        else:
            with init_context():
                model = self.get_model_cls().from_pretrained(hf_checkpoint, **self._load_kwargs)

        # Apply PEFT adapter if --use-peft is set (after critic head swap, before FSDP)
        model = self._maybe_apply_peft(model)

        model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

        self.model = model

        if args.gradient_checkpointing:
            # Gradient checkpointing requires inputs to have requires_grad=True
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:

                def _make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        # LR scheduler reads args.lr for max_lr; override for critic
        orig_lr = args.lr
        if self._is_critic:
            args.lr = lr
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        args.lr = orig_lr

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        # Critic does not need a reference model.
        self.ref_model = None
        if with_ref and not self._is_critic:
            self.ref_model = self._create_ref_model(args.ref_load)

        # Critic does not sync weights to rollout engines.
        self.weight_updater = None
        if not self._is_critic:
            self.weight_updater = (
                UpdateWeightFromTensor(self.args, self.model)
                if self.args.colocate
                else UpdateWeightFromDistributed(self.args, self.model)
            )

        # Handle to the paired critic (set by connect_actor_critic for actor role)
        self.critic_handle = None

        # Pre-computed data from compute_log_probs / compute_values, consumed by train()
        self._pending_episodes = None
        self._pending_packed_batches = None
        self._pending_grad_accum = None

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _apply_moe_patch(self):
        from .models.qwen3_moe_hf import apply_fsdp_moe_patch

        apply_fsdp_moe_patch()

    def _setup_device_mesh(self) -> None:
        """Setup device mesh for data parallelism."""
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.dp_size = world_size
        self.dp_rank = rank

        if self.args.fsdp_strategy == "hybrid":
            assert world_size == self.args.actor_num_nodes * self.args.actor_num_gpus_per_node, (
                f"world_size {world_size} != actor_num_nodes {self.args.actor_num_nodes} * "
                f"actor_num_gpus_per_node {self.args.actor_num_gpus_per_node}"
            )
            self.mesh = init_device_mesh(
                "cuda",
                mesh_shape=(self.args.actor_num_nodes, self.args.actor_num_gpus_per_node),
                mesh_dim_names=("replicate", "shard"),
            )
            self.dp_mesh = self.mesh
            self.dp_group = dist.new_group()
            logger.info(
                f"[Rank {rank}] Device mesh (2D HSDP): replicate={self.args.actor_num_nodes}, "
                f"shard={self.args.actor_num_gpus_per_node}, world_size={world_size}"
            )
        else:
            self.mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
            self.dp_mesh = self.mesh
            self.dp_group = self.mesh.get_group("dp")
            logger.info(f"[Rank {rank}] Device mesh (1D full shard): world_size={world_size}")

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _maybe_apply_peft(self, model):
        """Apply PEFT (LoRA/DoRA) adapter if --use-peft is set.

        Returns PeftModel if --use-peft is set, otherwise the original model unchanged.
        On resume, checkpoint.load() restores adapter weights via DCP after FSDP wrapping.
        """
        if not getattr(self.args, "use_peft", False):
            return model

        from peft import LoraConfig, get_peft_model

        defaults = dict(
            r=16,
            lora_alpha=32,
            use_dora=False,
            target_modules="all-linear",
            exclude_modules=["vision_tower", "multi_modal_projector"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        defaults.update(self.args.peft_config or {})
        config = LoraConfig(**defaults)
        logger.info(f"[Rank {dist.get_rank()}] Applying PEFT: {config}")
        model = get_peft_model(model, config)

        model.print_trainable_parameters()
        return model

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        save_dir = self._checkpoint_save_dir
        if self.args.debug_rollout_only or save_dir is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    def connect_actor_critic(self, critic_handle) -> None:
        """Store a handle to the paired critic Ray actor (for actor role)
        or to the paired actor Ray actor (for critic role)."""
        self.critic_handle = critic_handle

    def compute_values(self, rollout_id: int, rollout_data_refs: list) -> list[torch.Tensor]:
        """Compute per-token value predictions for all episodes (critic only).

        Returns:
            List of CPU tensors. values[i] has shape [episodes[i].response_length].
        """
        assert self._is_critic, "compute_values should only be called on the critic"

        if self.args.offload_train:
            self.wake_up()

        episodes = process_rollout_data(self.args, rollout_data_refs, self.dp_rank, self.dp_size)

        _init_dummy_advantages(episodes)

        # Use _packed_data which synchronizes batch count across DP ranks
        # (required for FSDP all-gather collectives during forward pass)
        packed_batches, grad_accum = self._packed_data(episodes)

        self.model.eval()
        with timer("critic_compute_values"), torch.no_grad():
            for batch in tqdm(packed_batches, desc="critic_values", disable=dist.get_rank() != 0):
                model_args = self._get_model_inputs_args(batch)
                values = self.model(**model_args).logits.squeeze(-1).squeeze(0).float()
                batch["cur_values"] = strip_cross_boundary(values[:-1], batch["cu_seqlens"])

        self.model.train()

        if self.args.offload_train:
            self.sleep()

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

    def compute_log_probs(self, rollout_id: int, rollout_data_refs: list) -> None:
        """Pre-compute log-probs and cache packed batches for the upcoming train() call.

        Packs episodes and computes ref/actor log-probs as needed. When
        use_rollout_logprobs is True, actor log-probs are skipped (rollout
        log-probs serve as the old baseline). The packed batches are stored
        on self for reuse in train(), avoiding redundant packing and forward passes.
        """
        assert not self._is_critic, "compute_log_probs should only be called on the actor"

        episodes = process_rollout_data(self.args, rollout_data_refs, self.dp_rank, self.dp_size)

        _init_dummy_advantages(episodes)

        packed_batches, grad_accum = self._packed_data(episodes)

        needs_forward = self.ref_model is not None or not self.args.use_rollout_logprobs
        if needs_forward:
            if self.args.offload_train:
                self.wake_up()

            if self.ref_model is not None:
                self._compute_log_prob("ref", packed_batches, store_prefix="ref_")
            if not self.args.use_rollout_logprobs:
                self._compute_log_prob("actor", packed_batches)

            if self.args.offload_train:
                self.sleep()

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

            active_model = self.ref_model
            active_model.eval()
        else:
            active_model = self.model

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    model_args = self._get_model_inputs_args(batch)
                    logits = active_model(**model_args).logits.squeeze(0).float()
                    log_probs_result, entropy_result = get_logprob_and_entropy(
                        logits=logits,
                        target_tokens=batch["tokens"],
                        allow_compile=True,
                        temperature=self.args.rollout_temperature,
                        need_full_log_probs=store_prefix == "" and self.args.use_rollout_entropy,
                        cu_seqlens=batch["cu_seqlens"],
                    )
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if entropy_result is not None:
                        batch["entropy"] = entropy_result
            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    def _packed_data(self, episodes: list[Episode]) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        """Pack variable-length episodes for efficient processing.

        Returns:
            A pair `(packed_batches, grad_accum)` where `packed_batches` is a list
            of packed batch dictionaries and `grad_accum` lists the micro-batch
            indices at which to perform optimizer steps.
        """
        packed_batches = []
        mbs_size_list = []
        local_batch_size = self.args.global_batch_size // self.dp_size
        assert (
            self.args.global_batch_size % self.dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {self.dp_size}"

        if self.args.use_dynamic_batch_size:
            max_tokens = self.args.max_tokens_per_gpu

            for i in range(0, len(episodes), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(ep.tokens) for ep in episodes[i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
                len(episodes) // local_batch_size
            )

        start = 0
        for mbs_size in num_microbatches:
            end = start + local_batch_size
            chunk_batches = pack_sequences(episodes[start:end], num_packs=mbs_size)
            # Offset _episode_indices to be absolute (relative to full episode list)
            for batch in chunk_batches:
                if "_episode_indices" in batch:
                    batch["_episode_indices"] = [idx + start for idx in batch["_episode_indices"]]
            packed_batches.extend(chunk_batches)
            start = end
        grad_accum = list(accumulate(num_microbatches))

        return packed_batches, grad_accum

    def train(self, rollout_id: int, rollout_data_refs: list, values_refs: list | None = None) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_refs: List of Ray ObjectRefs, one per DP rank,
                each containing a list[Episode] partition.
            values_refs: Optional list of Ray ObjectRefs containing per-sample
                value predictions from the critic (one ref per DP rank).
                Required when advantage_estimator=="ppo".
        """
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Use cached data from compute_log_probs / compute_values if available
            if self._pending_episodes is not None:
                episodes = self._pending_episodes
                packed_batches = self._pending_packed_batches
                grad_accum = self._pending_grad_accum
                self._pending_episodes = None
                self._pending_packed_batches = None
                self._pending_grad_accum = None
            else:
                episodes = process_rollout_data(self.args, rollout_data_refs, self.dp_rank, self.dp_size)
                packed_batches = None
                grad_accum = None

            if self.args.debug_rollout_only:
                return

            values = ray.get(values_refs[self.dp_rank]) if values_refs is not None else None
            self._train_core(
                rollout_id=rollout_id,
                episodes=episodes,
                values=values,
                packed_batches=packed_batches,
                grad_accum=grad_accum,
            )

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _log_rollout_data(self, rollout_id: int, episodes: list[Episode], packed_batches):
        log_dict = {}
        if dist.get_rank() == 0 and episodes:
            raw_rewards = [getattr(ep, "raw_reward", ep.reward) for ep in episodes]
            log_dict["rollout/raw_reward"] = sum(raw_rewards) / len(raw_rewards)

        for metric_key in ["log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"]:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for _mbs_id, batches in enumerate(packed_batches):
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            logger.info(f"rollout {rollout_id}: {log_dict}")
            log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            logging_utils.log(self.args, log_dict, step_key="rollout/step")

    def _train_core(
        self,
        rollout_id: int,
        episodes: list[Episode],
        values: list | None = None,
        packed_batches: list | None = None,
        grad_accum: list | None = None,
    ) -> None:
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            # For GRPO/GSPO, advantages = returns = reward repeated per edge
            for ep in episodes:
                ep._advantages = [ep.reward] * ep.num_edges
                ep._returns = ep._advantages
        elif self.args.advantage_estimator == "ppo":
            assert values is not None, "PPO requires value predictions from critic"
            self._compute_ppo_advantages(episodes, values)
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        if packed_batches is not None:
            # Reuse pre-packed batches, update advantages/returns with real values
            _update_packed_advantages(packed_batches, episodes)
        else:
            packed_batches, grad_accum = self._packed_data(episodes)

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self._is_critic:
            self._critic_train_loop(rollout_id, packed_batches, grad_accum)
        else:
            # Compute log-probs if not pre-computed
            if "log_probs" not in packed_batches[0] and "ref_log_probs" not in packed_batches[0]:
                if self.ref_model is not None:
                    self._compute_log_prob("ref", packed_batches, store_prefix="ref_")
                if not self.args.use_rollout_logprobs:
                    self._compute_log_prob("actor", packed_batches)

            self._log_rollout_data(rollout_id, episodes, packed_batches)

            with timer("actor_train"):
                reported_accum: dict[str, list[torch.Tensor]] = {}
                self.optimizer.zero_grad(set_to_none=True)
                for mbs_id, packed_batch in self.prof.iterate_train_actor(
                    enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
                ):
                    self._train_step(
                        packed_batch=packed_batch,
                        reported_accum=reported_accum,
                        mbs_id=mbs_id,
                        grad_accum=grad_accum,
                    )

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=None)

        # Update ref model if needed (copy actor weights to ref)
        if (
            not self._is_critic
            and self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _compute_ppo_advantages(self, episodes: list[Episode], values: list[torch.Tensor]) -> None:
        """Compute GAE advantages and returns for PPO, and store on episodes.

        Values are per-edge tensors (length = num_edges) from the critic.
        We compute GAE over all edges, then store per-edge advantages/returns.
        """
        assert len(episodes) == len(
            values
        ), f"Number of episodes ({len(episodes)}) != number of value predictions ({len(values)})"

        B = len(episodes)
        max_E = max(ep.num_edges for ep in episodes)

        # Build [B, E] reward and value tensors (padded)
        rewards_padded = torch.zeros(B, max_E)
        values_padded = torch.zeros(B, max_E)

        for i, ep in enumerate(episodes):
            E = ep.num_edges
            # Place reward at the last edge
            rewards_padded[i, E - 1] = ep.reward
            values_padded[i, :E] = values[i].float()

        advantages_padded, returns_padded = vanilla_gae(
            rewards_padded, values_padded, self.args.gamma, self.args.lambd
        )

        if self.args.normalize_advantages:
            advantages_flat = torch.cat([advantages_padded[i, : episodes[i].num_edges] for i in range(B)])
            # Global normalization across all DP ranks
            local_sum = advantages_flat.sum()
            local_sq_sum = (advantages_flat**2).sum()
            local_count = torch.tensor(float(advantages_flat.numel()))
            stats = torch.stack([local_sum, local_sq_sum, local_count]).to(torch.cuda.current_device())
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.dp_group)
            mean = stats[0] / stats[2]
            std = ((stats[1] / stats[2] - mean**2).clamp(min=0)).sqrt().clamp(min=1e-8)
            advantages_padded = (advantages_padded - mean.cpu()) / std.cpu()

        for i, ep in enumerate(episodes):
            E = ep.num_edges
            ep._advantages = advantages_padded[i, :E].tolist()
            ep._returns = returns_padded[i, :E].tolist()
            ep._values = values[i].tolist()

    def _critic_train_loop(self, rollout_id: int, packed_batches: list, grad_accum: list) -> None:
        """Training loop for the critic model (value loss)."""
        with timer("critic_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in enumerate(
                tqdm(packed_batches, desc="critic_train", disable=dist.get_rank() != 0)
            ):
                self._critic_train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

    def _critic_train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
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

        self._accumulate_and_step(reported, reported_accum, mbs_id, grad_accum, log_prefix="critic")

    def _accumulate_and_step(self, reported, reported_accum, mbs_id, grad_accum, log_prefix="train"):
        """Accumulate metrics, and on grad_accum boundaries: clip grads, step optimizer, log."""
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) not in grad_accum:
            return

        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad))
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # TODO: change this, this is slow.
        aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
        reduced_aggregated = [None] * self.dp_size
        dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
        aggregated = {k: sum(r[k] for r in reduced_aggregated) / self.args.global_batch_size for k in reported_accum}
        reported_accum.clear()

        if dist.get_rank() == 0:
            log_dict = {
                f"{log_prefix}/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
            }
            log_dict[f"{log_prefix}/grad_norm"] = grad_norm
            lr_values = self.lr_scheduler.get_last_lr()
            for gid, _group in enumerate(self.optimizer.param_groups):
                log_dict[f"{log_prefix}/lr-pg_{gid}"] = lr_values[gid]

            if log_prefix == "train" and self.args.use_kl_loss and "kl_loss" in aggregated:
                kl_info = f"kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                logger.info(kl_info)

            logger.info(f"{log_prefix} step {self.global_step}: {log_dict}")
            log_dict[f"{log_prefix}/step"] = self.global_step
            logging_utils.log(self.args, log_dict, step_key=f"{log_prefix}/step")

        self.global_step += 1

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        model_args = self._get_model_inputs_args(packed_batch)
        logits = self.model(**model_args).logits.squeeze(0).float()

        # Compute log probs and entropy
        need_full_log_probs = self.args.entropy_coef != 0.0
        log_probs, entropy_result = get_logprob_and_entropy(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            allow_compile=True,
            temperature=self.args.rollout_temperature,
            need_full_log_probs=need_full_log_probs,
            cu_seqlens=packed_batch["cu_seqlens"],
        )
        packed_batch["cur_log_probs"] = log_probs
        if entropy_result is not None:
            packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)

        old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if old_log_prob_key not in batch or not isinstance(batch[old_log_prob_key], torch.Tensor)
        ]
        if missing_old_log_probs:
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"use_rollout_logprobs is set to {self.args.use_rollout_logprobs}. Missing in batches: {missing_old_log_probs}"
            )
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        edge_lengths = [batch["edge_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        ppo_kl = old_log_probs - log_probs

        if self.args.use_opsm:
            opsm_mask, opsm_clipfrac = compute_opsm_mask(
                args=self.args,
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                advantages=[batch["advantages"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        if self.args.advantage_estimator == "gspo":
            ppo_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        if self.args.use_opsm:
            pg_loss = pg_loss * opsm_mask

        def _has_rollout_log_probs(batch) -> bool:
            rollout_tensor = batch.get("rollout_log_probs")
            return isinstance(rollout_tensor, torch.Tensor) and rollout_tensor.numel() > 0

        has_rollout_log_probs = all(_has_rollout_log_probs(batch) for batch in unpacked_batches)
        rollout_log_probs = (
            torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
            if has_rollout_log_probs
            else None
        )

        if self.args.calculate_per_token_loss:
            pg_loss = sum_of_token(pg_loss, edge_lengths, loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, edge_lengths, loss_masks)
            ppo_kl = sum_of_token(ppo_kl.abs(), edge_lengths, loss_masks)
        else:
            pg_loss = sum_of_sample_mean(pg_loss, edge_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, edge_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), edge_lengths, loss_masks)

        # Only compare rollout vs. train log probs when they originate from different stages.
        train_rollout_logprob_abs_diff = None
        if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
            train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
            train_rollout_logprob_abs_diff = sum_of_sample_mean(
                train_rollout_logprob_abs_diff, edge_lengths, loss_masks
            ).detach()

        if need_full_log_probs:
            entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
            entropy_loss = sum_of_sample_mean(entropy, edge_lengths, loss_masks)
        else:
            entropy_loss = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
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
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if train_rollout_logprob_abs_diff is not None:
            reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

        if self.args.use_kl_loss:
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_opsm:
            reported["opsm_clipfrac"] = opsm_clipfrac

        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        self._accumulate_and_step(reported, reported_accum, mbs_id, grad_accum, log_prefix="train")

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        Critic does not sync weights to rollout engines (no-op).
        """
        if self._is_critic:
            return
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.recover_and_get_updatable_engines.remote()
        )
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

        # PEFT: merge adapters into base weights so state_dict() returns merged weights
        # that sglang can use directly. Unmerge after sync to restore training state.
        # When model is on CPU (after offload_train/sleep), wake up for merge then sleep after.
        is_peft = hasattr(self.model, "peft_config")
        if is_peft:
            was_cpu = next(self.model.parameters()).is_cpu
            if was_cpu:
                self.wake_up()
            self.model.merge_adapter()
            try:
                self.weight_updater.update_weights(peft_remap=True)
            finally:
                self.model.unmerge_adapter()
                if was_cpu:
                    self.sleep()
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

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            with init_context():
                ref_model = self.get_model_cls().from_pretrained(ref_load_path, **self._load_kwargs)

            # Apply PEFT to ref model (fresh adapters, no checkpoint resume)
            ref_model = self._maybe_apply_peft(ref_model)

            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        position_ids = packed_sequence["position_ids"].unsqueeze(0)

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        if packed_sequence.get("multimodal_train_inputs"):
            model_args.update(packed_sequence["multimodal_train_inputs"])
        return model_args


def _init_dummy_advantages(episodes: list[Episode]) -> None:
    """Set zero advantages/returns on episodes so pack_sequences can proceed."""
    for ep in episodes:
        ep._advantages = [0.0] * ep.num_edges
        ep._returns = [0.0] * ep.num_edges


def _update_packed_advantages(packed_batches: list[dict], episodes: list[Episode]) -> None:
    """Update advantages/returns/old_values in pre-packed batches from episodes.

    After advantages are computed on episodes (e.g. via GAE), this function
    overwrites the dummy values that were used during initial packing.
    """
    for batch in packed_batches:
        ep_indices = batch["_episode_indices"]
        adv_parts = []
        ret_parts = []
        val_parts = []
        for idx in ep_indices:
            ep = episodes[idx]
            adv_parts.append(torch.tensor(ep._advantages, dtype=torch.float32))
            ret_parts.append(torch.tensor(ep._returns, dtype=torch.float32))
            values = getattr(ep, "_values", None)
            if values is not None:
                val_parts.append(torch.tensor(values, dtype=torch.float32))
        batch["advantages"] = torch.cat(adv_parts)
        batch["returns"] = torch.cat(ret_parts)
        if val_parts:
            batch["old_values"] = torch.cat(val_parts)


def strip_cross_boundary(flat_edges: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Remove cross-boundary entries from a flat edge tensor computed via [:-1].

    When multiple sequences are packed and we compute logits[:-1], the result
    has total_tokens - 1 entries, including cross-boundary entries between
    adjacent sequences.  This function extracts only the valid within-sequence
    edges (total_tokens - num_sequences entries).
    """
    if len(cu_seqlens) <= 2:
        # Single sequence — no cross-boundary entries
        return flat_edges
    indices = torch.cat([
        torch.arange(cu_seqlens[i], cu_seqlens[i + 1] - 1, device=flat_edges.device)
        for i in range(len(cu_seqlens) - 1)
    ])
    return flat_edges[indices]


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
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


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
        log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
        probs = torch.softmax(shifted_logits, dim=-1)
        entropy = -(probs * log_probs_full).sum(dim=-1)
    if cu_seqlens is not None:
        log_probs = strip_cross_boundary(log_probs, cu_seqlens)
        if entropy is not None:
            entropy = strip_cross_boundary(entropy, cu_seqlens)
    return log_probs, entropy


def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Compute sum of per-sample means across variable-length responses.

    Parameters:
        x: Flat tensor containing concatenated per-token values across samples.
        response_lengths: Lengths of each sample's response segment in `x`.
        loss_masks: Per-sample masks aligned with `response_lengths`.

    Returns:
        A scalar tensor equal to the sum over samples of the mean value within
        each sample's response segment.
    """
    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    # PeftModel doesn't expose _no_split_modules, so unwrap to find which layer classes FSDP should shard
    base_hf = getattr(model, "base_model", model)
    base_hf = getattr(base_hf, "model", base_hf)
    layer_cls_to_wrap = getattr(base_hf, "_no_split_modules", None)
    assert layer_cls_to_wrap and next(iter(layer_cls_to_wrap)) is not None
    model_config = getattr(base_hf, "config", model.config)

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model_config.tie_word_embeddings)
    ]

    # Determine precision policy based on args
    param_dtype = torch.bfloat16  # Default to bf16 as before
    reduce_dtype = torch.float32

    if args.fp16:
        param_dtype = torch.float16

    mesh_desc = f"{mesh.ndim}D mesh={mesh.shape}" if mesh is not None else "default"
    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, {mesh_desc}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def sum_of_token(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    return sum(
        [
            (x_i * loss_mask_i).sum()
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )
