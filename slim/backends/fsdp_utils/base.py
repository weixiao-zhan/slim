"""Base FSDP trainer shared by the policy and critic roles.

Holds the role-agnostic skeleton: distributed setup, device mesh, the shared
FSDP model-build primitive, the train/save/sleep/wake lifecycle, and the role
hooks overridden by PolicyFSDPTrainer / CriticFSDPTrainer.
"""

import contextlib
import logging
import os
import random
from argparse import Namespace
from datetime import timedelta
from itertools import accumulate

import ray
import torch
import torch.distributed as dist

from transformers import AutoConfig

import slim.utils.eval_config
from slim.ray.ray_worker import RayWorker
from slim.utils import logging_utils, train_dump_utils, train_metric_utils
from slim.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slim.utils.distributed_utils import get_gloo_group, init_gloo_group
from slim.utils.logging_utils import configure_logger, init_tracking
from slim.utils.memory_utils import clear_memory, print_memory
from slim.utils.ppo_utils import vanilla_gae
from slim.utils.processing_utils import load_processor, load_tokenizer
from slim.utils.profile_utils import TrainProfiler
from slim.utils.timer import Timer, inverse_timer, timer, with_defer
from slim.utils.types import Episode

from . import checkpoint
from .data_packing import pack_sequences, update_packed_advantages
from .fsdp_helpers import (
    _get_replicated_frozen_module_roots,
    apply_fsdp2,
    get_local_gpu_id,
    move_torch_optimizer,
)
from .lr_scheduler import get_lr_scheduler
from .models import apply_hf_model_patches
from .routing_replay import RoutingReplay

logger = logging.getLogger(__name__)


class FSDPTrainer(RayWorker):
    """Base trainer for pure HF+FSDP training, shared by actor and critic.

    Responsibilities:
      * Set up the distributed process group (run inside a Ray actor)
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide the shared train / save / update_weights skeleton

    Role-specific behavior lives in the ``PolicyFSDPTrainer`` /
    ``CriticFSDPTrainer`` subclasses, which override the hooks
    ``_resolve_checkpoint_paths`` / ``_create_model`` /
    ``_build_optimizer_param_groups`` / ``_post_model_setup`` /
    ``_run_train_loop`` / ``_train_step`` / ``_log_train_metrics``.

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    #: wandb/log prefix for this role's training metrics; overridden per subclass.
    _train_log_prefix = "train"

    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()

        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def _init_distributed(self, args: Namespace, role: str, with_ref: bool) -> None:
        """Set up the torch.distributed process group and NUMA affinity."""
        self.args = args
        self.role = role
        self.with_ref = with_ref

        torch.serialization.add_safe_globals([slim.utils.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

        # Use hybrid backend when FSDP CPU offload is enabled with a CPU backend
        backend = args.distributed_backend
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            cpu_backend = args.fsdp_cpu_backend
            backend = f"cpu:{cpu_backend},cuda:{args.distributed_backend}"
            logger.info(f"FSDP CPU offload enabled, using hybrid backend: {backend}")

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        try:
            if torch.version.hip is not None:
                logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
            else:
                import pynvml

                pynvml.nvmlInit()
                local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node
                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                pynvml.nvmlDeviceSetCpuAffinity(handle)
                logger.info(f"Set NUMA affinity for GPU {local_rank}")
                pynvml.nvmlShutdown()
        except ImportError:
            logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Warning: Failed to set NUMA affinity: {e}")

    def clear_memory(self):
        if self.args.debug_rollout_only:
            return
        print_memory("before FSDPTrainer.clear_memory")
        clear_memory()
        print_memory("after FSDPTrainer.clear_memory")

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        self._init_distributed(args, role, with_ref)

        # Setup device mesh for data parallelism
        self._setup_device_mesh()
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        self._need_offload = (
            self.args.rollout_colocate or self.args.critic_colocate
        ) and not self.fsdp_cpu_offload

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        # Determine checkpoint paths and the HF checkpoint to init weights from (role-specific).
        hf_checkpoint = self._resolve_checkpoint_paths()

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self._routing_replay_adapter = apply_hf_model_patches(self.hf_config, self.args)

        init_context = self._get_init_weight_context_manager()

        load_dtype = torch.float32 if self.args.master_weight_dtype == "fp32" else None

        # Shared kwargs for from_pretrained — reused by _create_ref_model
        self._load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=self.args.attn_implementation,
        )
        if load_dtype is not None:
            self._load_kwargs["dtype"] = load_dtype

        self.model = self._build_fsdp_model(
            lambda: self._create_model(hf_checkpoint, init_context),
            trainable=True,
        )

        if self.fsdp_cpu_offload:
            self._register_replicated_frozen_module_offload_hooks(self.model)

        if self._routing_replay_adapter is not None:
            n_routers = self._routing_replay_adapter.register_layer_indices(self.model)
            if dist.get_rank() == 0:
                logger.info(f"[routing-replay] tagged {n_routers} MoE routers with layer indices")

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
            # Param groups (and their per-group max_lr/start_step) are role-specific.
            self.optimizer = torch.optim.AdamW(
                self._build_optimizer_param_groups(),
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Role-specific post-setup (ref model, weight updater, ...).
        self._post_model_setup()

        # Handle to the paired critic (set by connect_actor_critic for actor role)
        self.critic_handle = None

        # Pre-computed data from compute_log_probs / compute_values, consumed by train()
        self._pending_episodes = None
        self._pending_packed_batches = None
        self._pending_grad_accum = None

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    # ------------------------------------------------------------------
    # Role hooks: overridden by PolicyFSDPTrainer / CriticFSDPTrainer.
    # ------------------------------------------------------------------

    def _resolve_checkpoint_paths(self) -> str:
        """Set ``self._checkpoint_load_dir`` / ``self._checkpoint_save_dir`` and
        return the HF checkpoint path to initialize training weights from."""
        raise NotImplementedError

    def _create_model(self, hf_checkpoint: str, init_context):
        """Build the (pre-FSDP) model for this role."""
        raise NotImplementedError

    def _build_optimizer_param_groups(self) -> list[dict]:
        """Return optimizer param groups (with per-group ``max_lr`` / ``start_step``)."""
        raise NotImplementedError

    def _post_model_setup(self) -> None:
        """Role-specific setup after the model/optimizer/checkpoint are ready."""
        pass

    def _run_train_loop(self, rollout_id: int, packed_batches: list, grad_accum: list) -> None:
        """Iterate microbatches for one rollout update (role-specific)."""
        raise NotImplementedError

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum) -> None:
        """Single microbatch: forward, loss, backward, optimizer step (role-specific)."""
        raise NotImplementedError

    def _log_train_metrics(self, packed_batches) -> None:
        """Log role-specific pre-train metrics. Default: no-op."""
        pass

    def update_weights(self) -> None:
        """Synchronize training weights to rollout engines. Default: no-op
        (only the actor pushes weights to the rollout engines)."""
        pass

    def connect_actor_critic(self, critic_handle) -> None:
        """Store a handle to the paired actor/critic Ray actor."""
        self.critic_handle = critic_handle

    def _steps_per_rollout(self) -> int:
        """Optimizer steps per rollout, used to convert rollout-unit start-steps
        into optimizer-step units for the LR scheduler."""
        return (
            self.args.rollout_batch_size * self.args.n_samples_per_prompt // self.args.global_batch_size
        )

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _setup_device_mesh(self) -> None:
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        self.dp_size = world_size
        self.dp_rank = rank

        if self.role == "critic":
            shard = self.args.critic_num_gpus_per_replica
        else:
            shard = self.args.actor_num_gpus_per_replica
        assert world_size % shard == 0, f"world_size {world_size} not divisible by num_gpus_per_replica {shard}"
        replicate = world_size // shard

        if shard == world_size:
            self.mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
            self.dp_mesh = self.mesh
            self.dp_group = self.mesh.get_group("dp")
            logger.info(f"[Rank {rank}] Device mesh (1D full shard): world_size={world_size}")
        else:
            # 2D mesh shards within the last dim and replicates across the first.
            # shard == 1 -> pure DDP; 1 < shard < world_size -> HSDP.
            self.mesh = init_device_mesh(
                "cuda",
                mesh_shape=(replicate, shard),
                mesh_dim_names=("replicate", "shard"),
            )
            self.dp_mesh = self.mesh
            self.dp_group = dist.new_group()
            kind = "DDP" if shard == 1 else "HSDP"
            logger.info(
                f"[Rank {rank}] Device mesh (2D {kind}): replicate={replicate}, "
                f"shard={shard}, world_size={world_size}"
            )

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
            exclude_modules=["visual", "vision_tower", "vision_model", "audio", "speech"],
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

    def _build_fsdp_model(self, raw_model_factory, *, apply_peft: bool = True, trainable: bool = True):
        """Build an FSDP2-wrapped model from a raw-model factory.

        Shared by the policy model, the reference model, and the critic model:
        builds the raw module, keeps replicated/frozen encoders in bf16, applies
        PEFT, captures the full state dict, wraps with FSDP2, and broadcasts the
        weights from rank 0.

        Per-role divergence stays with the caller: ``raw_model_factory`` builds
        the role's module (plain causal LM, value-head-swapped, ...), and
        ``trainable`` gates ``model.train()`` (the frozen ref keeps eval mode).
        The caller is responsible for role-specific post-steps such as assigning
        ``self.model``, registering offload hooks, or parking the model on CPU.
        """
        model = raw_model_factory()

        # Frozen modules stay in bf16, fp32 master weights aren't needed.
        for module in _get_replicated_frozen_module_roots(model).values():
            module.to(torch.bfloat16)

        # Apply PEFT adapter if --use-peft is set (after critic head swap, before FSDP)
        if apply_peft:
            model = self._maybe_apply_peft(model)

        if trainable:
            model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        return self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

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

    def _register_replicated_frozen_module_offload_hooks(self, model) -> None:
        """Stage frozen replicated modules GPU<->CPU around their forward.

        These modules are ``ignored_params`` so ``CPUOffloadPolicy`` never stages them. 
        The pre-forward hook copies params to GPU; the post-forward hook copies them back.
        No backward_hook since they are frozen.
        """
        device = torch.cuda.current_device()

        def _to(module, target):
            for p in module.parameters(recurse=True):
                p.data = p.data.to(target, non_blocking=True)
            for b in module.buffers(recurse=True):
                b.data = b.data.to(target, non_blocking=True)

        for name, module in _get_replicated_frozen_module_roots(model).items():
            module.register_forward_pre_hook(lambda m, _args: _to(m, device))
            module.register_forward_hook(lambda m, _args, _out: _to(m, "cpu"))
            logger.info(f"[cpu-offload] on-demand GPU staging for: '{name}'")

    @timer
    def sleep(self) -> None:
        """
        Offload the trainer to CPU.
        No-op if not self._need_offload.
        """
        if not self._need_offload:
            return

        print_memory("before offload model")

        if hasattr(self.model, "peft_config"):
            self.model.merge_adapter()

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """
        Resume the trainer onto GPU; Inverse of ``sleep``.
        No-op if not self._need_offload.
        """
        if not self._need_offload:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")

        if hasattr(self.model, "peft_config"):
            self.model.unmerge_adapter()

        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        save_dir = self._checkpoint_save_dir
        if self.args.debug_rollout_only or save_dir is None:
            return

        assert not self.args.async_save, "FSDPTrainer does not support async_save yet."
        checkpoint.save(self, rollout_id)

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

    def train(self, rollout_id: int, rollout_data_ref: list, values_refs: list | None = None) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: List of Ray ObjectRefs, one per DP rank,
                each containing a list[Episode] partition.
            values_refs: Optional list of Ray ObjectRefs containing per-sample
                value predictions from the critic (one ref per DP rank).
                Required when advantage_estimator=="ppo_gae".
        """
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
                episodes = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
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
            self._deactivate_routing_replay()

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

        self.sleep()
        clear_memory()

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
        elif self.args.advantage_estimator == "ppo_gae":
            assert values is not None, "PPO requires value predictions from critic"
            self._compute_ppo_advantages(episodes, values)
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        if packed_batches is not None:
            # Reuse pre-packed batches, update advantages/returns with real values
            update_packed_advantages(packed_batches, episodes)
        else:
            packed_batches, grad_accum = self._packed_data(episodes)

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        self._run_train_loop(rollout_id, packed_batches, grad_accum)

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=None)

        self._maybe_update_ref_model(rollout_id)

    def _maybe_update_ref_model(self, rollout_id: int) -> None:
        """Hook for periodic ref-model refresh. Default: no-op (actor overrides)."""
        pass

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

    def _accumulate_and_step(self, reported, reported_accum, mbs_id, grad_accum):
        """Accumulate metrics, and on grad_accum boundaries: clip grads, step optimizer, log."""
        log_prefix = self._train_log_prefix
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

            if self.args.use_kl_loss and "kl_loss" in aggregated:
                kl_info = f"kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                logger.info(kl_info)

            logger.info(f"{log_prefix} step {self.global_step}: {log_dict}")
            log_dict["train/step"] = self.global_step
            logging_utils.log(self.args, log_dict)

        self.global_step += 1

    def _log_packed_metrics(self, packed_batches, metric_keys):
        """Log the loss-masked mean of per-token quantities in the packed batches.

        Each subclass passes the role-specific keys it cares about; they go to
        ``{self._train_log_prefix}/<key>`` (``train/actor/*`` or ``train/critic/*``).
        Keys absent from the batch are skipped. Values are reduced across DP ranks
        and normalized by the global sample count.
        """
        from .data_packing import unpack_sequences

        log_dict = {}
        for metric_key in metric_keys:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for batches in packed_batches:
                for unpacked_batch in unpack_sequences(batches):
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            log_dict[f"{self._train_log_prefix}/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            logger.info(f"{self._train_log_prefix} train metrics: {log_dict}")
            log_dict["train/step"] = self.global_step
            logging_utils.log(self.args, log_dict)

    @contextlib.contextmanager
    def _maybe_routing_replay(self, packed_batch: dict, *, enabled: bool):
        """Activate the MoE routing-replay buffer for the upcoming forward.

        IMPORTANT: replay must stay active across BOTH the original forward
        AND the gradient-checkpointing recomputation that runs during backward.
        We deliberately do NOT deactivate when this context exits — the next
        batch's activate() overwrites the buffer, and end-of-rollout cleanup
        is the trainer's responsibility (see ``_deactivate_routing_replay``).
        """
        if not (enabled and self._routing_replay_adapter is not None):
            yield
            return

        routed = packed_batch.get("rollout_routed_experts")
        if routed is None:
            yield
            return

        # expand router index from edge align to token align.
        device = torch.cuda.current_device()
        cu_seqlens = packed_batch["cu_seqlens"].to(device)
        total_tokens = int(cu_seqlens[-1].item())

        # Each sequence's last token has no successor edge → zero-pad row;
        # all other token positions take their edge row in order.
        is_pad_row = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        is_pad_row[cu_seqlens[1:] - 1] = True
        padded = torch.zeros(
            (total_tokens, *routed.shape[1:]), dtype=routed.dtype, device=device
        )
        padded[~is_pad_row] = routed.to(device=device, non_blocking=True)

        RoutingReplay.activate(padded)
        yield  # NOTE: deactivate is intentionally not called here; see docstring.

    def _deactivate_routing_replay(self) -> None:
        if self._routing_replay_adapter is not None:
            RoutingReplay.deactivate()

    def _get_model_inputs_args(self, packed_sequence: dict) -> dict:
        input_ids = packed_sequence["tokens"].unsqueeze(0)
        mm_inputs = packed_sequence.get("multimodal_inputs") or {}

        # Pass cu_seq_lens_q/k + max_length_q/k to the model so that HF's
        # attention layers use the safe varlen-kwargs branch (sidesteps
        # flash-attn #2381's _is_packed_sequence misinference) and patched
        # Qwen DeltaNet layers reset recurrent state at episode boundaries.
        cu_seqlens = packed_sequence["cu_seqlens"].to(
            device=input_ids.device, dtype=torch.int32, non_blocking=True
        )
        max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        model_args = {
            "input_ids": input_ids,
            "attention_mask": None,
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_len,
            "max_length_k": max_len,
        }

        if mm_inputs:
            # VLM path: build [4, 1, N] position_ids per episode. Axis 0 is the
            # text axis (episode-local arange); axes 1-3 are 3D MRope positions
            # from Qwen3_5Model.get_rope_index. HF strips axis 0 for the
            # causal mask and forwards [3, 1, N] to the attention kernel.
            model_args["position_ids"] = self._build_vlm_position_ids(packed_sequence, mm_inputs)
            model_args.update(mm_inputs)
        else:
            model_args["position_ids"] = packed_sequence["position_ids"].unsqueeze(0)

        return model_args

    def _build_vlm_position_ids(self, packed_sequence: dict, mm_inputs: dict) -> torch.Tensor:
        """Build [4, 1, N] position_ids for a packed VLM batch.

        Walks cu_seqlens to slice each episode's tokens, synthesizes
        mm_token_type_ids from input_ids (0=text, 1=image, 2=video), and calls
        the HF model's get_rope_index to get 3D MRope positions. Prepends an
        episode-local arange as the text axis so HF's varlen detection works.
        """
        tokens = packed_sequence["tokens"]
        cu_seqlens = packed_sequence["cu_seqlens"]
        device = tokens.device

        image_token_id = getattr(self.hf_config, "image_token_id", None)
        video_token_id = getattr(self.hf_config, "video_token_id", None)

        image_grid_thw = mm_inputs.get("image_grid_thw")
        video_grid_thw = mm_inputs.get("video_grid_thw")
        num_items = packed_sequence.get("multimodal_num_items") or {}
        image_counts = num_items.get("image_grid_thw", [])
        video_counts = num_items.get("video_grid_thw", [])

        # Unwrap PEFT (base_model.model) then walk HF's .model nesting to
        # reach the base model that defines get_rope_index.
        inner = self.model
        if getattr(self.args, "use_peft", False):
            inner = inner.base_model.model
        inner = inner.model
        get_rope_index = inner.get_rope_index

        pieces = []
        img_cursor = 0
        vid_cursor = 0
        num_episodes = len(cu_seqlens) - 1
        for i in range(num_episodes):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            n = e - s
            ids = tokens[s:e].unsqueeze(0)  # [1, n]

            mm_tti = torch.zeros_like(ids, dtype=torch.int)
            if image_token_id is not None:
                mm_tti = mm_tti + (ids == image_token_id).int()
            if video_token_id is not None:
                mm_tti = mm_tti + 2 * (ids == video_token_id).int()

            n_img = image_counts[i] if i < len(image_counts) else 0
            n_vid = video_counts[i] if i < len(video_counts) else 0
            img_slice = image_grid_thw[img_cursor : img_cursor + n_img] if n_img else None
            vid_slice = video_grid_thw[vid_cursor : vid_cursor + n_vid] if n_vid else None
            img_cursor += n_img
            vid_cursor += n_vid

            rope_axes, _ = get_rope_index(
                input_ids=ids,
                mm_token_type_ids=mm_tti,
                image_grid_thw=img_slice,
                video_grid_thw=vid_slice,
            )  # [3, 1, n]
            text_axis = torch.arange(n, device=device).view(1, 1, n).expand(1, 1, n)
            pieces.append(torch.cat([text_axis, rope_axes], dim=0))  # [4, 1, n]

        return torch.cat(pieces, dim=-1)  # [4, 1, N]




