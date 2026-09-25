# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoModel optional dependencies and packed Qwen3.5 context parallelism."""

import py_compile
from pathlib import Path

from ._utils import patch_file


def patch_automodel_optional_transformer_engine(parallelizer: Path) -> bool:
    """Keep AutoModel's native CP path usable without TE attention."""
    import_anchor = "    from transformer_engine.pytorch.attention import DotProductAttention\n"
    optional_import = """    try:
        from transformer_engine.pytorch.attention import DotProductAttention
    except ModuleNotFoundError as error:
        if not error.name or not error.name.startswith("transformer_engine"):
            raise
        DotProductAttention = ()
"""
    return patch_file(
        parallelizer,
        [(import_anchor, optional_import)],
        log_reason="AutoModel optional Transformer Engine attention",
    )


def patch_automodel_blockdiag_cp1(batch: Path, exchange: Path) -> None:
    """Use AutoModel's block-diagonal batch context at every CP degree."""
    cp1_noop = """    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend, sdpa_kernel

    world = cp_mesh.size()
    if world <= 1:
        primary = batch.get("inputs_embeds", batch.get("input_ids"))
        layout = None
        if primary is not None:
            layout = ShardLayout(original_seq_len=primary.shape[1], padded_seq_len=primary.shape[1])
        return nullcontext, batch, layout

    rank = cp_mesh.get_local_rank()
"""
    unified_context = """    from torch.nn.attention import SDPBackend, sdpa_kernel

    world = cp_mesh.size()
    rank = cp_mesh.get_local_rank()
"""
    group_world = "    _group_world = torch.distributed.get_world_size(group)\n"
    singleton_group_world = (
        "    _group_world = world if world == 1 else torch.distributed.get_world_size(group)\n"
    )
    patch_file(
        batch,
        [
            (cp1_noop, unified_context),
            (group_world, singleton_group_world),
        ],
        log_reason="AutoModel block-diagonal CP1 batch context",
    )

    gather_forward = """        ctx.world = world
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)
"""
    singleton_gather_forward = """        ctx.world = world
        x = x.contiguous()
        if world == 1:
            return x
        gathered = [torch.empty_like(x) for _ in range(world)]
        torch.distributed.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)
"""
    gather_backward = """        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(local, chunks, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return local, None, None
"""
    singleton_gather_backward = """        if ctx.world == 1:
            return grad_out, None, None
        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        torch.distributed.reduce_scatter(local, chunks, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return local, None, None
"""
    patch_file(
        exchange,
        [
            (gather_forward, singleton_gather_forward),
            (gather_backward, singleton_gather_backward),
        ],
        log_reason="AutoModel singleton block-diagonal K/V gather",
    )


def patch_qwen3_5_packed_cp(automodel_dir: Path) -> tuple[Path, ...]:
    """Use native packed attention and contiguous primary shards at every CP degree."""
    components = automodel_dir / "components"
    sharder = components / "distributed" / "context_parallel" / "sharder.py"
    gdn = components / "models" / "qwen3_5_moe" / "cp_linear_attn.py"
    dense = components / "models" / "qwen3_5" / "model.py"
    moe = components / "models" / "qwen3_5_moe" / "model.py"

    patch_file(
        sharder,
        [
            (
                """        cp_mesh: The context-parallel (sub)mesh; None or size <= 1 is an identity.
        tensor: Full-length sequence tensor, e.g. ``inputs_embeds`` ``[B, S, H]``
            or ``per_layer_inputs`` ``[B, S, L, H]`` (seq axis 1).
        seq_dim: The sequence axis of ``tensor``.
        pad_value: Fill for the CP-padding slots appended on ``seq_dim``.
        pad_multiple:""",
                """        cp_mesh: The context-parallel (sub)mesh. None is an identity; size one
            pads only inside a packed block-diagonal context.
        tensor: Full-length sequence tensor, e.g. ``inputs_embeds`` ``[B, S, H]``
            or ``per_layer_inputs`` ``[B, S, L, H]`` (seq axis 1).
        seq_dim: The sequence axis of ``tensor``.
        pad_value: Fill for the CP-padding slots appended on ``seq_dim``.
        pad_multiple:""",
            ),
            (
                """    seq_len = tensor.shape[seq_dim]
    if cp_mesh is None or cp_mesh.size() <= 1:
        return tensor, torch.arange(seq_len, device=tensor.device, dtype=torch.long), seq_len
    cp_divisor = cp_mesh.size() * max(int(pad_multiple or 1), 2)
""",
                """    from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

    seq_len = tensor.shape[seq_dim]
    if cp_mesh is None or (cp_mesh.size() <= 1 and current_blockdiag_cp_state() is None):
        return tensor, torch.arange(seq_len, device=tensor.device, dtype=torch.long), seq_len
    cp_divisor = cp_mesh.size() * max(int(pad_multiple or 1), 2)
""",
            ),
        ],
        log_reason="AutoModel packed CP1 primary padding",
    )
    patch_file(
        gdn,
        [
            (
                """        # Fast path: no CP → run HF forward with fp32-safe gate computation.
        if self._cp_mesh is None or self._cp_mesh.size() <= 1:
""",
                """        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        blockdiag_state = current_blockdiag_cp_state()
        if blockdiag_state is None and (self._cp_mesh is None or self._cp_mesh.size() <= 1):
""",
            ),
            (
                """        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        return self._forward_with_cp(
            hidden_states,
            position_ids=position_ids,
            seq_index=seq_index,
            blockdiag_state=current_blockdiag_cp_state(),
        )
""",
                """        if blockdiag_state is not None and cache_params is not None:
            raise ValueError("Packed Qwen3.5 training does not support a Gated DeltaNet cache")
        return self._forward_with_cp(
            hidden_states,
            position_ids=position_ids,
            seq_index=seq_index,
            blockdiag_state=blockdiag_state,
        )
""",
            ),
        ],
        log_reason="AutoModel Qwen3.5 packed Gated DeltaNet dispatch including CP1",
    )

    gdn_import = "from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet\n"
    patch_file(
        dense,
        [
            (
                gdn_import,
                gdn_import + "from nemo_automodel.components.models.qwen3_5_moe.model import _Qwen3_5MoeAttention\n",
            ),
            (
                "    shard_batch_aux_only,\n    shard_sequence_for_cp_round_robin,\n",
                "    shard_batch_aux_only,\n    shard_sequence_for_cp_contiguous,\n"
                "    shard_sequence_for_cp_round_robin,\n",
            ),
            (
                """        if self.layer_type == "linear_attention":
            self.linear_attn = CPAwareGatedDeltaNet(config, layer_idx)
""",
                """        if self.layer_type == "linear_attention":
            self.linear_attn = CPAwareGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = _Qwen3_5MoeAttention(config, layer_idx, backend)
""",
            ),
            (
                """        # Context-parallel: embed + vision-splice the full sequence, then keep this
        # rank's round-robin chunk pair (aux streams + mRoPE aligned by
        # shard_batch_aux_only). The local shard matches the old dispatch-level
        # pre-embed and stays differentiable (gradients reach embeddings/vision).
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if (
            cp_size > 1
""",
                """        # Embed and splice before selecting the shard aligned with the auxiliary streams.
        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        blockdiag_state = current_blockdiag_cp_state()
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if (
            (cp_size > 1 or blockdiag_state is not None)
""",
            ),
            (
                """            inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            input_ids = None
            pixel_values = None
            pixel_values_videos = None
""",
                """            if blockdiag_state is not None:
                inputs_embeds, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, inputs_embeds, seq_dim=1)
            else:
                inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            input_ids = None
            pixel_values = None
            pixel_values_videos = None
            image_grid_thw = None
            video_grid_thw = None
            mm_token_type_ids = None
""",
            ),
        ],
        log_reason="AutoModel Qwen3.5 dense native packed attention and primary sharding",
    )
    patch_file(
        moe,
        [
            (
                """        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if cp_size > 1 and inputs_embeds is None and input_ids is not None and not torch.is_floating_point(input_ids):
""",
                """        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        blockdiag_state = current_blockdiag_cp_state()
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if (
            (cp_size > 1 or blockdiag_state is not None)
            and inputs_embeds is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
        ):
""",
            ),
            (
                """                from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

                if current_blockdiag_cp_state() is not None:
""",
                """                if blockdiag_state is not None:
""",
            ),
            (
                """                for media_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                    kwargs.pop(media_key, None)
""",
                """                for media_key in (
                    "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"
                ):
                    kwargs.pop(media_key, None)
""",
            ),
        ],
        log_reason="AutoModel Qwen3.5 MoE native packed primary sharding including CP1",
    )
    return sharder, gdn, dense, moe


def apply() -> None:
    import nemo_automodel

    automodel_dir = Path(nemo_automodel.__file__).resolve().parent
    parallelizer = automodel_dir / "components" / "moe" / "parallelizer.py"
    blockdiag_dir = automodel_dir / "components" / "distributed" / "blockdiag_cp"
    batch, exchange = blockdiag_dir / "batch.py", blockdiag_dir / "exchange.py"
    patch_automodel_optional_transformer_engine(parallelizer)
    patch_automodel_blockdiag_cp1(batch, exchange)
    model_files = patch_qwen3_5_packed_cp(automodel_dir)
    for path in (parallelizer, batch, exchange, *model_files):
        py_compile.compile(str(path), doraise=True)
