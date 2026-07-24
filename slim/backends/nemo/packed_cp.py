# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified packed context parallelism for dense and MoE Qwen3.5."""

from __future__ import annotations

import contextlib
from functools import partial
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist


def _blockdiag_attention(self, query, key, value, **attn_kwargs):
    from nemo_automodel.components.distributed.blockdiag_cp import (
        cp_blockdiag_sdpa,
        current_blockdiag_cp_state,
    )

    if current_blockdiag_cp_state() is not None:
        if self.backend.attn != "sdpa":
            raise RuntimeError("packed Qwen3.5 requires the internal SDPA model dispatch")
        return cp_blockdiag_sdpa(query, key, value, **attn_kwargs)
    return self._slim_base_attn_func(query, key, value, **attn_kwargs)


def _install_attention_dispatch(model: torch.nn.Module) -> None:
    from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
    from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention

    seen = set()
    for module in model.modules():
        attention = unwrap_checkpoint_wrapper(module)
        if not isinstance(attention, Qwen3NextAttention) or id(attention) in seen:
            continue
        seen.add(id(attention))
        if getattr(attention, "_slim_packed_cp_dispatch", False):
            continue
        if attention.backend.attn != "sdpa":
            raise RuntimeError("packed Qwen3.5 requires the internal SDPA model dispatch")
        attention._slim_base_attn_func = attention.attn_func
        attention.attn_func = MethodType(_blockdiag_attention, attention)
        attention._slim_packed_cp_dispatch = True


def _blockdiag_gdn(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    position_ids=None,
    qkv_format=None,
    cu_seqlens=None,
    indices=None,
    seq_index=None,
):
    from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

    blockdiag_state = current_blockdiag_cp_state()
    if blockdiag_state is None:
        return self._slim_base_gdn_forward(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            indices=indices,
            seq_index=seq_index,
        )
    if cache_params is not None:
        raise ValueError("packed Qwen3.5 training does not support a Gated DeltaNet cache")
    return self._forward_with_cp(
        hidden_states,
        position_ids=position_ids,
        seq_index=seq_index,
        blockdiag_state=blockdiag_state,
    )


def _install_gdn_dispatch(model: torch.nn.Module, cp_mesh) -> None:
    from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

    seen = set()
    for module in model.modules():
        gdn = unwrap_checkpoint_wrapper(module)
        if not isinstance(gdn, CPAwareGatedDeltaNet) or id(gdn) in seen:
            continue
        seen.add(id(gdn))
        gdn._cp_mesh = cp_mesh
        if getattr(gdn, "_slim_packed_cp_dispatch", False):
            continue
        gdn._slim_base_gdn_forward = gdn.forward
        gdn.forward = MethodType(_blockdiag_gdn, gdn)
        gdn._slim_packed_cp_dispatch = True


def _install_moe_cp(model: torch.nn.Module, cp_mesh) -> None:
    from nemo_automodel.components.moe.layers import MoE

    for module in model.modules():
        if isinstance(module, MoE):
            module.cp_mesh = cp_mesh


def _global_media_presence(
    *,
    pixel_values: torch.Tensor | None,
    pixel_values_videos: torch.Tensor | None,
    device: torch.device,
    group,
) -> tuple[bool, bool]:
    flags = torch.tensor(
        [pixel_values is not None, pixel_values_videos is not None],
        dtype=torch.int32,
        device=device,
    )
    if dist.get_world_size(group) > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    return bool(flags[0].item()), bool(flags[1].item())


def _dummy_visual_inputs(model: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    config = model.config.vision_config
    merge_size = int(config.spatial_merge_size)
    patch_dim = (
        int(config.in_channels)
        * int(config.temporal_patch_size)
        * int(config.patch_size)
        * int(config.patch_size)
    )
    grid_thw = torch.tensor([[1, merge_size, merge_size]], dtype=torch.long, device=device)
    pixel_values = torch.zeros(
        merge_size * merge_size,
        patch_dim,
        dtype=torch.float32,
        device=device,
    )
    return pixel_values, grid_thw


def _install_synchronized_vision(model: torch.nn.Module, group) -> None:
    if getattr(model, "_slim_synchronized_vision", False):
        return

    def embed_and_splice(
        module: torch.nn.Module,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = module.get_input_embeddings()(input_ids)
        has_images, has_videos = _global_media_presence(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            device=input_ids.device,
            group=group,
        )
        if not has_images and not has_videos:
            return inputs_embeds

        from nemo_automodel.components.distributed.context_parallel.utils import (
            cp_dispatcher_suspended,
        )

        dummy_pixels = dummy_grid = None

        def visual_inputs(local_pixels, local_grid):
            nonlocal dummy_pixels, dummy_grid
            if local_pixels is not None:
                return local_pixels, local_grid, True
            if dummy_pixels is None:
                dummy_pixels, dummy_grid = _dummy_visual_inputs(module, input_ids.device)
            return dummy_pixels, dummy_grid, False

        with cp_dispatcher_suspended(module.cp_mesh):
            if has_images:
                image_pixels, image_grid, is_local = visual_inputs(pixel_values, image_grid_thw)
                if hasattr(module.model.visual, "rotary_pos_emb"):
                    module.model.visual.rotary_pos_emb.to(image_pixels.device)
                image_embeds = module._encode_vision_for_cp(
                    image_pixels,
                    image_grid,
                    is_video=False,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                if is_local:
                    image_mask, _ = module.model.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        image_features=image_embeds,
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                elif image_embeds.requires_grad:
                    inputs_embeds = inputs_embeds + image_embeds.sum() * 0.0

            if has_videos:
                video_pixels, video_grid, is_local = visual_inputs(pixel_values_videos, video_grid_thw)
                if hasattr(module.model.visual, "rotary_pos_emb"):
                    module.model.visual.rotary_pos_emb.to(video_pixels.device)
                video_embeds = module._encode_vision_for_cp(
                    video_pixels,
                    video_grid,
                    is_video=True,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                if is_local:
                    _, video_mask = module.model.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=inputs_embeds,
                        video_features=video_embeds,
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                elif video_embeds.requires_grad:
                    inputs_embeds = inputs_embeds + video_embeds.sum() * 0.0

        return inputs_embeds

    model._embed_and_splice_for_cp = MethodType(embed_and_splice, model)
    model._slim_synchronized_vision = True


def _pad_sequence(tensor: torch.Tensor, *, seq_dim: int, pad_len: int, fill: float | int) -> torch.Tensor:
    if pad_len == 0:
        return tensor
    shape = list(tensor.shape)
    shape[seq_dim] = pad_len
    padding = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=seq_dim)


def _install_primary_shard(model: torch.nn.Module, cp_mesh) -> None:
    if getattr(model, "_slim_packed_cp_primary_shard", False):
        return

    from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

    prepare_inputs_embeds = getattr(model, "_embed_and_splice_for_cp", None)
    if prepare_inputs_embeds is None:
        raise TypeError("Qwen3.5 AutoModel must expose _embed_and_splice_for_cp")

    def shard_primary(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if current_blockdiag_cp_state() is None:
            return None

        if kwargs.get("inputs_embeds") is not None:
            return None

        input_ids = kwargs.get("input_ids")
        if input_ids is None or torch.is_floating_point(input_ids):
            return None

        inputs_embeds = module._embed_and_splice_for_cp(
            input_ids,
            pixel_values=kwargs.get("pixel_values"),
            pixel_values_videos=kwargs.get("pixel_values_videos"),
            image_grid_thw=kwargs.get("image_grid_thw"),
            video_grid_thw=kwargs.get("video_grid_thw"),
        )
        cp_size = cp_mesh.size()
        pad_len = (-inputs_embeds.shape[1]) % (2 * cp_size)
        inputs_embeds = _pad_sequence(inputs_embeds, seq_dim=1, pad_len=pad_len, fill=0)
        local_len = inputs_embeds.shape[1] // cp_size
        rank = cp_mesh.get_local_rank()
        inputs_embeds = inputs_embeds[:, rank * local_len : (rank + 1) * local_len].contiguous()

        kwargs = dict(kwargs)
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = inputs_embeds
        for media_key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        ):
            kwargs.pop(media_key, None)
        return args, kwargs

    model.register_forward_pre_hook(shard_primary, with_kwargs=True)
    model._slim_packed_cp_primary_shard = True


def install_qwen3_5_packed_cp(model: torch.nn.Module, device_mesh) -> None:
    """Install one packed attention, GDN, and primary-token path for Qwen3.5."""
    backend = getattr(model, "backend", None)
    if getattr(backend, "attn", None) != "sdpa":
        raise RuntimeError("packed Qwen3.5 requires the internal SDPA model dispatch")

    from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh

    cp_mesh = device_mesh["cp"]
    vision_sync_group = get_fsdp_dp_mesh(device_mesh).get_group()
    model.cp_mesh = cp_mesh
    _install_attention_dispatch(model)
    _install_gdn_dispatch(model, cp_mesh)
    _install_moe_cp(model, cp_mesh)
    _install_synchronized_vision(model, vision_sync_group)
    _install_primary_shard(model, cp_mesh)


def build_packed_position_ids(model: torch.nn.Module, pack: dict, model_batch: dict) -> torch.Tensor:
    """Build document-local text or multimodal positions for one physical pack."""
    device = model_batch["input_ids"].device
    if not pack.get("multimodal_inputs"):
        return pack["position_ids"].to(device=device, non_blocking=True).unsqueeze(0)

    image_grid_hws = model_batch.get("image_grid_hws")
    if model_batch.get("image_grid_thw") is None and image_grid_hws is not None:
        if image_grid_hws.shape[-1] == 2:
            temporal = torch.ones(
                image_grid_hws.shape[0],
                1,
                dtype=image_grid_hws.dtype,
                device=image_grid_hws.device,
            )
            model_batch["image_grid_thw"] = torch.cat((temporal, image_grid_hws), dim=-1)
        else:
            model_batch["image_grid_thw"] = image_grid_hws
        model_batch.pop("image_grid_hws")

    prepare = getattr(model, "prepare_model_inputs_for_cp", None)
    if prepare is None:
        raise TypeError("Qwen3.5 AutoModel must expose prepare_model_inputs_for_cp")

    boundaries = pack["cu_seqlens"].tolist()
    counts = pack.get("multimodal_num_items") or {}
    offsets = {}
    for name in ("image_grid_thw", "video_grid_thw"):
        item_counts = counts.get(name, counts.get("image_grid_hws", []) if name == "image_grid_thw" else [])
        cursor = [0]
        for count in item_counts:
            cursor.append(cursor[-1] + count)
        offsets[name] = cursor

    pieces = []
    for document_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        document_batch = {"input_ids": model_batch["input_ids"][:, start:end]}
        for name, cursor in offsets.items():
            tensor = model_batch.get(name)
            if tensor is None or len(cursor) <= document_index + 1:
                continue
            item_start, item_end = cursor[document_index : document_index + 2]
            if item_end > item_start:
                document_batch[name] = tensor[item_start:item_end]
        prepared = prepare(document_batch, num_chunks=1)
        pieces.append(prepared["position_ids"])
    return torch.cat(pieces, dim=-1)


def make_packed_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    shard_primary: bool = False,
):
    """Apply the same padded block-diagonal state at every CP degree."""
    from nemo_automodel.components.distributed.blockdiag_cp import make_cp_blockdiag_batch_and_ctx

    if cp_mesh.size() > 1:
        return make_cp_blockdiag_batch_and_ctx(
            cp_mesh,
            tp_mesh,
            batch,
            loss_mask=loss_mask,
            padding_token_id=padding_token_id,
            shard_primary=shard_primary,
        )

    from torch.nn.attention import SDPBackend, sdpa_kernel

    from nemo_automodel.components.distributed.blockdiag_cp import kernels
    from nemo_automodel.components.distributed.blockdiag_cp import state as state_module
    from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout

    del tp_mesh, padding_token_id
    primary_key = "inputs_embeds" if "inputs_embeds" in batch else "input_ids"
    primary = batch[primary_key]
    batch_size, seq_len = primary.shape[:2]
    if batch_size != 1:
        raise ValueError(f"packed Qwen3.5 requires local_batch_size=1, got {batch_size}")
    if shard_primary:
        raise ValueError("packed Qwen3.5 primary tokens are sharded after multimodal embedding")

    doc_ids = batch["_packed_seq_ids"].to(device=primary.device, dtype=torch.long)
    batch.pop("attention_mask", None)
    batch["padding_mask"] = doc_ids.eq(0)
    if loss_mask is not None:
        batch["loss_mask"] = loss_mask

    pad_len = (-seq_len) % 2
    doc_ids = _pad_sequence(doc_ids, seq_dim=1, pad_len=pad_len, fill=0)
    fills = {
        "labels": -100,
        "_packed_seq_ids": 0,
        "loss_mask": 0,
        "padding_mask": True,
        "visual_pos_masks": False,
    }
    for key in (
        "labels",
        "position_ids",
        "_packed_seq_ids",
        "loss_mask",
        "padding_mask",
        "visual_pos_masks",
    ):
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        seq_dim = 2 if key == "position_ids" and tensor.dim() == 3 else 1
        batch[key] = _pad_sequence(tensor, seq_dim=seq_dim, pad_len=pad_len, fill=fills.get(key, 0))

    flat_doc_ids = doc_ids.reshape(-1)
    boundaries = torch.where(flat_doc_ids[1:] != flat_doc_ids[:-1])[0].to(torch.long) + 1
    packed_cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=primary.device),
            boundaries,
            torch.tensor([flat_doc_ids.numel()], dtype=torch.long, device=primary.device),
        )
    )
    group = cp_mesh.get_group()
    runtime_config = state_module.cp_varlen_runtime_config()
    step_state = {
        "group": group,
        "doc_ids": doc_ids,
        "packed_cu_seqlens": packed_cu_seqlens,
        "packed_cu_seqlens_cpu": packed_cu_seqlens.detach().cpu(),
        "row_offset": 0,
        "seq_dim": 2,
        "attn_backend": runtime_config["attn_backend"],
        "kv_exchange": runtime_config["kv_exchange"],
        "varlen_meta": kernels.precompute_blockdiag_varlen_meta(
            doc_ids,
            row_offset=0,
            local_len=doc_ids.shape[1],
            device=primary.device,
        ),
    }
    step_state["model_state"] = state_module.BlockdiagCpModelState(
        group=group,
        packed_cu_seqlens=packed_cu_seqlens,
        packed_cu_seqlens_cpu=step_state["packed_cu_seqlens_cpu"],
    )

    @contextlib.contextmanager
    def context():
        token = state_module._CP_BLOCKDIAG_STATE.set(step_state)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            try:
                yield
            finally:
                state_module._CP_BLOCKDIAG_STATE.reset(token)

    layout = ShardLayout(original_seq_len=seq_len, padded_seq_len=seq_len + pad_len)
    return context, batch, layout


def build_packed_cp_sharder(device_mesh, *, padding_token_id: int):
    """Build the common contiguous block-diagonal sharder."""
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices

    return ContextParallelSharder(
        device_mesh=device_mesh,
        shard_batch=partial(make_packed_cp_batch_and_ctx, shard_primary=False),
        local_token_global_indices=contiguous_local_indices,
        padding_token_id=padding_token_id,
    )
