# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 construction and packed context-parallel graph patches."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.distributed.blockdiag_cp import (
    configure_cp_varlen,
    cp_blockdiag_sdpa,
    current_blockdiag_cp_state,
)
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.distributed.cp_vision_shard import (
    CpVisionShardingConfig,
    reset_cp_vision_group,
    set_cp_vision_group,
)
from nemo_automodel.components.distributed.parallelizer import (
    PARALLELIZATION_STRATEGIES,
    Qwen3_5ParallelizationStrategy,
    register_parallel_strategy,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention
from nemo_automodel.components.moe.layers import MoE

from ..topology import flat_mesh

MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")
CP_VISION_SHARDING = CpVisionShardingConfig(enabled=True)


def _text_config(config):
    return getattr(config, "text_config", config)


def _is_moe(config) -> bool:
    return getattr(config, "model_type", "") == "qwen3_5_moe"


def validate_config(config, topology) -> None:
    model_type = getattr(config, "model_type", "")
    if model_type not in MODEL_TYPES:
        supported = ", ".join(MODEL_TYPES)
        raise ValueError(
            f"first-party NeMo training does not support model_type={model_type!r}; supported: {supported}"
        )

    if not _is_moe(config):
        if topology.expert_model_parallel_size != 1:
            raise ValueError("dense Qwen3.5 requires expert_model_parallel_size=1")
        return

    num_experts = int(_text_config(config).num_experts)
    if num_experts % topology.expert_model_parallel_size:
        raise ValueError(
            f"num_experts {num_experts} must be divisible by expert_model_parallel_size "
            f"{topology.expert_model_parallel_size}"
        )


def build_backend_config(args):
    return BackendConfig(
        attn="sdpa",
        linear=args.nemo_linear_backend,
        rms_norm=args.nemo_rms_norm_backend,
        rope_fusion=False,
        experts=args.nemo_experts_backend,
        dispatcher=args.nemo_dispatcher,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )


def register_qwen3_5_moe_parallel_strategy() -> None:
    model_name = "Qwen3_5MoeForConditionalGeneration"
    if model_name not in PARALLELIZATION_STRATEGIES:
        register_parallel_strategy(name=model_name)(Qwen3_5ParallelizationStrategy)


def build_model(
    config,
    args,
    checkpoint: str,
    distributed_setup,
    *,
    routing_replay: bool,
):
    if routing_replay and not _is_moe(config):
        raise ValueError("rollout routing replay requires Qwen3.5 MoE")

    register_qwen3_5_moe_parallel_strategy()
    kwargs = {
        "distributed_setup": distributed_setup,
        "backend": build_backend_config(args),
        "attn_implementation": "sdpa",
        "has_packed_sequence": True,
        "trust_remote_code": True,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "num_nextn_predict_layers": 0,
        "text_config": {"router_aux_loss_coef": 0.0},
        "freeze_config": {
            "freeze_vision_tower": args.freeze_vision_tower,
            "freeze_audio_tower": args.freeze_audio_tower,
            "freeze_language_model": args.freeze_language_model,
        },
    }
    if routing_replay:
        kwargs["moe_overrides"] = {"enable_routing_replay": True}
    model = NeMoAutoModelForImageTextToText.from_pretrained(checkpoint, **kwargs)
    install_packed_cp(model, distributed_setup.mesh_context.device_mesh)
    return model


def _blockdiag_attention(self, query, key, value, **attn_kwargs):
    if current_blockdiag_cp_state() is not None:
        if self.backend.attn != "sdpa":
            raise RuntimeError("packed Qwen3.5 requires the internal SDPA model dispatch")
        return cp_blockdiag_sdpa(query, key, value, **attn_kwargs)
    return self._slim_base_attn_func(query, key, value, **attn_kwargs)


def _install_attention_dispatch(model: torch.nn.Module) -> None:
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


def _vision_sync_group(device_mesh):
    """Return the FSDP-shard group that aligns multimodal forward branches."""
    return flat_mesh(device_mesh, "dp_shard_cp").get_group()


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


def _install_synchronized_vision(model: torch.nn.Module, sync_group, cp_group) -> None:
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
            group=sync_group,
        )
        if not has_images and not has_videos:
            return inputs_embeds

        dummy_pixels = dummy_grid = None

        def visual_inputs(local_pixels, local_grid):
            nonlocal dummy_pixels, dummy_grid
            if local_pixels is not None:
                return local_pixels, local_grid, True
            if dummy_pixels is None:
                dummy_pixels, dummy_grid = _dummy_visual_inputs(module, input_ids.device)
            return dummy_pixels, dummy_grid, False

        token = set_cp_vision_group(cp_group, config=CP_VISION_SHARDING)
        try:
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
        finally:
            reset_cp_vision_group(token)

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


def install_packed_cp(model: torch.nn.Module, device_mesh) -> None:
    """Install the packed training path missing from Qwen3.5 AutoModel."""
    backend = getattr(model, "backend", None)
    if getattr(backend, "attn", None) != "sdpa":
        raise RuntimeError("packed Qwen3.5 requires the internal SDPA model dispatch")

    configure_cp_varlen(attn_backend="flash", kv_exchange="halo")
    cp_mesh = device_mesh["cp"]
    vision_sync_group = _vision_sync_group(device_mesh)
    model.cp_mesh = cp_mesh
    if not _is_moe(model.config):
        _install_attention_dispatch(model)
    if cp_mesh.size() == 1:
        _install_gdn_dispatch(model, cp_mesh)
    _install_moe_cp(model, cp_mesh)
    _install_synchronized_vision(model, vision_sync_group, cp_mesh.get_group())
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
