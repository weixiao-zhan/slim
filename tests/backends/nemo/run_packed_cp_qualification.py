# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed Qwen3.5 packed-context-parallel qualification harness."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from slim.backends.nemo.forward import model_forward, prepare_forward
from slim.backends.nemo.loss import selective_log_probs
from slim.backends.nemo.model import build_policy_model
from slim.backends.nemo.topology import NeMoTopology

MAX_LOG_PROB_DELTA = 0.3
MAX_LOG_PROB_MEAN_DELTA = 0.1
MAX_MEAN_LOSS_ABS_DELTA = 0.1
MAX_MEAN_LOSS_RELATIVE_DELTA = 0.02
MAX_GRADIENT_DELTA = 0.3
MAX_GRADIENT_RELATIVE_L2 = 0.5
MAX_FIRST_LAYER_RELATIVE_L2 = 0.02
EP_MAX_LOG_PROB_DELTA = 0.4
EP_MAX_LOG_PROB_MEAN_DELTA = 0.12
EP_MAX_GRADIENT_RELATIVE_L2 = 0.6


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-parallel-size", type=int, required=True)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _model_args(activation_checkpointing: bool) -> SimpleNamespace:
    return SimpleNamespace(
        activation_checkpointing=activation_checkpointing,
        gradient_checkpointing=False,
        nemo_cpu_offload=False,
        defer_fsdp_grad_sync=False,
        distributed_timeout_minutes=30,
        nemo_linear_backend="torch",
        nemo_rms_norm_backend="torch_fp32",
        nemo_experts_backend="torch_mm",
        nemo_dispatcher="torch",
        freeze_vision_tower=True,
        freeze_audio_tower=True,
        freeze_language_model=False,
    )


def _pack(first_document: list[int]) -> dict:
    second_document = [101, 102, 103, 104, 105, 106]
    tokens = torch.tensor(first_document + second_document, dtype=torch.long)
    first_end = len(first_document)
    return {
        "tokens": tokens,
        "position_ids": torch.cat(
            (
                torch.arange(first_end, dtype=torch.long),
                torch.arange(len(second_document), dtype=torch.long),
            )
        ),
        "loss_masks": torch.ones(tokens.numel() - 2, dtype=torch.int32),
        "advantages": torch.ones(tokens.numel() - 2, dtype=torch.float32),
        "returns": torch.ones(tokens.numel() - 2, dtype=torch.float32),
        "cu_seqlens": torch.tensor([0, first_end, tokens.numel()], dtype=torch.int32),
        "edge_lengths": [first_end - 1, len(second_document) - 1],
        "response_lengths": [first_end - 1, len(second_document) - 1],
        "rewards": torch.ones(2, dtype=torch.float32),
        "raw_reward": [1.0, 1.0],
        "_episode_indices": [0, 1],
    }


def _forward_log_probs(model, device_mesh, pack: dict) -> tuple[torch.Tensor, object]:
    prepared = prepare_forward(model, device_mesh, pack, padding_token_id=0)
    with prepared.context_factory():
        output = model_forward(model, prepared.model_batch)
        local = selective_log_probs(output.logits, prepared.fields["labels"])
    return prepared.gather(local, fill=0), prepared


def _canonical_module_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _gradient_vector(model: torch.nn.Module) -> tuple[str, torch.Tensor]:
    target = "model.language_model.layers.0.linear_attn.norm.weight"
    for name, parameter in model.named_parameters():
        canonical_name = _canonical_module_name(name)
        if parameter.grad is None or canonical_name != target:
            continue
        gradient = parameter.grad
        if isinstance(gradient, torch.distributed.tensor.DTensor):
            gradient = gradient.full_tensor()
        return canonical_name, gradient.detach().float().cpu()
    raise RuntimeError(f"no gradient was produced for {target}")


@contextlib.contextmanager
def _capture_sequence_modules(model: torch.nn.Module):
    from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
    from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention

    outputs: dict[str, torch.Tensor] = {}
    handles = []
    seen = set()

    def capture(name):
        def hook(_module, _args, output):
            if name not in outputs:
                outputs[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        unwrapped = unwrap_checkpoint_wrapper(module)
        if id(unwrapped) in seen or not isinstance(unwrapped, (CPAwareGatedDeltaNet, Qwen3NextAttention)):
            continue
        seen.add(id(unwrapped))
        handles.append(unwrapped.register_forward_hook(capture(_canonical_module_name(name))))
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


def _compare_with_baseline(payload: dict, baseline_path: Path) -> tuple[dict[str, float], list[str]]:
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    expert_parallel_size = payload.get("expert_parallel_size", 1)
    baseline_expert_parallel_size = baseline.get("expert_parallel_size", 1)
    if expert_parallel_size != baseline_expert_parallel_size:
        raise RuntimeError(
            "baseline and candidate use different expert-parallel sizes: "
            f"{baseline_expert_parallel_size} and {expert_parallel_size}"
        )
    if payload["gradient_name"] != baseline["gradient_name"]:
        raise RuntimeError("baseline and candidate gradients come from different parameters")
    log_prob_delta = (payload["log_probs"].float() - baseline["log_probs"].float()).abs()
    gradient_delta = (payload["gradient"].float() - baseline["gradient"].float()).abs()
    metrics = {
        "log_prob_max_delta": log_prob_delta.max().item(),
        "log_prob_mean_delta": log_prob_delta.mean().item(),
        "mean_loss_abs_delta": abs(payload["mean_loss"] - baseline["mean_loss"]),
        "mean_loss_relative_delta": abs(payload["mean_loss"] - baseline["mean_loss"])
        / max(abs(baseline["mean_loss"]), 1e-12),
        "gradient_max_delta": gradient_delta.max().item(),
        "gradient_relative_l2": _relative_l2(payload["gradient"], baseline["gradient"]),
    }

    layer_outputs = payload.get("layer_outputs", {})
    baseline_layer_outputs = baseline.get("layer_outputs", {})
    if layer_outputs.keys() != baseline_layer_outputs.keys():
        raise RuntimeError("baseline and candidate layer captures do not have the same modules")
    if layer_outputs:
        first_layer = next(iter(layer_outputs))
        metrics["first_layer_relative_l2"] = _relative_l2(
            layer_outputs[first_layer],
            baseline_layer_outputs[first_layer],
        )

    failures = []
    limits = {
        "log_prob_max_delta": EP_MAX_LOG_PROB_DELTA if expert_parallel_size > 1 else MAX_LOG_PROB_DELTA,
        "log_prob_mean_delta": (
            EP_MAX_LOG_PROB_MEAN_DELTA if expert_parallel_size > 1 else MAX_LOG_PROB_MEAN_DELTA
        ),
        "mean_loss_abs_delta": MAX_MEAN_LOSS_ABS_DELTA,
        "mean_loss_relative_delta": MAX_MEAN_LOSS_RELATIVE_DELTA,
        "gradient_max_delta": MAX_GRADIENT_DELTA,
        "gradient_relative_l2": (
            EP_MAX_GRADIENT_RELATIVE_L2 if expert_parallel_size > 1 else MAX_GRADIENT_RELATIVE_L2
        ),
        "first_layer_relative_l2": MAX_FIRST_LAYER_RELATIVE_L2,
    }
    for name, limit in limits.items():
        value = metrics.get(name)
        if value is not None and value > limit:
            failures.append(f"{name}={value:.7g} exceeds {limit:.7g}")
    return metrics, failures


def main() -> None:
    cli = _arguments()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    args = _model_args(cli.activation_checkpointing)
    topology = NeMoTopology(
        world_size=world_size,
        context_parallel_size=cli.context_parallel_size,
        expert_model_parallel_size=cli.expert_parallel_size,
    )
    setup = topology.build(args)
    device_mesh = setup.mesh_context.device_mesh
    model = build_policy_model(
        args,
        cli.checkpoint,
        setup,
        routing_replay=False,
    )
    model.train()

    baseline_pack = _pack([11, 12, 13, 14, 15])
    changed_prefix_pack = _pack([21, 22, 23, 24, 25])
    with torch.no_grad():
        with _capture_sequence_modules(model) as local_layer_outputs:
            baseline, baseline_prepared = _forward_log_probs(model, device_mesh, baseline_pack)
        layer_outputs = {
            name: baseline_prepared.gather(output, fill=0).float().cpu()
            for name, output in local_layer_outputs.items()
        }
        changed, _ = _forward_log_probs(model, device_mesh, changed_prefix_pack)
    second_start = baseline_pack["cu_seqlens"][1].item()
    second_end = baseline_pack["cu_seqlens"][2].item()
    isolation_error = (
        baseline[:, second_start : second_end - 1] - changed[:, second_start : second_end - 1]
    ).abs().max()

    from nemo_automodel.components.distributed.blockdiag_cp.state import (
        cp_attn_fire_count,
        reset_cp_attn_fire_count,
    )
    from nemo_automodel.components.training.utils import (
        prepare_for_final_backward,
        prepare_for_grad_accumulation,
    )

    prepare_for_grad_accumulation([model], pp_enabled=False)
    prepare_for_final_backward([model], pp_enabled=False)
    reset_cp_attn_fire_count()
    prepared = prepare_forward(model, device_mesh, baseline_pack, padding_token_id=0)
    with prepared.context_factory():
        output = model_forward(model, prepared.model_batch)
        local = selective_log_probs(output.logits, prepared.fields["labels"])
        valid = prepared.fields["labels"] != -100
        valid_targets = sum(baseline_pack["edge_lengths"])
        local_loss = -(local * valid).sum() / (valid_targets * topology.logical_dp_size)
        (local_loss * world_size).backward()
    full_log_probs = prepared.gather(local.detach(), fill=0).cpu()
    mean_loss = (-full_log_probs.sum() / valid_targets).item()
    gradient_name, gradient = _gradient_vector(model)

    fire_count = cp_attn_fire_count()
    if cli.context_parallel_size > 1 and fire_count == 0:
        raise RuntimeError("block-diagonal CP attention did not execute")
    if not torch.isfinite(full_log_probs).all() or not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite packed CP outputs or gradients")
    if isolation_error.item() > 1e-4:
        raise RuntimeError(f"packed document isolation failed with max error {isolation_error.item()}")

    qualification_failures = []
    if rank == 0:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "log_probs": full_log_probs,
            "mean_loss": mean_loss,
            "gradient_name": gradient_name,
            "gradient": gradient,
            "isolation_error": isolation_error.cpu(),
            "cp_attn_fire_count": fire_count,
            "layer_outputs": layer_outputs,
            "context_parallel_size": cli.context_parallel_size,
            "expert_parallel_size": cli.expert_parallel_size,
            "logical_dp_size": topology.logical_dp_size,
        }
        if cli.baseline_output is not None:
            parity, qualification_failures = _compare_with_baseline(payload, cli.baseline_output)
            payload["parity"] = parity
        torch.save(payload, cli.output)
        print(
            json.dumps(
                {
                    "output": str(cli.output),
                    "gradient_name": gradient_name,
                    "gradient_norm": gradient.norm().item(),
                    "mean_loss": mean_loss,
                    "isolation_error": isolation_error.item(),
                    "cp_attn_fire_count": fire_count,
                    "parity": payload.get("parity"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    failed = torch.tensor(
        bool(qualification_failures),
        dtype=torch.int32,
        device=torch.device("cuda", local_rank),
    )
    dist.broadcast(failed, src=0)
    if failed.item():
        detail = "; ".join(qualification_failures) if rank == 0 else "rank 0 rejected CP parity"
        raise RuntimeError(detail)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
