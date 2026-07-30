# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.critic import CriticNeMoTrainer
from slim.backends.nemo.lr_scheduler import NeMoLRScheduler


NUM_GPUS = 0


@pytest.mark.unit
def test_scheduler_uses_parameter_group_max_lrs_from_step_zero():
    head = torch.nn.Parameter(torch.ones(1))
    backbone = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [head], "max_lr": 3e-4},
            {"params": [backbone], "max_lr": 2e-5},
        ],
        lr=1e-6,
    )
    scheduler = NeMoLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=10,
        lr_decay_style="constant",
    )

    assert scheduler.get_last_lr() == [3e-4, 2e-5]


@pytest.mark.unit
def test_scheduler_starts_parameter_groups_at_independent_steps():
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameters[0]], "max_lr": 3e-4, "start_step": 0},
            {"params": [parameters[1]], "max_lr": 2e-5, "start_step": 1},
            {"params": [parameters[2]], "max_lr": 4e-6, "start_step": 2},
        ],
        lr=1e-6,
    )
    scheduler = NeMoLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=3,
        lr_decay_style="constant",
    )

    assert scheduler.get_last_lr() == [3e-4, 0.0, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 0.0]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [3e-4, 2e-5, 4e-6]


@pytest.mark.unit
def test_scheduler_rejects_warmup_that_overlaps_wsd_decay():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "name": "actor", "start_step": 5}],
        lr=1e-6,
    )

    with pytest.raises(ValueError, match="actor warmup ends at step 7"):
        NeMoLRScheduler(
            optimizer,
            init_lr=0.0,
            max_lr=1e-6,
            min_lr=0.0,
            lr_warmup_steps=2,
            lr_decay_steps=8,
            lr_decay_style="WSD",
            wsd_decay_steps=2,
            lr_wsd_decay_style="linear",
        )


@pytest.mark.unit
def test_critic_uses_independent_backbone_and_value_head_lrs(monkeypatch):
    trainer = CriticNeMoTrainer.__new__(CriticNeMoTrainer)
    trainer.args = SimpleNamespace(
        lr_critic=2e-5,
        lr_critic_value_head=3e-4,
        lr_critic_start_step=1,
        lr_critic_value_head_start_step=0,
        rollout_batch_size=8,
        n_samples_per_prompt=1,
        global_batch_size=8,
    )
    trainer.distributed_setup = object()
    trainer.device_mesh = object()
    trainer.hf_config = object()
    backbone = torch.nn.Linear(4, 4, bias=False)
    value_head = torch.nn.Linear(4, 1, bias=False)
    captured = {}

    monkeypatch.setattr(
        "slim.backends.nemo.critic.build_model",
        lambda *_args, **_kwargs: backbone,
    )
    monkeypatch.setattr(
        "slim.backends.nemo.critic.build_value_head",
        lambda *_args: value_head,
    )

    def capture_optimizer(_args, _mesh, *, param_groups):
        captured["groups"] = param_groups
        return torch.optim.AdamW(param_groups, lr=1e-6)

    monkeypatch.setattr("slim.backends.nemo.critic.build_optimizer", capture_optimizer)

    trainer._create_model_and_optimizer("/models/qwen3.5")

    head_group, backbone_group = captured["groups"][1], captured["groups"][0]
    assert backbone_group["max_lr"] == 2e-5
    assert backbone_group["start_step"] == 1
    assert head_group["max_lr"] == 3e-4
    assert head_group["start_step"] == 0
    assert list(backbone_group["params"]) == list(backbone.parameters())
    assert list(head_group["params"]) == list(value_head.parameters())
