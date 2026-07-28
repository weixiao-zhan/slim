# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.critic import CriticNeMoTrainer
from slim.backends.nemo.lr_scheduler import NeMoLRScheduler
from slim.backends.nemo.model import CriticModules


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
def test_critic_uses_independent_backbone_and_value_head_lrs(monkeypatch):
    trainer = CriticNeMoTrainer.__new__(CriticNeMoTrainer)
    trainer.args = SimpleNamespace(
        lr_critic=2e-5,
        lr_critic_value_head=3e-4,
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
        "slim.backends.nemo.critic.build_critic_model",
        lambda *_args: CriticModules(backbone=backbone, value_head=value_head),
    )

    def capture_optimizer(_args, _model, _mesh, *, param_groups):
        captured["groups"] = param_groups
        return torch.optim.AdamW(param_groups, lr=1e-6)

    monkeypatch.setattr("slim.backends.nemo.critic.build_optimizer", capture_optimizer)

    trainer._create_model_and_optimizer("/models/qwen3.5")

    head_group, backbone_group = captured["groups"][1], captured["groups"][0]
    assert backbone_group["max_lr"] == 2e-5
    assert head_group["max_lr"] == 3e-4
    assert list(backbone_group["params"]) == list(backbone.parameters())
    assert list(head_group["params"]) == list(value_head.parameters())
