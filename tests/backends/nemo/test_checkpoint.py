# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.checkpoint import (
    _initialize_missing_adamw_state,
    _save_consolidated_processor,
    build_checkpointer,
    finalize_load,
    is_hf_checkpoint,
    resolve_checkpoint_dir,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_resolve_checkpoint_dir_uses_tracker_or_explicit_step(tmp_path):
    first = tmp_path / "iter_0000003"
    second = tmp_path / "iter_0000007"
    first.mkdir()
    second.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")

    assert resolve_checkpoint_dir(str(tmp_path), SimpleNamespace(ckpt_step=None)) == second
    assert resolve_checkpoint_dir(str(tmp_path), SimpleNamespace(ckpt_step=3)) == first
    assert resolve_checkpoint_dir(str(first), SimpleNamespace(ckpt_step=None)) == first


@pytest.mark.unit
def test_hf_checkpoint_is_not_treated_as_training_resume(tmp_path):
    (tmp_path / "config.json").write_text("{}")

    assert is_hf_checkpoint(tmp_path)
    assert resolve_checkpoint_dir(str(tmp_path), SimpleNamespace(ckpt_step=None)) is None


@pytest.mark.unit
def test_fresh_scheduler_resumes_from_optimizer_global_step(monkeypatch, tmp_path):
    trainer = SimpleNamespace(
        role="critic",
        global_step=0,
        lr_scheduler=SimpleNamespace(last_epoch=0),
        args=SimpleNamespace(
            no_load_rng=True,
            no_load_lr_scheduler=True,
            lr_scheduler_start_step=None,
            start_rollout_id=0,
        ),
    )
    payload = {
        "checkpoint_dir": tmp_path,
        "metadata": {"global_step": 12, "next_rollout_id": 4},
        "iteration": 4,
    }
    monkeypatch.setattr("slim.backends.nemo.checkpoint.dist.barrier", lambda: None)
    monkeypatch.setattr("slim.backends.nemo.checkpoint.torch.cuda.is_available", lambda: False)

    finalize_load(trainer, payload)

    assert trainer.global_step == 12
    assert trainer.lr_scheduler.last_epoch == 12
    assert trainer.args.start_rollout_id == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "model_save_format", "save_consolidated"),
    [
        ("actor", "safetensors", "every"),
        ("critic", "torch_save", "false"),
    ],
)
def test_build_checkpointer_uses_role_specific_automodel_config(
    monkeypatch,
    tmp_path,
    role,
    model_save_format,
    save_consolidated,
):
    captured = {}

    def fake_build(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return object()

    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

    monkeypatch.setattr(CheckpointingConfig, "build", fake_build)
    trainer = SimpleNamespace(
        role=role,
        dp_cp_rank=5,
        moe_mesh=object(),
        mesh_context=SimpleNamespace(process_group=object()),
        _checkpoint_save_dir=str(tmp_path),
        _checkpoint_load_dir=None,
        args=SimpleNamespace(
            hf_checkpoint="/models/qwen",
            checkpoint_save_consolidated="every",
            checkpoint_cpu_offload=True,
            async_save=True,
        ),
    )

    build_checkpointer(trainer)

    config = captured["config"]
    assert config.model_save_format.value == model_save_format
    assert config.save_consolidated.value == save_consolidated
    assert config.cpu_offload is True
    assert config.is_async is True
    assert captured["kwargs"]["dp_rank"] == 5
    assert captured["kwargs"]["tp_rank"] == 0
    assert captured["kwargs"]["pp_rank"] == 0


@pytest.mark.unit
def test_checkpoint_initializes_adamw_state_for_parameters_without_gradients():
    used = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.AdamW([used, unused])
    used.sum().backward()
    optimizer.step()

    assert optimizer.state[used]
    assert not optimizer.state[unused]

    _initialize_missing_adamw_state(optimizer)

    assert set(optimizer.state[unused]) == {"step", "exp_avg", "exp_avg_sq"}
    assert optimizer.state[unused]["step"].item() == 0
    torch.testing.assert_close(optimizer.state[unused]["exp_avg"], torch.zeros_like(unused))
    torch.testing.assert_close(optimizer.state[unused]["exp_avg_sq"], torch.zeros_like(unused))


@pytest.mark.unit
def test_checkpoint_does_not_reallocate_existing_adamw_state(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()

    def fail_zeros_like(*args, **kwargs):
        raise AssertionError("existing AdamW state must not be reallocated")

    monkeypatch.setattr("slim.backends.nemo.checkpoint.torch.zeros_like", fail_zeros_like)

    _initialize_missing_adamw_state(optimizer)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "is_final", "expected"),
    [
        ("every", False, True),
        ("final", False, False),
        ("final", True, True),
        ("false", True, False),
    ],
)
def test_consolidated_actor_checkpoint_includes_processor(monkeypatch, tmp_path, mode, is_final, expected):
    saved_paths = []
    trainer = SimpleNamespace(
        role="actor",
        processor=SimpleNamespace(save_pretrained=lambda path: saved_paths.append(path)),
        args=SimpleNamespace(checkpoint_save_consolidated=mode),
    )
    monkeypatch.setattr("slim.backends.nemo.checkpoint.dist.get_rank", lambda: 0)
    monkeypatch.setattr("slim.backends.nemo.checkpoint.dist.barrier", lambda: None)

    _save_consolidated_processor(trainer, tmp_path, is_final=is_final)

    expected_paths = [tmp_path / "model" / "consolidated"] if expected else []
    assert saved_paths == expected_paths
