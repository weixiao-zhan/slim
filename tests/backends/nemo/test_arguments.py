# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from slim.backends.nemo.arguments import NeMoArgs, _field_type, _parse_nemo_cli, validate_args


NUM_GPUS = 0


def _args(**overrides):
    values = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "dp_replicate_size": 1,
        "sequence_parallel": False,
        "actor_num_gpus": 8,
        "world_size": 8,
        "nemo_linear_backend": "torch",
        "nemo_rms_norm_backend": "torch_fp32",
        "nemo_experts_backend": "torch_mm",
        "nemo_dispatcher": "torch",
        "optimizer": "adam",
        "checkpoint_save_consolidated": "final",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_optional_cli_types_resolve_from_postponed_annotations():
    type_hints = get_type_hints(NeMoArgs)
    by_name = {field.name: field for field in fields(NeMoArgs)}

    assert _field_type(by_name["lr_decay_iters"], type_hints) is int
    assert by_name["freeze_vision_tower"].default is True
    assert by_name["freeze_audio_tower"].default is True
    assert by_name["freeze_language_model"].default is False
    assert by_name["defer_fsdp_grad_sync"].default is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (True, True, False)),
        (
            [
                "--freeze-vision-tower",
                "--freeze-audio-tower",
                "--freeze-language-model",
            ],
            (True, True, True),
        ),
        (
            [
                "--no-freeze-vision-tower",
                "--no-freeze-audio-tower",
                "--no-freeze-language-model",
            ],
            (False, False, False),
        ),
    ],
)
def test_freeze_tower_cli_flags(monkeypatch, flags, expected):
    monkeypatch.setattr("sys.argv", ["nemo"] + flags)

    args = _parse_nemo_cli()

    assert (
        args.freeze_vision_tower,
        args.freeze_audio_tower,
        args.freeze_language_model,
    ) == expected


@pytest.mark.unit
def test_validate_args_accepts_cp_ep_topology():
    validate_args(_args(context_parallel_size=2, expert_model_parallel_size=4))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
        ({"pipeline_model_parallel_size": 2}, "pipeline_model_parallel_size"),
        ({"sequence_parallel": True}, "sequence_parallel"),
        ({"context_parallel_size": 3}, "must be divisible"),
        ({"nemo_linear_backend": "te"}, "nemo_linear_backend"),
        ({"nemo_rms_norm_backend": "te"}, "nemo_rms_norm_backend"),
        ({"nemo_experts_backend": "te"}, "nemo_experts_backend"),
        ({"nemo_dispatcher": "deepep"}, "nemo_dispatcher"),
    ],
)
def test_validate_args_rejects_unsupported_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_args(_args(**overrides))
