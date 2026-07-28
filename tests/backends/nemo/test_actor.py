# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from slim.backends.nemo.actor import ActorNeMoTrainer


NUM_GPUS = 0


@pytest.mark.unit
def test_custom_mismatch_metrics_are_preserved(monkeypatch):
    trainer = ActorNeMoTrainer.__new__(ActorNeMoTrainer)
    trainer.args = SimpleNamespace(
        mismatch_correction="custom",
        get_mismatch_metrics=False,
        custom_mismatch_correction_function_path="custom.correction",
    )
    pack = {
        "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
        "edge_lengths": [2, 2],
        "actor_old_log_probs": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "rollout_log_probs": torch.tensor([0.0, 0.1, 0.2, 0.3]),
        "loss_masks": torch.ones(4),
    }

    def correction(**_kwargs):
        return None, None, {"rs_keep": [torch.ones(2), torch.zeros(2)]}

    monkeypatch.setattr("slim.backends.nemo.actor.load_function", lambda _path: correction)

    trainer._prepare_mismatch(pack)

    torch.testing.assert_close(
        pack["_mismatch_metrics"]["rs_keep"],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )
