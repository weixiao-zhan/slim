# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from slim.backends.nemo.base import NeMoTrainer


@pytest.mark.unit
def test_debug_rollout_only_train_does_not_touch_trainer_state():
    class PlaceholderTrainer:
        args = SimpleNamespace(debug_rollout_only=True)

        def wake_up(self):
            raise AssertionError("debug rollout-only trainer must remain uninitialized")

    NeMoTrainer.train(PlaceholderTrainer(), rollout_id=0, rollout_data_ref=None)
