# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from slim.backends.nemo.routing_replay import replay_router_targets


NUM_GPUS = 0


@pytest.fixture(autouse=True)
def clear_router_replay_registry():
    from nemo_automodel.components.moe.router_replay import RouterReplay

    RouterReplay.clear_registry()
    yield
    RouterReplay.clear_registry()


@pytest.mark.unit
def test_replay_router_targets_replays_policy_layers_and_clears_targets():
    from nemo_automodel.components.moe.router_replay import RouterReplay, RouterReplayMode

    handles = [RouterReplay(), RouterReplay()]
    routed_experts = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

    with replay_router_targets(routed_experts):
        assert [handle.mode for handle in handles] == [RouterReplayMode.REPLAY, RouterReplayMode.REPLAY]
        torch.testing.assert_close(handles[0].target_indices, torch.tensor([[1, 2], [5, 6]]))
        torch.testing.assert_close(handles[1].target_indices, torch.tensor([[3, 4], [7, 8]]))
        assert all(handle.target_indices.is_contiguous() for handle in handles)

    assert [handle.mode for handle in handles] == [None, None]
    assert [handle.target_indices for handle in handles] == [None, None]


@pytest.mark.unit
def test_replay_router_targets_rejects_extra_router_layers():
    from nemo_automodel.components.moe.router_replay import RouterReplay

    RouterReplay()
    RouterReplay()
    routed_experts = torch.zeros(1, 4, 3, 2, dtype=torch.int32)

    with pytest.raises(ValueError, match="3 layers.*2 routers"):
        with replay_router_targets(routed_experts):
            pass
