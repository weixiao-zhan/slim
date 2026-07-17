from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slim.backends.megatron.routing_replay import ModelScopedRouterReplay


pytestmark = pytest.mark.unit


@pytest.fixture
def fake_router_replay(monkeypatch):
    module = ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplayAction(Enum):
        REPLAY_FORWARD = "replay_forward"
        REPLAY_BACKWARD = "replay_backward"

    class RouterReplay:
        global_router_replay_instances = []

        def __init__(self, layer_number, events=None):
            self.layer_number = layer_number
            self.events = events
            self.target_topk_idx = None
            self.replay_backward_list = []
            self.router_replay_action = None
            type(self).global_router_replay_instances.append(self)

        @classmethod
        def clear_global_router_replay_instances(cls):
            cls.global_router_replay_instances.clear()

        def set_target_indices(self, indices):
            self.target_topk_idx = indices
            self.replay_backward_list.append(indices)

        def set_router_replay_action(self, action):
            self.router_replay_action = action
            if self.events is not None:
                self.events.append(action.value)

        def clear_router_replay_action(self):
            self.router_replay_action = None

        def clear_indices(self):
            self.target_topk_idx = None
            self.replay_backward_list.clear()

    module.RouterReplay = RouterReplay
    module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return RouterReplay, RouterReplayAction


class FakeModel:
    def __init__(self, modules):
        self._modules = [self, *modules]

    def modules(self):
        return iter(self._modules)


def _moe_layer(layer_number, replay):
    router = SimpleNamespace(layer_number=layer_number, router_replay=replay)
    return SimpleNamespace(layer_number=layer_number, router=router)


def test_adapter_maps_rollout_axis_by_one_based_global_layer_and_clears_static_list(
    fake_router_replay,
):
    replay_type, action_type = fake_router_replay
    replay_five = replay_type(5)
    replay_two = replay_type(2)
    model = FakeModel([_moe_layer(5, replay_five), _moe_layer(2, replay_two)])

    adapter = ModelScopedRouterReplay([model])

    assert adapter.layer_numbers == (2, 5)
    assert replay_type.global_router_replay_instances == []

    routes = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    adapter.prepare_forward(routes)

    assert torch.equal(replay_two.target_topk_idx, routes[:, 1, :])
    assert torch.equal(replay_five.target_topk_idx, routes[:, 4, :])
    assert replay_two.router_replay_action is action_type.REPLAY_FORWARD
    assert replay_five.router_replay_action is action_type.REPLAY_FORWARD


def test_adapter_rejects_dense_model_and_disabled_moe_replay(fake_router_replay):
    replay_type, _ = fake_router_replay
    replay_type(1)
    with pytest.raises(ValueError, match="requires an MCore MoE model"):
        ModelScopedRouterReplay(FakeModel([]))
    assert replay_type.global_router_replay_instances == []

    disabled_layer = _moe_layer(3, None)
    with pytest.raises(ValueError, match="moe_enable_routing_replay=True"):
        ModelScopedRouterReplay(FakeModel([disabled_layer]))


def test_adapter_rejects_multiple_model_chunks_and_duplicate_layers(fake_router_replay):
    replay_type, _ = fake_router_replay
    with pytest.raises(ValueError, match="exactly one model chunk"):
        ModelScopedRouterReplay([FakeModel([]), FakeModel([])])

    first = replay_type(2)
    second = replay_type(2)
    with pytest.raises(ValueError, match="Multiple RouterReplay"):
        ModelScopedRouterReplay(FakeModel([_moe_layer(2, first), _moe_layer(2, second)]))
    assert replay_type.global_router_replay_instances == []


def test_model_scoped_actions_do_not_cross_models(fake_router_replay):
    replay_type, action_type = fake_router_replay
    actor_replay = replay_type(1)
    actor = ModelScopedRouterReplay(FakeModel([_moe_layer(1, actor_replay)]))
    reference_replay = replay_type(1)
    reference = ModelScopedRouterReplay(FakeModel([_moe_layer(1, reference_replay)]))

    actor.prepare_forward(torch.tensor([[[4, 7]]]))

    assert actor_replay.router_replay_action is action_type.REPLAY_FORWARD
    assert reference_replay.router_replay_action is None
    assert reference_replay.target_topk_idx is None
    assert reference.layer_numbers == (1,)


def test_backward_hook_switches_mode_before_recompute(fake_router_replay):
    replay_type, action_type = fake_router_replay
    events = []
    replay = replay_type(1, events=events)
    adapter = ModelScopedRouterReplay(FakeModel([_moe_layer(1, replay)]))
    adapter.prepare_forward(torch.tensor([[[2, 3]]]))

    class HookOutput:
        requires_grad = True

        def register_hook(self, hook):
            self.hook = hook
            return "handle"

    output = HookOutput()
    assert adapter.register_backward_hook({"logits": output}) == "handle"

    output.hook("gradient")
    events.append("recompute")

    assert events == ["replay_forward", "replay_backward", "recompute"]
    assert replay.router_replay_action is action_type.REPLAY_BACKWARD


def test_cleanup_and_consumption_validation_are_model_local(fake_router_replay):
    replay_type, _ = fake_router_replay
    replay = replay_type(1)
    adapter = ModelScopedRouterReplay(FakeModel([_moe_layer(1, replay)]))
    adapter.prepare_forward(torch.tensor([[[0, 1]]]))

    with pytest.raises(RuntimeError, match="not fully consumed"):
        adapter.assert_replay_consumed()

    replay.replay_backward_list.pop(0)
    adapter.assert_replay_consumed()
    adapter.cleanup()
    assert replay.router_replay_action is None
    assert replay.target_topk_idx is None


def test_replay_data_validates_all_global_layers_before_mutation(fake_router_replay):
    replay_type, _ = fake_router_replay
    replay_one = replay_type(1)
    replay_four = replay_type(4)
    adapter = ModelScopedRouterReplay(
        FakeModel([_moe_layer(1, replay_one), _moe_layer(4, replay_four)])
    )

    with pytest.raises(ValueError, match="outside rollout layer axis"):
        adapter.set_replay_data(torch.zeros(2, 3, 2))

    assert replay_one.replay_backward_list == []
    assert replay_four.replay_backward_list == []
