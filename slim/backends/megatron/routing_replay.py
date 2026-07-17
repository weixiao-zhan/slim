"""Model-scoped routing replay for MCore MoE routers."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .topology import require_single_model_chunk


def _router_replay_api() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module("megatron.core.transformer.moe.router_replay")
    return module.RouterReplay, module.RouterReplayAction


def _iter_modules(model: Any) -> Iterable[Any]:
    modules = getattr(model, "modules", None)
    if callable(modules):
        yield from modules()
    else:
        yield model


def _layer_number(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{owner} must have a positive, one-based global layer_number; got {value!r}.")
    return value


class ModelScopedRouterReplay:
    """Own the RouterReplay instances reachable from one local model chunk."""

    def __init__(
        self,
        model: Any | Sequence[Any],
        *,
        expected_layer_numbers: Iterable[int] | None = None,
    ) -> None:
        if isinstance(model, Sequence) and not isinstance(model, (str, bytes)):
            model = require_single_model_chunk(model)

        router_replay_type, action_type = _router_replay_api()
        self._action_type = action_type
        try:
            self._routers = self._collect_routers(model, expected_layer_numbers)
        finally:
            clear_instances = getattr(router_replay_type, "clear_global_router_replay_instances", None)
            if callable(clear_instances):
                clear_instances()
            else:
                router_replay_type.global_router_replay_instances.clear()

    @staticmethod
    def _collect_routers(
        model: Any,
        expected_layer_numbers: Iterable[int] | None,
    ) -> dict[int, Any]:
        routers: dict[int, Any] = {}
        discovered_moe_layers: set[int] = set()
        replay_disabled_layers: set[int] = set()
        seen_replays: set[int] = set()

        for module in _iter_modules(model):
            router = getattr(module, "router", None)
            if router is None or not hasattr(router, "router_replay"):
                continue

            owner_layer_number = getattr(module, "layer_number", getattr(router, "layer_number", None))
            if owner_layer_number is not None:
                owner_layer_number = _layer_number(owner_layer_number, owner=type(module).__name__)
                discovered_moe_layers.add(owner_layer_number)

            replay = router.router_replay
            if replay is None:
                if owner_layer_number is not None:
                    replay_disabled_layers.add(owner_layer_number)
                continue
            if id(replay) in seen_replays:
                continue
            seen_replays.add(id(replay))

            replay_layer_number = _layer_number(
                getattr(replay, "layer_number", None),
                owner=type(replay).__name__,
            )
            if owner_layer_number is not None and owner_layer_number != replay_layer_number:
                raise ValueError(
                    "MoE layer and RouterReplay disagree on global layer_number; "
                    f"got {owner_layer_number} and {replay_layer_number}."
                )
            if replay_layer_number in routers:
                raise ValueError(f"Multiple RouterReplay instances claim global layer {replay_layer_number}.")
            routers[replay_layer_number] = replay

        if replay_disabled_layers:
            layers = ", ".join(map(str, sorted(replay_disabled_layers)))
            raise ValueError(
                "MCore routing replay is disabled for local MoE layers "
                f"{layers}; set provider.moe_enable_routing_replay=True before model construction."
            )
        if not routers:
            raise ValueError("Routing replay requires an MCore MoE model; the local model chunk has no MoE routers.")

        expected = (
            {_layer_number(layer, owner="expected layer") for layer in expected_layer_numbers}
            if expected_layer_numbers is not None
            else discovered_moe_layers
        )
        if expected and set(routers) != expected:
            missing = sorted(expected - set(routers))
            unexpected = sorted(set(routers) - expected)
            raise ValueError(
                "RouterReplay layers do not match the expected local MoE layers; "
                f"missing={missing}, unexpected={unexpected}."
            )
        return dict(sorted(routers.items()))

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        """Return one-based global layer numbers owned by this model."""

        return tuple(self._routers)

    @property
    def routers_by_layer(self) -> Mapping[int, Any]:
        """Return a read-only view of the model-local layer mapping."""

        from types import MappingProxyType

        return MappingProxyType(self._routers)

    def set_replay_data(self, routed_experts: Any) -> None:
        """Install token-aligned rollout routes using global layer numbers."""

        shape = getattr(routed_experts, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError(
                "routed_experts must have shape [num_tokens, num_hidden_layers, top_k]; "
                f"got {shape!r}."
            )
        if shape[2] < 1:
            raise ValueError("routed_experts top_k dimension must be positive.")

        num_hidden_layers = shape[1]
        highest_local_layer = max(self._routers)
        if highest_local_layer > num_hidden_layers:
            raise ValueError(
                f"RouterReplay global layer {highest_local_layer} is outside rollout layer axis "
                f"with size {num_hidden_layers}."
            )
        for layer_number, replay in self._routers.items():
            replay.set_target_indices(routed_experts[:, layer_number - 1, :])

    def set_forward_mode(self) -> None:
        """Use rollout routes during an original forward."""

        action = self._action_type.REPLAY_FORWARD
        for replay in self._routers.values():
            replay.set_router_replay_action(action)

    def set_backward_mode(self) -> None:
        """Use queued rollout routes during activation recomputation."""

        action = self._action_type.REPLAY_BACKWARD
        for replay in self._routers.values():
            replay.set_router_replay_action(action)

    def prepare_forward(self, routed_experts: Any) -> None:
        """Install one microbatch and select original-forward replay mode."""

        self.set_replay_data(routed_experts)
        self.set_forward_mode()

    def register_backward_hook(self, output: Any) -> Any:
        """Switch to backward replay before autograd enters recomputation."""

        hookable = self._find_hookable_output(output)
        if hookable is None:
            raise ValueError("The schedule output has no differentiable tensor for the routing replay hook.")

        def switch_to_backward(gradient: Any) -> Any:
            self.set_backward_mode()
            return gradient

        return hookable.register_hook(switch_to_backward)

    @classmethod
    def _find_hookable_output(cls, output: Any) -> Any | None:
        register_hook = getattr(output, "register_hook", None)
        if callable(register_hook) and getattr(output, "requires_grad", True):
            return output
        if isinstance(output, Mapping):
            values = output.values()
        elif isinstance(output, (tuple, list)):
            values = output
        else:
            return None
        for value in values:
            hookable = cls._find_hookable_output(value)
            if hookable is not None:
                return hookable
        return None

    def pending_backward_replays(self) -> dict[int, int]:
        """Return the number of queued recompute routes per local layer."""

        return {
            layer_number: len(getattr(replay, "replay_backward_list", ()))
            for layer_number, replay in self._routers.items()
        }

    def assert_replay_consumed(self) -> None:
        """Require every activation-recompute queue to be empty."""

        pending = {layer: count for layer, count in self.pending_backward_replays().items() if count}
        if pending:
            raise RuntimeError(f"Routing replay backward queues were not fully consumed: {pending}.")

    def cleanup(self) -> None:
        """Clear model-local actions, targets, recordings, and backward queues."""

        for replay in self._routers.values():
            replay.clear_router_replay_action()
            replay.clear_indices()

    def __enter__(self) -> ModelScopedRouterReplay:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.cleanup()
