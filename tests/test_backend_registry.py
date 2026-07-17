import sys
from types import SimpleNamespace

import pytest

from slim.backends import registry


NUM_GPUS = 0


@pytest.mark.unit
def test_backend_selection_does_not_import_backend_modules(monkeypatch):
    def fail_import(name):
        raise AssertionError(f"backend module imported while selecting backend: {name}")

    monkeypatch.setattr(registry.importlib, "import_module", fail_import)

    assert registry.get_training_backend([]) == "fsdp"
    assert registry.get_training_backend(["--training-backend", "megatron"]) == "megatron"


@pytest.mark.unit
def test_default_parser_loads_only_fsdp(monkeypatch):
    calls = []

    def fsdp_parse_args(**kwargs):
        assert kwargs["ignore_unknown_args"] is True
        return SimpleNamespace(training_backend="fsdp")

    fsdp_module = SimpleNamespace(fsdp_parse_args=fsdp_parse_args)

    def import_module(name):
        calls.append(name)
        if name == "slim.backends.fsdp_utils.arguments":
            return fsdp_module
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(registry.importlib, "import_module", import_module)
    monkeypatch.setattr(sys, "argv", ["test"])

    args = registry.parse_backend_args(ignore_unknown_args=True)

    assert args.training_backend == "fsdp"
    assert calls == ["slim.backends.fsdp_utils.arguments"]


@pytest.mark.unit
def test_trainer_classes_are_resolved_by_backend_and_role(monkeypatch):
    actor_cls = type("ActorTrainer", (), {})
    critic_cls = type("CriticTrainer", (), {})
    modules = {
        "slim.backends.fsdp_utils.actor": SimpleNamespace(ActorFSDPTrainer=actor_cls),
        "slim.backends.fsdp_utils.critic": SimpleNamespace(CriticFSDPTrainer=critic_cls),
    }
    calls = []

    def import_module(name):
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(registry.importlib, "import_module", import_module)

    assert registry.get_trainer_class("fsdp", "actor") is actor_cls
    assert registry.get_trainer_class("fsdp", "critic") is critic_cls
    assert calls == [
        "slim.backends.fsdp_utils.actor",
        "slim.backends.fsdp_utils.critic",
    ]


@pytest.mark.unit
def test_megatron_rejects_critic_trainer_role_before_import(monkeypatch):
    def fail_import(name):
        raise AssertionError(f"unexpected module import: {name}")

    monkeypatch.setattr(registry.importlib, "import_module", fail_import)

    with pytest.raises(ValueError, match="does not support role 'critic'"):
        registry.get_trainer_class("megatron", "critic")


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["unknown", "", None])
def test_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="Unknown training backend"):
        registry.get_trainer_class(backend, "actor")
