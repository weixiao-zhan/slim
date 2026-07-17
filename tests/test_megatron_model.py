from __future__ import annotations

from types import SimpleNamespace

import pytest

from slim.backends.megatron import model as model_module
from slim.backends.megatron.model import MegatronModelAPI, build_megatron_model

pytestmark = pytest.mark.unit


def _fake_runtime(*, chunks=None, derived_vp_field=None, num_moe_experts=8, expected_ep=2):
    events = []
    state = SimpleNamespace(events=events, setup_calls=[])

    def config_type(name):
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                state.__dict__.setdefault(f"{name}_instances", []).append(self)

            def finalize(self):
                self.finalized = True
                events.append(f"finalize:{name}")

        Config.__name__ = name
        return Config

    class Provider:
        bf16 = True
        fp16 = False
        params_dtype = "bf16"
        num_moe_experts = None
        calculate_per_token_loss = False
        recompute_granularity = None
        recompute_method = None
        recompute_num_layers = None

        def __init__(self):
            self.num_moe_experts = num_moe_experts
            self.provide_calls = []

        def finalize(self):
            events.append("finalize:provider")
            assert self.tensor_model_parallel_size == 2
            assert self.pipeline_model_parallel_size == 2
            assert self.context_parallel_size == 2
            assert self.sequence_parallel is True
            assert self.expert_model_parallel_size == expected_ep
            assert self.virtual_pipeline_model_parallel_size is None
            if derived_vp_field is not None:
                setattr(self, derived_vp_field, 2)

        def initialize_model_parallel(self, **kwargs):
            events.append("initialize_model_parallel")
            self.initialize_kwargs = kwargs

        def provide_distributed_model(self, **kwargs):
            events.append("provide_distributed_model")
            self.provide_calls.append(kwargs)
            return [object()] if chunks is None else chunks

    provider = Provider()

    class Bridge:
        def to_megatron_provider(self, **kwargs):
            events.append("to_megatron_provider")
            self.provider_kwargs = kwargs
            return provider

    bridge = Bridge()

    class AutoBridge:
        @classmethod
        def from_hf_pretrained(cls, checkpoint, **kwargs):
            events.append("from_hf_pretrained")
            state.checkpoint = checkpoint
            state.bridge_kwargs = kwargs
            return bridge

    pg_collection = object()

    class ProcessGroupCollection:
        @classmethod
        def use_mpu_process_groups(cls):
            events.append("use_mpu_process_groups")
            return pg_collection

    def setup_optimizer(**kwargs):
        events.append("setup_optimizer")
        state.setup_calls.append(kwargs)
        return "optimizer", "scheduler"

    api = MegatronModelAPI(
        AutoBridge=AutoBridge,
        DistributedDataParallelConfig=config_type("DistributedDataParallelConfig"),
        OptimizerConfig=config_type("OptimizerConfig"),
        SchedulerConfig=config_type("SchedulerConfig"),
        ProcessGroupCollection=ProcessGroupCollection,
        setup_optimizer=setup_optimizer,
    )
    state.provider = provider
    state.bridge = bridge
    state.pg_collection = pg_collection
    return api, state


def _build(api, **kwargs):
    return build_megatron_model(
        "/models/qwen",
        world_size=16,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        expert_model_parallel_size=2,
        use_rollout_routing_replay=True,
        api=api,
        **kwargs,
    )


def test_lazy_loader_uses_bridge_revision_import_paths(monkeypatch):
    symbols = {
        "megatron.bridge": SimpleNamespace(AutoBridge=object()),
        "megatron.bridge.training.config": SimpleNamespace(
            DistributedDataParallelConfig=object(),
            OptimizerConfig=object(),
            SchedulerConfig=object(),
        ),
        "megatron.core.process_groups_config": SimpleNamespace(ProcessGroupCollection=object()),
        "megatron.bridge.training.optim": SimpleNamespace(setup_optimizer=object()),
    }
    imported = []

    def fake_import(name):
        imported.append(name)
        return symbols[name]

    monkeypatch.setattr(model_module.importlib, "import_module", fake_import)
    api = model_module.load_megatron_model_api()

    assert api.AutoBridge is symbols["megatron.bridge"].AutoBridge
    assert (
        api.DistributedDataParallelConfig is symbols["megatron.bridge.training.config"].DistributedDataParallelConfig
    )
    assert api.ProcessGroupCollection is symbols["megatron.core.process_groups_config"].ProcessGroupCollection
    assert api.setup_optimizer is symbols["megatron.bridge.training.optim"].setup_optimizer
    assert imported == list(symbols)


def test_builds_bridge_model_and_distributed_optimizer():
    api, state = _fake_runtime()

    bundle = _build(
        api,
        bridge_load_kwargs={"local_files_only": True},
        ddp_config_kwargs={"overlap_grad_reduce": True},
        optimizer_config_kwargs={"lr": 2e-6, "weight_decay": 0.1},
        scheduler_config_kwargs={
            "lr_decay_style": "cosine",
            "lr_decay_steps": 100,
            "lr_warmup_steps": 10,
        },
    )

    assert state.checkpoint == "/models/qwen"
    assert state.bridge_kwargs == {"local_files_only": True, "trust_remote_code": True}
    assert state.bridge.provider_kwargs == {"load_weights": True}
    assert state.provider.moe_enable_routing_replay is True
    assert state.provider._enable_in_batch_packing is True
    assert state.provider.initialize_kwargs == {"seed": 1234}
    assert bundle.bridge is state.bridge
    assert bundle.provider is state.provider
    assert bundle.optimizer == "optimizer"
    assert bundle.scheduler == "scheduler"
    assert bundle.pg_collection is state.pg_collection
    assert bundle.topology.data_parallel_size == 2

    ddp = state.DistributedDataParallelConfig_instances[0]
    optimizer_config = state.OptimizerConfig_instances[0]
    scheduler_config = state.SchedulerConfig_instances[0]
    assert ddp.use_distributed_optimizer is True
    assert ddp.overlap_grad_reduce is True
    assert optimizer_config.use_distributed_optimizer is True
    assert optimizer_config.lr == 2e-6
    assert optimizer_config.bf16 is True
    assert scheduler_config.lr_decay_steps == 100
    assert scheduler_config.lr_warmup_steps == 10
    assert scheduler_config.wd_incr_steps == 100
    assert scheduler_config.start_weight_decay == 0.1
    assert all(config.finalized for config in (ddp, optimizer_config, scheduler_config))

    assert state.provider.provide_calls == [{"ddp_config": ddp, "pg_collection": state.pg_collection}]
    assert state.setup_calls == [
        {
            "optimizer_config": optimizer_config,
            "scheduler_config": scheduler_config,
            "model": bundle.model,
            "use_gloo_process_groups": False,
            "pg_collection": state.pg_collection,
        }
    ]
    assert state.events.index("finalize:provider") < state.events.index("initialize_model_parallel")
    assert state.events.index("provide_distributed_model") < state.events.index("setup_optimizer")


def test_applies_native_provider_config_before_finalize():
    api, state = _fake_runtime()

    _build(
        api,
        provider_config_kwargs={
            "calculate_per_token_loss": True,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    )

    assert state.provider.calculate_per_token_loss is True
    assert state.provider.recompute_granularity == "full"
    assert state.provider.recompute_method == "uniform"
    assert state.provider.recompute_num_layers == 1


def test_rejects_unknown_provider_config_field():
    api, _state = _fake_runtime()

    with pytest.raises(AttributeError, match="no attribute"):
        _build(api, provider_config_kwargs={"not_a_provider_field": True})


@pytest.mark.parametrize(
    "field",
    [
        "virtual_pipeline_model_parallel_size",
        "num_layers_per_virtual_pipeline_stage",
        "num_virtual_stages_per_pipeline_rank",
    ],
)
def test_rejects_virtual_pipeline_fields_derived_during_finalize(field):
    api, state = _fake_runtime(derived_vp_field=field)

    with pytest.raises(ValueError, match="does not support virtual pipeline parallelism"):
        _build(api)

    assert "initialize_model_parallel" not in state.events


def test_rejects_multiple_model_chunks_before_optimizer_setup():
    api, state = _fake_runtime(chunks=[object(), object()])

    with pytest.raises(ValueError, match="exactly one model chunk"):
        _build(api)

    assert state.setup_calls == []


def test_rejects_routing_replay_for_dense_provider():
    api, state = _fake_runtime(num_moe_experts=None, expected_ep=1)

    with pytest.raises(ValueError, match="requires a Bridge MoE provider"):
        build_megatron_model(
            "/models/qwen-dense",
            world_size=8,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            sequence_parallel=True,
            expert_model_parallel_size=1,
            use_rollout_routing_replay=True,
            api=api,
        )

    assert "initialize_model_parallel" not in state.events
