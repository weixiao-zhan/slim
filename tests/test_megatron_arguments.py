import sys

import pytest

from slim.backends.megatron.arguments import megatron_parse_args, validate_args


NUM_GPUS = 0


def _add_slim_test_arguments(parser):
    parser.add_argument("--training-backend", choices=("fsdp", "megatron"), default="megatron")
    parser.add_argument("--actor-num-gpus", type=int, default=8)
    parser.add_argument("--actor-num-gpus-per-replica", type=int, default=None)
    parser.add_argument("--advantage-estimator", default="grpo")
    parser.add_argument("--critic-train-only", action="store_true", default=False)
    parser.add_argument("--use-peft", action="store_true", default=False)
    parser.add_argument("--async-save", action="store_true", default=False)
    parser.add_argument("--no-save-optim", action="store_true", default=False)
    parser.add_argument("--rollout-colocate", action="store_true", default=False)
    parser.add_argument("--only-train-params-name-list", nargs="+", default=None)
    parser.add_argument("--freeze-params-name-list", nargs="+", default=None)
    parser.add_argument("--loss-type", default="policy_loss")
    parser.add_argument("--kl-loss-coef", type=float, default=0.0)
    parser.add_argument("--ref-update-interval", type=int, default=None)
    parser.add_argument("--mismatch-correction", default="none")
    parser.add_argument("--get-mismatch-metrics", action="store_true", default=False)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-actor", type=float, default=1e-6)
    return parser


def _parse(monkeypatch, *options, ignore_unknown_args=False):
    monkeypatch.setattr(sys, "argv", ["test", *options])
    return megatron_parse_args(
        extra_args_provider=_add_slim_test_arguments,
        ignore_unknown_args=ignore_unknown_args,
    )


@pytest.mark.unit
def test_bridge_parallelism_names_and_defaults(monkeypatch):
    args = _parse(monkeypatch)

    assert args.tensor_model_parallel_size == 1
    assert args.pipeline_model_parallel_size == 1
    assert args.context_parallel_size == 1
    assert args.sequence_parallel is False
    assert args.expert_model_parallel_size == 1
    assert args.use_distributed_optimizer is True
    assert args.world_size == 8
    assert not hasattr(args, "virtual_pipeline_model_parallel_size")


@pytest.mark.unit
def test_bridge_parallelism_names_parse_without_mapping(monkeypatch):
    args = _parse(
        monkeypatch,
        "--actor-num-gpus",
        "16",
        "--tensor-model-parallel-size",
        "2",
        "--pipeline-model-parallel-size",
        "2",
        "--context-parallel-size",
        "2",
        "--sequence-parallel",
        "--expert-model-parallel-size",
        "4",
        "--no-use-distributed-optimizer",
    )

    assert args.tensor_model_parallel_size == 2
    assert args.pipeline_model_parallel_size == 2
    assert args.context_parallel_size == 2
    assert args.sequence_parallel is True
    assert args.expert_model_parallel_size == 4
    assert args.use_distributed_optimizer is False


@pytest.mark.unit
def test_optimizer_arguments_use_mcore_names(monkeypatch):
    args = _parse(
        monkeypatch,
        "--optimizer",
        "adam",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.8",
        "--adam-beta2",
        "0.9",
        "--adam-eps",
        "1e-6",
        "--lr-min",
        "1e-7",
        "--lr-decay-style",
        "cosine",
        "--lr-decay-iters",
        "100",
        "--lr-warmup-iters",
        "10",
        "--gradient-checkpointing",
    )

    assert args.optimizer == "adam"
    assert args.weight_decay == 0.1
    assert args.adam_beta1 == 0.8
    assert args.adam_beta2 == 0.9
    assert args.adam_eps == 1e-6
    assert args.lr_min == 1e-7
    assert args.lr_decay_style == "cosine"
    assert args.lr_decay_iters == 100
    assert args.lr_warmup_iters == 10
    assert args.gradient_checkpointing is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("short_option", "long_option"),
    [
        ("--tp-size", "--tensor-model-parallel-size"),
        ("--pp-size", "--pipeline-model-parallel-size"),
        ("--cp-size", "--context-parallel-size"),
        ("--sp", "--sequence-parallel"),
        ("--ep-size", "--expert-model-parallel-size"),
    ],
)
def test_short_parallelism_names_are_rejected(monkeypatch, short_option, long_option):
    with pytest.raises(ValueError, match=long_option):
        _parse(monkeypatch, short_option, ignore_unknown_args=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "option",
    [
        "--virtual-pipeline-model-parallel-size",
        "--num-layers-per-virtual-pipeline-stage",
        "--num-virtual-stages-per-pipeline-rank",
    ],
)
def test_virtual_pipeline_cli_is_rejected(monkeypatch, option):
    with pytest.raises(ValueError, match="does not support virtual pipeline parallelism"):
        _parse(monkeypatch, option, "2", ignore_unknown_args=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "virtual_pipeline_model_parallel_size",
        "num_layers_per_virtual_pipeline_stage",
        "num_virtual_stages_per_pipeline_rank",
    ],
)
def test_virtual_pipeline_custom_config_fields_are_rejected(monkeypatch, field):
    args = _parse(monkeypatch)
    setattr(args, field, 1)

    with pytest.raises(ValueError, match="does not support virtual pipeline parallelism"):
        validate_args(args)


@pytest.mark.unit
def test_world_size_must_be_divisible_by_dense_topology(monkeypatch):
    with pytest.raises(ValueError, match="world size must be divisible"):
        _parse(
            monkeypatch,
            "--actor-num-gpus",
            "8",
            "--tensor-model-parallel-size",
            "2",
            "--pipeline-model-parallel-size",
            "2",
            "--context-parallel-size",
            "3",
        )


@pytest.mark.unit
def test_sequence_parallel_requires_tensor_parallelism(monkeypatch):
    with pytest.raises(ValueError, match="requires --tensor-model-parallel-size greater than 1"):
        _parse(monkeypatch, "--sequence-parallel")


@pytest.mark.unit
@pytest.mark.parametrize(
    "option",
    [
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--expert-model-parallel-size",
    ],
)
def test_parallelism_sizes_must_be_positive(monkeypatch, option):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse(monkeypatch, option, "0")


@pytest.mark.unit
def test_megatron_backend_rejects_critic_role(monkeypatch):
    with pytest.raises(ValueError, match="supports only the actor role"):
        _parse(monkeypatch, "--advantage-estimator", "ppo_gae")


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        ("--use-peft",),
        ("--async-save",),
        ("--no-save-optim",),
        ("--rollout-colocate",),
        ("--only-train-params-name-list", "layer"),
        ("--freeze-params-name-list", "layer"),
    ],
)
def test_unsupported_training_flags_are_rejected(monkeypatch, options):
    with pytest.raises(ValueError, match="is not supported"):
        _parse(monkeypatch, *options)


@pytest.mark.unit
def test_fsdp_replica_size_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="FSDP sharding option"):
        _parse(
            monkeypatch,
            "--actor-num-gpus",
            "8",
            "--actor-num-gpus-per-replica",
            "4",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        ("--weight-decay", "-0.1"),
        ("--adam-beta1", "1"),
        ("--adam-beta2", "-0.1"),
        ("--adam-eps", "0"),
        ("--lr-min=-1e-6",),
        ("--lr-decay-iters", "0"),
        ("--lr-warmup-iters", "-1"),
        ("--clip-grad", "0"),
        ("--lr-actor", "0"),
    ],
)
def test_invalid_optimizer_arguments_are_rejected(monkeypatch, options):
    with pytest.raises(ValueError):
        _parse(monkeypatch, *options)


@pytest.mark.unit
def test_expert_parallel_grid_must_divide_world_size(monkeypatch):
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        _parse(
            monkeypatch,
            "--actor-num-gpus",
            "8",
            "--pipeline-model-parallel-size",
            "2",
            "--expert-model-parallel-size",
            "3",
        )
