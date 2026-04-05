from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

try:
    from ._shared import get_contract_path, install_paths, install_stubs, run_contract_test_for_file
except ImportError:
    try:
        from plugin_contracts._shared import (
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )
    except ImportError:
        from _shared import get_contract_path, install_paths, install_stubs, run_contract_test_for_file

install_paths()
install_stubs()

NUM_GPUS = 0

from slim.utils.misc import load_function
from slim.utils.types import Episode, Sample


def run_contract_test_file() -> None:
    run_contract_test_for_file(
        __file__,
        path_args=[
            "custom-rollout-log-function-path",
            "custom-eval-rollout-log-function-path",
            "custom-reward-post-process-path",
            "rollout-data-postprocess-path",
        ],
    )


def reference_custom_rollout_log(rollout_id, args, episodes, rollout_extra_metrics, rollout_time) -> bool:
    args.logged_rollout_id = rollout_id
    return True


def reference_custom_eval_rollout_log(rollout_id, args, data, extra_metrics) -> bool:
    args.logged_eval_rollout_id = rollout_id
    return True


def reference_reward_post_process(args, episodes):
    """Custom reward post-process: operates on list[Episode] in-place."""
    for ep in episodes:
        ep.reward = ep.reward + 1.0


def reference_rollout_data_postprocess(args) -> None:
    args.rollout_data_postprocess_called = True


def make_sample(index: int, reward: float = 1.0) -> Sample:
    return Sample(
        index=index,
        reward=reward,
        tokens=[index, index + 1],
        response_length=2,
        status=Sample.Status.COMPLETED,
        loss_mask=[1, 1],
    )


@dataclass(frozen=True)
class HookCase:
    name: str
    env_key: str
    default_path: str
    source_path: str
    runtime_marker: str
    expected_params: tuple[str, ...]
    invoke: object


def invoke_custom_rollout_log(fn):
    args = type("Args", (), {})()
    assert isinstance(fn(3, args, [Episode(tokens=[1], response_length=1, reward=1.0)], {"reward": 1.0}, 0.5), bool)
    assert args.logged_rollout_id == 3


def invoke_custom_eval_rollout_log(fn):
    args = type("Args", (), {})()
    sample = Sample(index=0, reward=1.0)
    assert isinstance(
        fn(4, args, {"eval_set": {"rewards": [1.0], "truncated": [False], "samples": [sample]}}, {"acc": 1.0}), bool
    )
    assert args.logged_eval_rollout_id == 4


def invoke_reward_post_process(fn):
    episodes = [Episode(tokens=[0], response_length=1, reward=0.5), Episode(tokens=[1], response_length=1, reward=1.5)]
    fn(type("Args", (), {})(), episodes)
    # rewards should have been modified in-place
    assert episodes[0].reward != 0.5 or episodes[1].reward != 1.5


def invoke_rollout_data_postprocess(fn):
    args = type("Args", (), {})()
    assert fn(args) is None
    assert args.rollout_data_postprocess_called is True


HOOK_CASES = [
    HookCase(
        "custom_rollout_log",
        "CUSTOM_ROLLOUT_LOG_FUNCTION_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_custom_rollout_log",
        "slim/ray/rollout.py",
        "custom_log_func(rollout_id, args, episodes, rollout_extra_metrics, rollout_time)",
        ("rollout_id", "args", "episodes", "rollout_extra_metrics", "rollout_time"),
        invoke_custom_rollout_log,
    ),
    HookCase(
        "custom_eval_rollout_log",
        "CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_custom_eval_rollout_log",
        "slim/ray/rollout.py",
        "custom_log_func(rollout_id, args, data, extra_metrics)",
        ("rollout_id", "args", "data", "extra_metrics"),
        invoke_custom_eval_rollout_log,
    ),
    HookCase(
        "custom_reward_post_process",
        "CUSTOM_REWARD_POST_PROCESS_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_reward_post_process",
        "slim/ray/rollout.py",
        "self.custom_reward_post_process_func(self.args, episodes)",
        ("args", "episodes"),
        invoke_reward_post_process,
    ),
    HookCase(
        "rollout_data_postprocess",
        "ROLLOUT_DATA_POSTPROCESS_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_rollout_data_postprocess",
        "slim/backends/fsdp_utils/actor.py",
        "self.rollout_data_postprocess(self.args)",
        ("args",),
        invoke_rollout_data_postprocess,
    ),
]


@pytest.mark.parametrize("case", HOOK_CASES, ids=[case.name for case in HOOK_CASES])
def test_runtime_hook_callsite_is_stable(case: HookCase):
    assert case.runtime_marker in Path(case.source_path).read_text()


@pytest.mark.parametrize("case", HOOK_CASES, ids=[case.name for case in HOOK_CASES])
def test_runtime_hook_path_aligns_with_expected_format(case: HookCase):
    fn = load_function(get_contract_path(case.env_key, case.default_path))
    assert tuple(inspect.signature(fn).parameters) == case.expected_params
    case.invoke(fn)


if __name__ == "__main__":
    run_contract_test_file()
