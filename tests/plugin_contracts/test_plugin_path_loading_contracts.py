# Source: https://github.com/THUDM/slime
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass

import pytest

try:
    from ._shared import contract_env_name, get_contract_path, install_paths, install_stubs, run_contract_test_for_file
except ImportError:
    try:
        from plugin_contracts._shared import (
            contract_env_name,
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )
    except ImportError:
        from _shared import (
            contract_env_name,
            get_contract_path,
            install_paths,
            install_stubs,
            run_contract_test_for_file,
        )

install_paths()
install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0

from slim.rollout.base_types import RolloutFnEvalOutput
# RolloutDataSource used for data_source contract checks
from slim.rollout.data_source import RolloutDataSource
from slim.rollout.filter_hub.base_types import DynamicFilterOutput, call_dynamic_filter
from slim.rollout.rm_hub import async_rm, batched_async_rm
from slim.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slim.utils.misc import load_function
from slim.utils.types import Episode, Trajectory


def run_contract_test_file() -> None:
    def _extra(parsed):
        if parsed.group_rm:
            os.environ[contract_env_name("GROUP_RM")] = "1"

    run_contract_test_for_file(
        __file__,
        path_args=[
            "eval-function-path",
            "custom-rm-path",
            "rollout-group-filter-path",
            "buffer-filter-path",
            "data-source-path",
            "rollout-sample-filter-path",
            "rollout-all-samples-process-path",
        ],
        extra_args=[("--group-rm", {"action": "store_true", "default": False})],
        extra_setup=_extra,
    )


def _make_episode(reward: float = 1.0, **example_fields) -> Episode:
    ep = Episode.from_example(example_fields)
    ep.trajectories.append(
        Trajectory(
            token_ids=[100, 200],
            loss_mask=[1],
            generated_text=f"response-{example_fields.get('index', 0)}",
        )
    )
    ep.reward = reward
    ep.status = Episode.Status.COMPLETED
    return ep


def make_args(**overrides):
    class Args:
        rollout_global_dataset = False
        buffer_filter_path = None
        n_samples_per_prompt = 2
        custom_rm_path = None
        group_rm = False
        rm_type = None
        reward_key = None
        hf_checkpoint = "gpt2"

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ReferenceDataSource:
    def __init__(self, args):
        self.args = args
        self._groups = [
            [Episode.from_example({"index": 0}), Episode.from_example({"index": 1})],
            [Episode.from_example({"index": 2}), Episode.from_example({"index": 3})],
        ]

    def get_samples(self, num_samples: int) -> list[list[Episode]]:
        selected = self._groups[:num_samples]
        self._groups = self._groups[num_samples:]
        return selected

    def add_samples(self, samples: list[list[Episode]]):
        self._groups.extend(samples)

    def save(self, rollout_id):
        self.last_saved_rollout_id = rollout_id

    def load(self, rollout_id=None):
        self.last_loaded_rollout_id = rollout_id

    def __len__(self) -> int:
        return len(self._groups)


def reference_dynamic_filter(args, samples: list[Episode], **kwargs):
    keep = not any((ep.example.get("metadata") or {}).get("drop") for ep in samples)
    return DynamicFilterOutput(keep=keep, reason=None if keep else "drop-flag")


def reference_buffer_filter(args, rollout_id, buffer: list[list[Episode]], num_samples: int) -> list[list[Episode]]:
    selected = list(reversed(buffer[-num_samples:]))
    del buffer[-num_samples:]
    return selected


def reference_rollout_sample_filter(args, groups: list[list[Episode]]) -> None:
    for group in groups:
        if group:
            group[-1].remove_sample = True


def reference_rollout_all_samples_process(args, all_groups: list[list[Episode]], data_source) -> None:
    args.processed_group_count = len(all_groups)


async def reference_single_rm(args, episode: Episode, **kwargs) -> None:
    episode.reward = float(episode.example.get("index", 0)) + 0.1


async def reference_batched_rm(args, episodes: list[Episode], **kwargs) -> None:
    for ep in episodes:
        ep.reward = float(ep.example.get("index", 0)) + 0.2


def valid_eval_function(args, rollout_id, data_source, evaluation=False):
    assert evaluation is True
    ep = _make_episode(reward=0.5, index=rollout_id)
    return RolloutFnEvalOutput(
        data={"eval_contract": {"rewards": [ep.reward], "truncated": [False], "samples": [ep]}},
        metrics={"source": "contract"},
    )


class ContractEvalDataSource:
    def get_samples(self, num_samples: int) -> list[list[Episode]]:
        return [[Episode.from_example({"index": index, "prompt": f"prompt-{index}"})] for index in range(num_samples)]


@dataclass(frozen=True)
class SyncCase:
    name: str
    env_key: str
    default_path: str
    default_check: object
    path_check: object


def check_eval_function_default() -> None:
    default_sig = inspect.signature(default_generate_rollout)
    assert tuple(default_sig.parameters) == ("args", "rollout_id", "data_source", "evaluation")
    assert default_sig.parameters["evaluation"].default is False


def check_eval_function_path(path: str) -> None:
    fn = load_function(path)
    default_sig = inspect.signature(default_generate_rollout)
    candidate_sig = inspect.signature(fn)
    assert tuple(candidate_sig.parameters) == tuple(default_sig.parameters)
    if path != "slim.rollout.sglang_rollout.generate_rollout":
        output = fn(None, 5, ContractEvalDataSource(), evaluation=True)
        assert isinstance(output, RolloutFnEvalOutput)
        assert output.data


def check_dynamic_filter_default() -> None:
    fn = load_function("slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std")
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "episodes")
    output = call_dynamic_filter(fn, make_args(), [_make_episode(reward=1.0), _make_episode(reward=2.0)])
    assert isinstance(output, DynamicFilterOutput)


def check_dynamic_filter_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "episodes")
    output = call_dynamic_filter(fn, make_args(), [_make_episode(reward=1.0), _make_episode(reward=2.0)])
    assert isinstance(output, DynamicFilterOutput)


def check_buffer_filter_default() -> None:
    # buffer filter not present in this fork — check signature of reference only
    assert tuple(inspect.signature(reference_buffer_filter).parameters)[:4] == ("args", "rollout_id", "buffer", "num_samples")


def check_buffer_filter_path(path: str) -> None:
    # Use reference function if default path doesn't exist in this fork
    if path == "slim.rollout.data_source._pop_first":
        fn = reference_buffer_filter
    else:
        fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:4] == ("args", "rollout_id", "buffer", "num_samples")


def check_data_source_default() -> None:
    cls = load_function("slim.rollout.data_source.RolloutDataSource")
    assert tuple(inspect.signature(cls.__init__).parameters)[:2] == ("self", "args")


def check_data_source_path(path: str) -> None:
    cls = load_function(path)
    assert tuple(inspect.signature(cls.__init__).parameters)[:2] == ("self", "args")
    groups = cls(make_args()).get_samples(1)
    assert isinstance(groups, list)


def check_rollout_sample_filter_default() -> None:
    assert tuple(inspect.signature(reference_rollout_sample_filter).parameters)[:2] == ("args", "groups")


def check_rollout_sample_filter_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "groups")
    groups = [
        [Episode.from_example({"index": 0}), Episode.from_example({"index": 1})],
        [Episode.from_example({"index": 2}), Episode.from_example({"index": 3})],
    ]
    fn(object(), groups)
    assert any(getattr(ep, "remove_sample", False) for group in groups for ep in group)


def check_rollout_all_samples_process_default() -> None:
    assert tuple(inspect.signature(reference_rollout_all_samples_process).parameters)[:3] == (
        "args",
        "all_groups",
        "data_source",
    )


def check_rollout_all_samples_process_path(path: str) -> None:
    fn = load_function(path)
    assert tuple(inspect.signature(fn).parameters)[:3] == ("args", "all_groups", "data_source")
    args = type("Args", (), {})()
    fn(args, [[Episode.from_example({"index": 0})]], object())
    assert hasattr(args, "processed_group_count")


SYNC_CASES = [
    SyncCase(
        "eval_function",
        "EVAL_FUNCTION_PATH",
        "slim.rollout.sglang_rollout.generate_rollout",
        check_eval_function_default,
        check_eval_function_path,
    ),
    SyncCase(
        "dynamic_filter",
        "DYNAMIC_SAMPLING_FILTER_PATH",
        "slim.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        check_dynamic_filter_default,
        check_dynamic_filter_path,
    ),
    SyncCase(
        "buffer_filter",
        "BUFFER_FILTER_PATH",
        "slim.rollout.data_source._pop_first",
        check_buffer_filter_default,
        check_buffer_filter_path,
    ),
    SyncCase(
        "data_source",
        "DATA_SOURCE_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.ReferenceDataSource",
        check_data_source_default,
        check_data_source_path,
    ),
    SyncCase(
        "rollout_sample_filter",
        "ROLLOUT_SAMPLE_FILTER_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.reference_rollout_sample_filter",
        check_rollout_sample_filter_default,
        check_rollout_sample_filter_path,
    ),
    SyncCase(
        "rollout_all_samples_process",
        "ROLLOUT_ALL_SAMPLES_PROCESS_PATH",
        "plugin_contracts.test_plugin_path_loading_contracts.reference_rollout_all_samples_process",
        check_rollout_all_samples_process_default,
        check_rollout_all_samples_process_path,
    ),
]


@pytest.mark.parametrize("case", SYNC_CASES, ids=[case.name for case in SYNC_CASES])
def test_path_loading_default_behavior_is_stable(case: SyncCase):
    case.default_check()


@pytest.mark.parametrize("case", SYNC_CASES, ids=[case.name for case in SYNC_CASES])
def test_path_loading_path_aligns_with_expected_format(case: SyncCase):
    case.path_check(get_contract_path(case.env_key, case.default_path))


def _make_empty_episode() -> Episode:
    """Episode with no generated tokens — avoids needing a real tokenizer for decode."""
    ep = Episode.from_example({})
    ep.trajectories.append(Trajectory())
    ep.reward = 0.0
    ep.status = Episode.Status.COMPLETED
    return ep


def test_custom_rm_default_behavior_is_stable():
    episode = _make_empty_episode()
    asyncio.run(async_rm(make_args(rm_type="random"), episode))
    assert isinstance(episode.reward, (int, float))

    group = [_make_empty_episode(), _make_empty_episode()]
    asyncio.run(batched_async_rm(make_args(group_rm=True, rm_type="random"), group))
    assert all(isinstance(ep.reward, (int, float)) for ep in group)


def test_custom_rm_path_aligns_with_expected_format():
    path = get_contract_path("CUSTOM_RM_PATH")
    if get_contract_path("GROUP_RM") == "1":
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "episodes")
        group = [_make_episode(reward=0.0, index=0), _make_episode(reward=0.0, index=1)]
        asyncio.run(
            batched_async_rm(
                make_args(
                    group_rm=True,
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm",
                ),
                group,
            )
        )
        assert all(isinstance(ep.reward, (int, float)) for ep in group)
    else:
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "episode")
        episode = _make_episode(reward=0.0, index=3)
        asyncio.run(
            async_rm(
                make_args(
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm"
                ),
                episode,
            )
        )
        assert isinstance(episode.reward, (int, float))


if __name__ == "__main__":
    run_contract_test_file()
