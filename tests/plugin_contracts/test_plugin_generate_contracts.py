# Source: https://github.com/THUDM/slime
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import types
from contextlib import contextmanager

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
install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0
REFERENCE_CUSTOM_GENERATE_PATH = "plugin_contracts.test_plugin_generate_contracts.custom_generate"
REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH = (
    "plugin_contracts.test_plugin_generate_contracts.custom_generate_with_evaluation"
)
REFERENCE_CUSTOM_GENERATE_MULTI_TRAJECTORY_PATH = (
    "plugin_contracts.test_plugin_generate_contracts.custom_generate_multi_trajectory"
)

from slim.rollout.sglang_rollout import generate_and_rm
from slim.utils.misc import load_function
from slim.utils.types import Episode, Trajectory


def run_contract_test_file() -> None:
    run_contract_test_for_file(__file__, path_args=["custom-generate-function-path"])


def make_args(**overrides):
    class Args:
        group_rm = False
        custom_generate_function_path = None

    args = Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _FakeTokenizer:
    def decode(self, ids) -> str:
        return " ".join(str(i) for i in ids)


class FakeGenerateState:
    def __init__(self, args) -> None:
        self.args = args
        self.tokenizer = _FakeTokenizer()
        self.semaphore = types.SimpleNamespace(__aenter__=None)
        self.pendings = set()
        self.aborted = False

    @contextmanager
    def dp_rank_context(self):
        yield 0


async def custom_generate(state, episode: Episode):
    episode.trajectory.token_ids = [11, 12, 13]
    episode.trajectory.generated_text = "generated"
    episode.reward = 0.25
    episode.status = Episode.Status.COMPLETED
    return episode


async def custom_generate_with_evaluation(state, episode: Episode, evaluation: bool = False):
    episode.trajectory.token_ids = [21, 22]
    episode.trajectory.generated_text = "eval-generated" if evaluation else "train-generated"
    episode.reward = 0.5 if evaluation else 0.75
    episode.status = Episode.Status.COMPLETED
    episode.example["evaluation"] = evaluation
    return episode


async def custom_generate_multi_trajectory(state, episode: Episode):
    """Agentic attempt: a second span appended for a sub-agent dispatch."""
    episode.trajectories[0].token_ids = [31, 32]
    episode.trajectories[0].generated_text = "main-span"
    episode.trajectories.append(Trajectory(token_ids=[33, 34], generated_text="sub-agent-span"))
    episode.reward = 0.5
    episode.status = Episode.Status.COMPLETED
    return episode


def assert_episode_contract(episode: Episode) -> None:
    assert isinstance(episode, Episode)
    assert episode.trajectories
    for trajectory in episode.trajectories:
        assert isinstance(trajectory.token_ids, list)
        assert isinstance(trajectory.generated_text, str)
    assert episode.get_reward_value() is not None


def assert_custom_generate_signature_matches_expected(fn) -> None:
    params = tuple(inspect.signature(fn).parameters)
    assert params[:2] == ("state", "episode") or params[:2] == ("state", "sample")


class _DummySemaphore:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PatchedGenerateState(FakeGenerateState):
    def __init__(self, args):
        super().__init__(args)
        self.semaphore = _DummySemaphore()


@pytest.fixture
def patch_generate_state(monkeypatch):
    """Patch GenerateState with a test-safe variant; returns the sglang_rollout module."""
    from slim.rollout import sglang_rollout

    monkeypatch.setattr(sglang_rollout, "GenerateState", _PatchedGenerateState)
    return sglang_rollout


def test_generate_and_rm_default_generate_branch_is_stable(patch_generate_state, monkeypatch):
    sglang_rollout = patch_generate_state

    async def official_default_generate(state, episode: Episode):
        episode.trajectory.token_ids = [31, 32]
        episode.trajectory.generated_text = "default-generate"
        episode.reward = 1.0
        episode.status = Episode.Status.COMPLETED
        return episode

    monkeypatch.setattr(sglang_rollout, "generate", official_default_generate)

    result = asyncio.run(
        generate_and_rm(
            make_args(custom_generate_function_path=None),
            Episode.from_example({"prompt": "prompt"}),
            evaluation=False,
        )
    )
    assert_episode_contract(result)
    assert result.trajectory.generated_text == "default-generate"


def test_generate_and_rm_prefers_per_episode_generate_function(patch_generate_state):
    args = make_args(custom_generate_function_path=REFERENCE_CUSTOM_GENERATE_PATH)
    ep = Episode.from_example({"prompt": "prompt"})
    ep.generate_function_path = REFERENCE_CUSTOM_GENERATE_WITH_EVAL_PATH
    result = asyncio.run(generate_and_rm(args, ep, evaluation=True))
    assert_episode_contract(result)
    assert result.example["evaluation"] is True


def test_generate_and_rm_decodes_every_trajectory(patch_generate_state):
    args = make_args(custom_generate_function_path=REFERENCE_CUSTOM_GENERATE_MULTI_TRAJECTORY_PATH)
    result = asyncio.run(generate_and_rm(args, Episode.from_example({"prompt": "prompt"}), evaluation=False))
    assert_episode_contract(result)
    assert [t.text for t in result.trajectories] == ["31 32", "33 34"]
    assert [t.generated_text for t in result.trajectories] == ["main-span", "sub-agent-span"]


def test_custom_generate_function_path_supports_user_override(patch_generate_state):
    custom_generate_path = get_contract_path(
        "CUSTOM_GENERATE_FUNCTION_PATH",
        REFERENCE_CUSTOM_GENERATE_PATH,
    )
    assert_custom_generate_signature_matches_expected(load_function(custom_generate_path))
    result = asyncio.run(
        generate_and_rm(
            make_args(custom_generate_function_path=custom_generate_path),
            Episode.from_example({"prompt": "prompt"}),
            evaluation=False,
        )
    )
    assert_episode_contract(result)


if __name__ == "__main__":
    run_contract_test_file()
