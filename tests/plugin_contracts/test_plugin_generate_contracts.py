# Source: https://github.com/THUDM/slime
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import types
from contextlib import contextmanager

import pytest

try:
    from ._shared import install_paths, install_stubs
except ImportError:
    try:
        from plugin_contracts._shared import install_paths, install_stubs
    except ImportError:
        from _shared import install_paths, install_stubs

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
from slim.utils.types import Episode, Trajectory


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
        self.aborted = False

    @contextmanager
    def dp_rank_context(self):
        yield 0


async def custom_generate(state, episode: Episode):
    episode.trajectories.append(Trajectory(token_ids=[11, 12, 13], generated_text="generated"))
    episode.reward = 0.25
    episode.status = Episode.Status.COMPLETED
    return episode


async def custom_generate_with_evaluation(state, episode: Episode, evaluation: bool = False):
    episode.trajectories.append(
        Trajectory(
            token_ids=[21, 22],
            generated_text="eval-generated" if evaluation else "train-generated",
        )
    )
    episode.reward = 0.5 if evaluation else 0.75
    episode.status = Episode.Status.COMPLETED
    episode.example["evaluation"] = evaluation
    return episode


async def custom_generate_multi_trajectory(state, episode: Episode):
    """Agentic attempt: a second trajectory appended for a sub-agent dispatch."""
    episode.trajectories.append(Trajectory(token_ids=[31, 32], generated_text="main-trajectory"))
    episode.trajectories.append(Trajectory(token_ids=[33, 34], generated_text="sub-agent-trajectory"))
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
    assert [t.generated_text for t in result.trajectories] == ["main-trajectory", "sub-agent-trajectory"]
