from __future__ import annotations

import inspect

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
DEFAULT_ROLLOUT_FUNCTION_PATH = "slim.rollout.sglang_rollout.generate_rollout"
REFERENCE_ROLLOUT_FUNCTION_PATH = "plugin_contracts.test_plugin_rollout_contracts.valid_rollout_function"

from slim.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slim.rollout.sglang_rollout import generate_rollout as default_generate_rollout
from slim.utils.misc import load_function
from slim.utils.types import Episode


def run_contract_test_file() -> None:
    run_contract_test_for_file(__file__, path_args=["rollout-function-path"])


def make_episode(index: int, reward: float = 1.0) -> Episode:
    tokens = [1000 + index, 2000 + index]
    return Episode(
        tokens=tokens,
        response_length=len(tokens),
        reward=reward,
        response=f"response-{index}",
    )


class ContractDataSource:
    def __init__(self, n_samples_per_prompt: int = 2):
        self.n_samples_per_prompt = n_samples_per_prompt
        self.next_index = 0

    def get_examples(self, num_prompts: int) -> list[dict]:
        examples = []
        for _ in range(num_prompts):
            examples.append({"prompt": f"prompt-{self.next_index}", "metadata": {}})
            self.next_index += 1
        return examples

    def add_examples(self, examples: list[dict]):
        pass


def valid_rollout_function(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        ep = make_episode(0, reward=0.75)
        return RolloutFnEvalOutput(
            data={"contract_eval": [ep]},
            metrics={"source": "contract"},
        )

    episodes = []
    examples = data_source.get_examples(2)
    for group_index, example in enumerate(examples):
        for sample_index in range(2):
            ep = Episode(
                tokens=[group_index, sample_index, rollout_id],
                response=f"group-{group_index}-sample-{sample_index}",
                response_length=3,
                reward=float(group_index + sample_index),
            )
            episodes.append(ep)
    return RolloutFnTrainOutput(episodes=episodes, metrics={"source": "contract"})


def invalid_rollout_function(args, rollout_id, data_source, evaluation=False):
    ep = make_episode(0)
    ep.tokens = []
    return RolloutFnTrainOutput(episodes=[ep])


def assert_episode_contract(ep: Episode) -> None:
    assert isinstance(ep, Episode)
    assert isinstance(ep.tokens, list)
    assert all(isinstance(token, int) for token in ep.tokens)
    assert isinstance(ep.response, str)
    assert isinstance(ep.response_length, int)
    assert ep.reward is not None


def assert_train_rollout_contract(output: RolloutFnTrainOutput, n_samples_per_prompt: int) -> None:
    assert isinstance(output, RolloutFnTrainOutput)
    assert output.episodes
    for ep in output.episodes:
        assert_episode_contract(ep)


def assert_eval_rollout_contract(output: RolloutFnEvalOutput) -> None:
    assert isinstance(output, RolloutFnEvalOutput)
    assert output.data
    for dataset_name, episodes in output.data.items():
        assert isinstance(episodes, list)
        for ep in episodes:
            assert_episode_contract(ep)


def assert_rollout_function_signature_matches_default(fn) -> None:
    default_sig = inspect.signature(default_generate_rollout)
    candidate_sig = inspect.signature(fn)

    assert tuple(candidate_sig.parameters) == tuple(default_sig.parameters)

    for name, default_param in default_sig.parameters.items():
        candidate_param = candidate_sig.parameters[name]
        assert candidate_param.kind == default_param.kind
        assert candidate_param.default == default_param.default


def assert_rollout_function_matches_default_contract(fn) -> None:
    assert_rollout_function_signature_matches_default(fn)

    data_source = ContractDataSource()
    train_output = fn(None, 2, data_source, evaluation=False)
    eval_output = fn(None, 2, data_source, evaluation=True)

    assert_train_rollout_contract(train_output, n_samples_per_prompt=2)
    assert_eval_rollout_contract(eval_output)


def test_rollout_function_path_contract_supports_user_override():
    rollout_path = get_contract_path("ROLLOUT_FUNCTION_PATH", DEFAULT_ROLLOUT_FUNCTION_PATH)
    rollout_fn = load_function(rollout_path)

    assert_rollout_function_signature_matches_default(rollout_fn)

    if rollout_path != DEFAULT_ROLLOUT_FUNCTION_PATH:
        assert_rollout_function_matches_default_contract(rollout_fn)


def test_load_function_can_load_local_rollout_plugin():
    func = load_function(REFERENCE_ROLLOUT_FUNCTION_PATH)
    assert func.__name__ == valid_rollout_function.__name__
    assert func.__module__ == "plugin_contracts.test_plugin_rollout_contracts"


def test_default_rollout_signature_is_stable():
    default_sig = inspect.signature(default_generate_rollout)
    assert tuple(default_sig.parameters) == ("args", "rollout_id", "data_source", "evaluation")
    assert default_sig.parameters["evaluation"].default is False


def test_local_rollout_plugin_aligns_with_default_input_output_format():
    assert_rollout_function_matches_default_contract(valid_rollout_function)


def test_misaligned_rollout_plugin_is_rejected():
    with pytest.raises(AssertionError):
        assert_rollout_function_matches_default_contract(invalid_rollout_function)


if __name__ == "__main__":
    run_contract_test_file()
