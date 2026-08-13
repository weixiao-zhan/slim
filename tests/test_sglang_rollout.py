# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

import slim.rollout.sglang_rollout as sglang_rollout
from slim.rollout.sglang_rollout import _prepare_episode_tokens, decode_generated_text
from slim.utils.types import Episode, Trajectory


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if not kwargs["tokenize"]:
            return "rendered prompt"
        return {"input_ids": [11, 12, 13]} if kwargs.get("return_dict", True) else [11, 12, 13]

    def decode(self, token_ids):
        self.calls.append(("decode", token_ids))
        return ",".join(str(token_id) for token_id in token_ids)


class Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("the tokenizer owns the chat template")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.tensor([[21, 22, 23]]),
            "attention_mask": torch.ones(1, 3),
            "pixel_values": torch.ones(2, 4),
        }


def state():
    return SimpleNamespace(
        tokenizer=Tokenizer(),
        processor=Processor(),
        chat_template_kwargs={},
    )


def test_text_chat_uses_tokenizer_template_when_processor_is_available():
    rollout_state = state()
    episode = Episode.from_example({"prompt": [{"role": "user", "content": "hello"}]})

    asyncio.run(_prepare_episode_tokens(rollout_state, episode))

    assert episode.trajectory.token_ids == [11, 12, 13]
    assert rollout_state.tokenizer.calls[0][1]["tokenize"] is True
    assert rollout_state.tokenizer.calls[0][1]["return_dict"] is False
    assert rollout_state.processor.calls == []


def test_multimodal_chat_renders_with_tokenizer_before_processing():
    rollout_state = state()
    episode = Episode.from_example(
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]}],
            "images": [object()],
        }
    )

    asyncio.run(_prepare_episode_tokens(rollout_state, episode))

    assert rollout_state.tokenizer.calls[0][1]["tokenize"] is False
    assert rollout_state.processor.calls[0]["text"] == "rendered prompt"
    assert episode.trajectory.token_ids == [21, 22, 23]
    assert set(episode.trajectory.multimodal_inputs) == {"pixel_values"}


def test_decode_generated_text_requires_rollout_prediction_alignment(monkeypatch):
    rollout_state = state()
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: rollout_state)
    trajectory = Trajectory(token_ids=[10, 11, 12, 13], loss_mask=[0, 1, 1])

    assert decode_generated_text(SimpleNamespace(), trajectory) == "12,13"

    trajectory.loss_mask.append(0)
    with pytest.raises(ValueError, match="zip\\(\\) argument 2 is longer"):
        decode_generated_text(SimpleNamespace(), trajectory)


def test_generate_requests_and_records_full_routing_replay(monkeypatch):
    requests = []
    routed_experts = np.arange(4, dtype=np.int32).reshape(2, 2, 1)

    async def fake_post(url, payload, max_retries=60, headers=None):
        del max_retries
        requests.append((url, payload, headers))
        return {
            "meta_info": {
                "output_token_logprobs": [[-0.25, 13, "13"]],
                "routed_experts": pybase64.b64encode(routed_experts.tobytes()).decode("ascii"),
                "finish_reason": {"type": "stop"},
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    rollout_state = SimpleNamespace(
        args=SimpleNamespace(
            router_ip="127.0.0.1",
            router_port=30000,
            router_policy="round_robin",
        ),
        routing_replay_shape=(2, 1),
    )
    episode = Episode.from_example({"prompt": "hello"})
    episode.max_tokens = 8
    episode._sampling_params = {"temperature": 1.0}
    episode.trajectory.token_ids = [11, 12]
    episode.trajectory.loss_mask = [0]
    episode.trajectory.rollout_log_probs = [0.0]

    result = asyncio.run(sglang_rollout.generate(rollout_state, episode))

    assert result.status == Episode.Status.COMPLETED
    assert result.trajectory.token_ids == [11, 12, 13]
    assert np.array_equal(result.trajectory.rollout_routed_experts, routed_experts)
    _, payload, headers = requests[0]
    assert payload["return_routed_experts"] is True
    assert payload["routed_experts_start_len"] == 0
    assert headers is None


def test_generate_recaptures_routing_replay_from_the_first_unheld_position(monkeypatch):
    requests = []
    held = np.arange(4, dtype=np.int32).reshape(2, 2, 1)
    fresh = np.arange(100, 106, dtype=np.int32).reshape(3, 2, 1)

    async def fake_post(url, payload, max_retries=60, headers=None):
        del max_retries, headers
        requests.append(payload)
        return {
            "meta_info": {
                "output_token_logprobs": [[-0.25, 15, "15"], [-0.5, 16, "16"]],
                "routed_experts": pybase64.b64encode(fresh.tobytes()).decode("ascii"),
                "finish_reason": {"type": "stop"},
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    rollout_state = SimpleNamespace(
        args=SimpleNamespace(
            router_ip="127.0.0.1",
            router_port=30000,
            router_policy="round_robin",
        ),
        routing_replay_shape=(2, 1),
    )
    episode = Episode.from_example({"prompt": "hello"})
    episode.max_tokens = 8
    episode._sampling_params = {"temperature": 1.0}
    # A previous call captured two predictions; the caller then appended a user turn.
    episode.trajectory.token_ids = [11, 12, 13, 14]
    episode.trajectory.loss_mask = [0, 1, 0]
    episode.trajectory.rollout_log_probs = [0.0, -0.1, 0.0]
    episode.trajectory.rollout_routed_experts = held

    result = asyncio.run(sglang_rollout.generate(rollout_state, episode))

    assert requests[0]["routed_experts_start_len"] == 2
    assert result.trajectory.token_ids == [11, 12, 13, 14, 15, 16]
    assert np.array_equal(
        result.trajectory.rollout_routed_experts,
        np.concatenate((held, fresh)),
    )
