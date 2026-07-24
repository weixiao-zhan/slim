# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import torch

from slim.rollout.sglang_rollout import _prepare_episode_tokens
from slim.utils.types import Episode


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if not kwargs["tokenize"]:
            return "rendered prompt"
        return {"input_ids": [11, 12, 13]} if kwargs.get("return_dict", True) else [11, 12, 13]


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

    assert episode.tokens == [11, 12, 13]
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
    assert episode.tokens == [21, 22, 23]
    assert set(episode.multimodal_inputs) == {"pixel_values"}
