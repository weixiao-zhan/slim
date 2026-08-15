# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands calculator rollout with one trajectory per model invocation.

Strands owns the agent loop: it formats every request, parses tool calls out of the
response, and runs the calculator. This module only points Strands at SGLang's chat
completions endpoint, hands it one dataset row, and keeps the token ids the engine ran.

Dataset columns this example reads:

| Column | Type | Content |
|--------|------|---------|
| `prompt` | `str` | The user turn. |
| `system` | `str` or absent | System prompt for the agent. |
| `images` | `list[PIL.Image]` or absent | Images of the user turn, in order. |
| `label` | any | Ground truth, read by the reward function rather than by rollout. |
"""

from __future__ import annotations

import io
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import pybase64
from openai.types.chat import ChatCompletion
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import EventLoopException, MaxTokensReachedException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from strands_tools.calculator import calculator

from slim.rollout.sglang_rollout import GenerateState, get_model_url
from slim.utils.http_utils import post
from slim.utils.processing_utils import decode_tensor_envelopes
from slim.utils.types import Episode, Trajectory

_MAX_AGENT_TURNS = 2
_MAX_TOKENS_PER_TURN = 16384

# How the engine ended the attempt's last call is what the attempt amounts to.
_STATUS_BY_FINISH_REASON = {
    "stop": Episode.Status.COMPLETED,
    "length": Episode.Status.TRUNCATED,
    "tool_calls": Episode.Status.TRUNCATED,  # the turn limit cut the tool loop short
}


class _RequestAborted(Exception):
    """slim aborted this in-flight request, so the attempt is dropped and resampled."""


def _image_block(image) -> ContentBlock:
    """Strands carries an image as raw bytes and re-encodes it per provider."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"image": {"format": "png", "source": {"bytes": buffer.getvalue()}}}


class SGLangChatModel(OpenAIModel):
    """Strands' OpenAI provider served by SGLang, recording each call as a trajectory."""

    def __init__(self, state: GenerateState, episode: Episode) -> None:
        # Everything here is a `/v1/chat/completions` body field; Strands merges
        # `params` into the request it builds.
        params: dict[str, Any] = {
            "temperature": state.args.rollout_temperature,
            "max_completion_tokens": min(_MAX_TOKENS_PER_TURN, episode.max_tokens),
            "chat_template_kwargs": state.chat_template_kwargs,
            # Token-in-token-out: keep the ids and logprobs the engine ran.
            "logprobs": True,
            "return_prompt_token_ids": True,
            "return_meta_info": True,
        }
        if episode.sampling_seed is not None:
            # The OpenAI endpoint spells the engine's `sampling_seed` as `seed`.
            params["seed"] = episode.sampling_seed
        if episode.has_multimodal:
            # Training reuses the tensors the engine ran, rather than reprocessing the media.
            params["return_processor_outputs"] = True
        if state.routing_replay_shape is not None:
            # One call is one whole trajectory, so every position is in scope.
            params["return_routed_experts"] = True
            params["routed_experts_start_len"] = 0
        super().__init__(model_id=state.args.hf_checkpoint, stream=False, params=params)

        self.state = state
        self.episode = episode
        self.finish_reason: str | None = None
        self.url = get_model_url(state.args, "actor", "/v1/chat/completions")
        self.headers = (
            {"X-SMG-Routing-Key": episode.session_id}
            if state.args.router_policy == "consistent_hashing" and episode.session_id
            else None
        )

    def _record_trajectory(self, choice: Any) -> None:
        """Keep one call's prompt and generation as a standalone trajectory."""
        meta_info = choice.meta_info
        entries = meta_info["output_token_logprobs"]  # [logprob, token_id, text]
        prompt_predictions = len(choice.prompt_token_ids) - 1
        trajectory = Trajectory(
            token_ids=choice.prompt_token_ids + [entry[1] for entry in entries],
            loss_mask=[0] * prompt_predictions + [1] * len(entries),
            rollout_log_probs=[0.0] * prompt_predictions + [entry[0] for entry in entries],
        )

        if self.episode.has_multimodal:
            trajectory.multimodal_inputs = decode_tensor_envelopes(meta_info["processor_outputs"])

        if self.state.routing_replay_shape is not None:
            experts = np.frombuffer(pybase64.b64decode(meta_info["routed_experts"]), dtype=np.int32)
            trajectory.rollout_routed_experts = experts.copy().reshape(-1, *self.state.routing_replay_shape)

        self.finish_reason = choice.finish_reason
        self.episode.trajectories.append(trajectory)

    async def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Serve one Strands model call from SGLang and record what it ran."""
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        payload = await post(self.url, request, headers=self.headers)
        if payload["choices"][0]["finish_reason"] == "abort":
            # `abort` is the engine's own finish reason, outside the OpenAI schema below.
            raise _RequestAborted
        response = ChatCompletion.model_validate(payload)
        self._record_trajectory(response.choices[0])
        for event in self._format_non_streaming_response(response):
            yield event


async def generate(state: GenerateState, episode: Episode, evaluation: bool = False) -> Episode:
    """Run one Strands calculator attempt and retain every model call it makes."""
    del evaluation
    content: list[ContentBlock] = [_image_block(image) for image in episode.example.get("images") or []]
    content.append({"text": episode.example["prompt"]})

    model = SGLangChatModel(state, episode)
    agent = Agent(
        model=model,
        tools=[calculator],
        system_prompt=episode.example.get("system"),
        callback_handler=None,
    )
    try:
        await agent.invoke_async([{"role": "user", "content": content}], limits={"turns": _MAX_AGENT_TURNS})
    except MaxTokensReachedException:
        # The engine stopped that call at the completion cap; it is already recorded.
        pass
    except (_RequestAborted, EventLoopException) as error:
        cause = getattr(error, "original_exception", error)
        if not isinstance(cause, _RequestAborted):
            raise
        episode.status = Episode.Status.ABORTED
        return episode

    if not episode.trajectories:
        raise RuntimeError("The calculator rollout recorded no model invocation")
    episode.status = _STATUS_BY_FINISH_REASON.get(model.finish_reason, Episode.Status.FAILED)
    return episode
