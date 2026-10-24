# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands calculator rollout with one trajectory per model invocation.

Strands owns the agent loop: it formats every request, parses tool calls out of the
response, and runs the calculator. This module only points Strands at SGLang's chat
completions endpoint, hands it one dataset row, and keeps the token ids the engine ran.

Each calculator call runs in a child process with a wall-clock limit, so a slow SymPy
computation cannot stall the rollout.

Dataset columns this example reads:

| Column | Type | Content |
|--------|------|---------|
| `prompt` | `str` | The user turn. |
| `system` | `str` or absent | System prompt for the agent. |
| `images` | `list[PIL.Image]` or absent | Images of the user turn, in order. |
| `label` | any | Ground truth, read by the reward function rather than by rollout. |
"""

from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing
from collections.abc import AsyncGenerator
from multiprocessing.connection import Connection
from typing import Any

import torch
from openai.types.chat import ChatCompletion
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import EventLoopException, MaxTokensReachedException
from strands.types.streaming import StreamEvent
from strands.types.tools import AgentTool, ToolChoice, ToolGenerator, ToolSpec, ToolUse
from strands_tools.calculator import calculator

from slim.rollout.sglang_rollout import GenerateState, get_model_url
from slim.utils.http_utils import post
from slim.utils.processing_utils import decode_tensor_envelopes
from slim.utils.types import Episode, Trajectory

# Generation cap of one model call.
_MAX_TOKENS_PER_TURN = 12 * 1024
# Cap on model calls in one attempt. Strands recurses once per tool round, so an uncapped
# run of short tool calls exhausts the Python stack before it fills the context.
_MAX_TURNS = 10
# Wall-clock limit of one calculator call. The model gets an error result for a call that runs longer.
_TOOL_TIMEOUT_SECONDS = 10

# A child forks from a server that has already imported this module, so a call starts in milliseconds.
# The fork server itself starts on the first call.
_PROCESSES = multiprocessing.get_context("forkserver")
_PROCESSES.set_forkserver_preload([__name__])


class _RequestAborted(Exception):
    """slim aborted this in-flight request, so the attempt is dropped and resampled."""


def _image_block(image) -> ContentBlock:
    """Strands carries an image as raw bytes and re-encodes it per provider."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return {"image": {"format": "png", "source": {"bytes": buffer.getvalue()}}}


def _calculate(tool_use: ToolUse, sender: Connection) -> None:
    """Run one calculator call in the child process and send its tool result."""
    # The calculator logs a deprecation warning on every call.
    logging.getLogger("strands_tools.calculator").setLevel(logging.ERROR)

    async def events() -> list[Any]:
        return [event async for event in calculator.stream(tool_use, {})]

    # The last event carries the tool result.
    sender.send(asyncio.run(events())[-1].tool_result)


class _TimeLimitedCalculator(AgentTool):
    """Strands' calculator, stopped when a call runs past `_TOOL_TIMEOUT_SECONDS`.

    SymPy can run for hours on a short expression such as `factorial(10**10)`. A thread
    cannot be stopped, so each call runs in a child process that is killed at the limit.
    """

    @property
    def tool_name(self) -> str:
        return calculator.tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        return calculator.tool_spec

    @property
    def tool_type(self) -> str:
        return calculator.tool_type

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        del invocation_state, kwargs
        try:
            result = await self._run_in_child(tool_use)
        except TimeoutError:
            error = f"the call ran past the {_TOOL_TIMEOUT_SECONDS} second limit and was stopped"
        except EOFError:
            error = "the calculator process exited without a result"
        else:
            error = None
        if error is not None:
            result = {"toolUseId": tool_use["toolUseId"], "status": "error", "content": [{"text": f"Error: {error}"}]}
        # The executor takes the last raw value a tool yields as its result.
        yield result

    @staticmethod
    async def _run_in_child(tool_use: ToolUse) -> Any:
        receiver, sender = _PROCESSES.Pipe(duplex=False)
        with receiver:
            with sender:
                process = _PROCESSES.Process(target=_calculate, args=(tool_use, sender), daemon=True)
                # The first start launches the fork server, which takes seconds.
                await asyncio.to_thread(process.start)
            # Only the child holds the sending end now, so the pipe becomes readable when
            # the child sends its result or exits.
            loop = asyncio.get_running_loop()
            ready = asyncio.Event()
            loop.add_reader(receiver.fileno(), ready.set)
            try:
                async with asyncio.timeout(_TOOL_TIMEOUT_SECONDS):
                    await ready.wait()
                return receiver.recv()
            finally:
                loop.remove_reader(receiver.fileno())
                process.kill()
                process.join()


class SGLangChatModel(OpenAIModel):
    """Strands' OpenAI provider served by SGLang, recording each call as a trajectory."""

    def __init__(self, state: GenerateState, episode: Episode) -> None:
        # Everything here is a `/v1/chat/completions` body field; Strands merges
        # `params` into the request it builds. Near the end of the context, the engine
        # lowers `max_tokens` to the tokens left, since slim enables `allow_auto_truncate`.
        params: dict[str, Any] = {
            "temperature": state.args.rollout_temperature,
            "max_tokens": _MAX_TOKENS_PER_TURN,
            "chat_template_kwargs": state.chat_template_kwargs,
            # Token-in-token-out: keep the ids and logprobs the engine ran.
            "logprobs": True,
            "return_prompt_token_ids": True,
            "return_meta_info": True,
        }
        if episode.sampling_seed is not None:
            # The OpenAI endpoint spells the engine's `sampling_seed` as `seed`.
            params["seed"] = episode.sampling_seed
        if state.args.use_score_centering is not None:
            # Score centering reads the sampler top-K from `meta_info["output_top_logprobs"]`.
            params["top_logprobs"] = state.args.use_score_centering
        if episode.has_multimodal:
            # Training reuses the tensors the engine ran, rather than reprocessing the media.
            params["return_processor_outputs"] = True
        if state.args.routing_replay_shape is not None:
            # One call is one whole trajectory, so every position is in scope.
            params["return_routed_experts"] = True
            params["routed_experts_start_len"] = 0
        super().__init__(model_id=state.args.hf_checkpoint, stream=False, params=params)

        self.state = state
        self.episode = episode
        self.url = get_model_url(state.args, "actor", "/v1/chat/completions")
        self.headers = (
            {"X-SMG-Routing-Key": episode.session_id}
            if state.args.router_policy == "consistent_hashing" and episode.session_id
            else None
        )

    @classmethod
    def _format_regular_messages(cls, messages: list[Message], **kwargs: Any) -> list[dict[str, Any]]:
        """Return assistant reasoning to the engine as `reasoning_content`.

        Strands drops reasoning blocks on the Chat Completions API; the chat template
        re-renders them, which keeps thinking interleaved across tool calls.
        """
        formatted_messages = []
        for message in messages:
            reasoning = "".join(
                content["reasoningContent"]["reasoningText"]["text"]
                for content in message["content"]
                if "reasoningContent" in content
            )
            contents = [content for content in message["content"] if "reasoningContent" not in content]
            formatted = super()._format_regular_messages([{**message, "content": contents}], **kwargs)
            if reasoning:
                formatted[0]["reasoning_content"] = reasoning
                formatted[0].setdefault("content", "")
            formatted_messages.extend(formatted)
        return formatted_messages

    def _record_trajectory(self, choice: Any) -> None:
        """Keep one call's prompt and generation as a standalone trajectory."""
        # The per-token arrays arrive in meta_info as tensor envelopes.
        meta_info = choice.meta_info
        prompt_token_ids = decode_tensor_envelopes(meta_info["prompt_token_ids"]).tolist()
        output_logprobs = decode_tensor_envelopes(meta_info["output_token_logprobs"])
        output_token_ids = output_logprobs["token_ids"].tolist()
        prompt_predictions = len(prompt_token_ids) - 1
        trajectory = Trajectory(
            token_ids=prompt_token_ids + output_token_ids,
            loss_mask=[0] * prompt_predictions + [1] * len(output_token_ids),
            rollout_log_probs=[0.0] * prompt_predictions + output_logprobs["logprobs"].tolist(),
        )

        if (top_k := self.state.args.use_score_centering) is not None:
            # Inert rows for the prompt, then one row per generated token.
            top_logprobs = decode_tensor_envelopes(meta_info["output_top_logprobs"])
            trajectory.rollout_topk_log_probs = torch.cat(
                (torch.zeros((prompt_predictions, top_k), dtype=torch.float32), top_logprobs["logprobs"])
            )
            trajectory.rollout_topk_token_ids = torch.cat(
                (torch.zeros((prompt_predictions, top_k), dtype=torch.int32), top_logprobs["token_ids"])
            )

        if self.episode.has_multimodal:
            trajectory.multimodal_inputs = decode_tensor_envelopes(meta_info["processor_outputs"])

        if self.state.args.routing_replay_shape is not None:
            experts = decode_tensor_envelopes(meta_info["routed_experts"])
            trajectory.rollout_routed_experts = experts.reshape(-1, *self.state.args.routing_replay_shape)

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
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice, **kwargs)
        payload = await post(self.url, request, headers=self.headers)
        if payload["choices"][0]["finish_reason"] == "abort":
            # `abort` is the engine's own finish reason, outside the OpenAI schema below.
            raise _RequestAborted
        if payload["usage"]["prompt_tokens"] < payload["choices"][0]["meta_info"]["prompt_token_ids"]["shape"][0]:
            # The conversation outgrew the context and the engine cut the prompt to fit. What the
            # engine ran then matches no prompt the agent sent, so the call is dropped.
            raise MaxTokensReachedException("The conversation outgrew the context")
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
        tools=[_TimeLimitedCalculator()],
        system_prompt=episode.example.get("system"),
        callback_handler=None,
    )
    try:
        result = await agent.invoke_async([{"role": "user", "content": content}], limits={"turns": _MAX_TURNS})
        # At the turn limit the last call asked for a tool, so the attempt ends without an answer.
        episode.status = Episode.Status.TRUNCATED if result.stop_reason == "limit_turns" else Episode.Status.COMPLETED
    except MaxTokensReachedException:
        # A call hit the token cap or the context limit, or the conversation outgrew the context.
        episode.status = Episode.Status.TRUNCATED
    except (_RequestAborted, EventLoopException) as error:
        cause = getattr(error, "original_exception", error)
        if not isinstance(cause, _RequestAborted):
            raise
        episode.status = Episode.Status.ABORTED
        return episode

    if not episode.trajectories:
        raise RuntimeError("The tool-use rollout recorded no model invocation")
    return episode
