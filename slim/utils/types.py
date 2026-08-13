# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - allows lightweight unit imports
    class _TorchStub:
        class dtype:
            pass

        class Size(tuple):
            pass

    torch = _TorchStub()


def _as_cpu_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


@dataclass
class Trajectory:
    """One contiguous generation span by the policy.

    Lifecycle:
      1. Created empty, then filled by one or more generation calls that append
         to the same span; sequence fields are Python lists.
      2. ``finalize_source_token_alignment()`` converts sequence fields to
         tensors and appends their terminal source-token slot.
      3. Flattened out of its episode, then packed and trained as one document.

    Source-token-aligned fields have ``len == len(token_ids)`` after
    finalization. Position ``i`` describes the prediction of ``token_ids[i+1]``.
    The final position has no prediction target and carries the field's neutral
    fill.
    """

    # Sequence state: prediction lists during generation, source-aligned tensors after finalization.
    token_ids: Any = field(default_factory=list)       # [int] → LongTensor
    loss_mask: Any | None = None                       # [int] → IntTensor
    rollout_log_probs: Any | None = None               # [float] → FloatTensor
    rollout_routed_experts: Any | None = None          # np.ndarray [prediction, layer, top_k] → IntTensor
    multimodal_inputs: dict[str, Any] | None = None
    # Non-token-aligned multimodal inputs from processor (concat dim=0):
    #   pixel_values: [num_vision_tokens, d] - image embeddings (concat dim=0)
    #   image_grid_thw: [num_images, 3] - image metadata (concat dim=0)
    #   pixel_values_videos: [num_vision_tokens, d] - video embeddings (concat dim=0)
    #   video_grid_thw: [num_videos, 3] - video metadata (concat dim=0)

    reward: float | None = None
    text: str | None = None                            # decode of all tokens in this span
    generated_text: str | None = None                  # decode of targets selected by active prediction slots

    # Training targets, set after advantage estimation.
    advantages: Any | None = None
    values: Any | None = None
    value_targets: Any | None = None

    # Stamped by the flattener. A padding trajectory has episode_index None.
    episode_index: int | None = None
    group_index: int | None = None
    loss_weight: float = 0.0

    # --- Rollout finalization ---

    def _finalize_prediction_field(
        self,
        name: str,
        value,
        *,
        dtype: torch.dtype,
        fill: float | int,
    ):
        if value is None:
            return None
        tensor = _as_cpu_tensor(value, dtype=dtype)
        if tensor.ndim == 0:
            raise ValueError(f"{name} must have a sequence dimension")

        token_count = len(self.token_ids)
        prediction_count = max(token_count - 1, 0)
        if tensor.shape[0] != prediction_count:
            raise ValueError(
                f"{name} length {tensor.shape[0]} must match prediction count {prediction_count}"
            )
        if token_count == 0:
            return tensor
        padding = torch.full((1, *tensor.shape[1:]), fill, dtype=dtype)
        return torch.cat((tensor, padding), dim=0)

    def _source_token_tensor(
        self,
        name: str,
        value,
        *,
        dtype: torch.dtype,
        fill: float | int,
    ):
        if value is None:
            return None
        tensor = _as_cpu_tensor(value, dtype=dtype)
        if tensor.ndim == 0 or tensor.shape[0] != len(self.token_ids):
            raise ValueError(
                f"{name} length {tensor.shape[0] if tensor.ndim else 0} "
                f"must match token count {len(self.token_ids)}"
            )
        if len(self.token_ids) and not torch.all(tensor[-1] == fill):
            raise ValueError(f"{name} terminal source-token slot must equal {fill}")
        return tensor

    def finalize_source_token_alignment(self) -> None:
        """Finalize rollout fields as CPU tensors aligned with source tokens."""
        if self.loss_mask is None:
            raise ValueError("loss_mask must be present")
        if any(target is not None for target in (self.advantages, self.values, self.value_targets)):
            raise ValueError("training targets must not be present before source-token finalization")

        self.token_ids = _as_cpu_tensor(self.token_ids, dtype=torch.long)
        self.loss_mask = self._finalize_prediction_field(
            "loss_mask",
            self.loss_mask,
            dtype=torch.int,
            fill=0,
        )
        self.rollout_log_probs = self._finalize_prediction_field(
            "rollout_log_probs",
            self.rollout_log_probs,
            dtype=torch.float32,
            fill=0.0,
        )
        self.rollout_routed_experts = self._finalize_prediction_field(
            "rollout_routed_experts",
            self.rollout_routed_experts,
            dtype=torch.int32,
            fill=0,
        )

    # --- Derived values ---

    @property
    def response_length(self) -> int:
        """Count source positions whose predictions contribute to training."""
        if self.loss_mask is None:
            return max(len(self.token_ids) - 1, 0)
        return int(sum(self.loss_mask))

    def set_train_targets(self, advantages, *, values=None, value_targets=None) -> None:
        """Set source-token-aligned training targets after rollout processing."""
        if advantages is None:
            raise ValueError("advantages must be present")
        if not isinstance(self.token_ids, torch.Tensor) or not isinstance(self.loss_mask, torch.Tensor):
            raise RuntimeError("trajectory must be source-token finalized before setting training targets")

        self.advantages = self._source_token_tensor(
            "advantages",
            advantages,
            dtype=torch.float32,
            fill=0.0,
        )
        self.values = self._source_token_tensor(
            "values",
            values,
            dtype=torch.float32,
            fill=0.0,
        )
        self.value_targets = self._source_token_tensor(
            "value_targets",
            value_targets,
            dtype=torch.float32,
            fill=0.0,
        )


@dataclass
class Episode:
    """One complete problem-solving attempt, from prompt to final answer.

    An episode holds one or more :class:`Trajectory` objects. Each trajectory is
    one contiguous generation span by the policy; trajectories carry no ordering
    or dependency relation between them. The append-only rollout produces a
    single trajectory, while interleaved thinking, sub-agent dispatch, and
    context compression produce several.
    """

    # Raw dataset row — rollout/RM functions read whatever columns they need
    example: dict = field(default_factory=dict)
    trajectories: list[Trajectory] = field(default_factory=list)
    # Shorthand for one scalar scoring the whole attempt; broadcast onto every
    # trajectory by the flattener. Setting this and Trajectory.reward is an error.
    reward: float | None = None

    # Identity, assigned at the rollout/train boundary.
    episode_index: int | None = None
    group_index: int | None = None

    # Rollout-time control and bookkeeping.
    generate_function_path: str | None = None
    session_id: str | None = None
    max_tokens: int = 0
    non_generation_time: float = 0.0
    sampling_seed: int | None = None                   # per-sample seed under deterministic inference

    # Status tracking
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: str = Status.PENDING

    @classmethod
    def from_example(cls, example: dict) -> "Episode":
        return cls(example=dict(example), trajectories=[Trajectory()])

    @property
    def trajectory(self) -> Trajectory:
        """The sole generation span of an append-only attempt."""
        if len(self.trajectories) != 1:
            raise ValueError(
                f"episode has {len(self.trajectories)} trajectories; read `trajectories` instead"
            )
        return self.trajectories[0]

    @property
    def has_multimodal(self) -> bool:
        return any(self.example.get(k) for k in ("images", "videos", "audios"))

    @property
    def token_count(self) -> int:
        return sum(len(trajectory.token_ids) for trajectory in self.trajectories)

    def finalize_source_token_alignment(self) -> None:
        for trajectory in self.trajectories:
            trajectory.finalize_source_token_alignment()

    # --- Shared helpers ---

    def update_status_from_finish_reason(self, finish_reason: str):
        match finish_reason:
            case "length":
                self.status = Episode.Status.TRUNCATED
            case "abort":
                self.status = Episode.Status.ABORTED
            case "stop":
                self.status = Episode.Status.COMPLETED

    def get_reward_value(self) -> float:
        """One scalar scoring the attempt, whichever level carries the reward."""
        if self.reward is not None:
            return self.reward
        rewards = [trajectory.reward for trajectory in self.trajectories]
        return sum(rewards) / len(rewards)


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
