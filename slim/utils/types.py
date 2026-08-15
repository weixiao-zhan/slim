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


@dataclass
class Episode:
    """A single rollout/training record.

    Lifecycle:
      1. Created from dataset example; fields are Python lists.
      2. Mutated in-place during async generation + reward.
      3. ``finalize_source_token_alignment()`` converts sequence fields to tensors
         and appends their terminal source-token slot.
      4. Consumed by normalization, packing, and training using tensor operations.

    Source-token-aligned fields have ``len == len(tokens)`` after finalization.
    Position ``i`` describes the prediction of ``tokens[i+1]``. The final
    position has no prediction target and contains the field's neutral fill.
    """

    # Raw dataset row — rollout/RM functions read whatever columns they need
    example: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    session_id: str | None = None

    # Sequence state: prediction lists during generation, source-aligned tensors after finalization.
    tokens: Any = field(default_factory=list)          # [int] → LongTensor
    loss_mask: Any | None = None                       # [int] → IntTensor
    reward: float | None = None
    rollout_log_probs: Any | None = None               # [float] → FloatTensor
    rollout_routed_experts: Any | None = None          # np.ndarray [prediction, layer, top_k] → IntTensor
    multimodal_inputs: dict[str, Any] | None = None
    # Non-token-aligned multimodal inputs from processor (concat dim=0):
    #   pixel_values: [num_vision_tokens, d] - image embeddings (concat dim=0)
    #   image_grid_thw: [num_images, 3] - image metadata (concat dim=0)
    #   pixel_values_videos: [num_vision_tokens, d] - video embeddings (concat dim=0)
    #   video_grid_thw: [num_videos, 3] - video metadata (concat dim=0)
    text: str | None = None                            # decode of all tokens
    generated_text: str | None = None                  # decode of targets selected by active prediction slots
    non_generation_time: float = 0.0
    max_tokens: int = 0
    _sampling_params: dict[str, Any] | None = None     # transient rollout params; cleared during finalization

    # Training targets, populated after rollout.
    episode_index: int | None = None
    advantages: Any | None = None
    values: Any | None = None
    value_targets: Any | None = None

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
        return cls(example=dict(example))

    @property
    def has_multimodal(self) -> bool:
        return any(self.example.get(k) for k in ("images", "videos", "audios"))

    # --- Rollout finalization ---

    @staticmethod
    def _as_cpu_tensor(value, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

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
        tensor = self._as_cpu_tensor(value, dtype=dtype)
        if tensor.ndim == 0:
            raise ValueError(f"{name} must have a sequence dimension")

        token_count = len(self.tokens)
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
        tensor = self._as_cpu_tensor(value, dtype=dtype)
        if tensor.ndim == 0 or tensor.shape[0] != len(self.tokens):
            raise ValueError(
                f"{name} length {tensor.shape[0] if tensor.ndim else 0} "
                f"must match token count {len(self.tokens)}"
            )
        if len(self.tokens) and not torch.all(tensor[-1] == fill):
            raise ValueError(f"{name} terminal source-token slot must equal {fill}")
        return tensor

    def finalize_source_token_alignment(self) -> None:
        """Finalize rollout fields as CPU tensors aligned with source tokens."""
        if self.loss_mask is None:
            raise ValueError("loss_mask must be present")
        if any(target is not None for target in (self.advantages, self.values, self.value_targets)):
            raise ValueError("training targets must not be present before source-token finalization")

        self.tokens = self._as_cpu_tensor(self.tokens, dtype=torch.long)
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
        self._sampling_params = None

    # --- Derived values ---

    @property
    def response_length(self) -> int:
        """Count source positions whose predictions contribute to training."""
        if self.loss_mask is None:
            return max(len(self.tokens) - 1, 0)
        return int(sum(self.loss_mask))

    def set_train_targets(self, advantages, *, values=None, value_targets=None) -> None:
        """Set source-token-aligned training targets after rollout processing."""
        if advantages is None:
            raise ValueError("advantages must be present")
        if not isinstance(self.tokens, torch.Tensor) or not isinstance(self.loss_mask, torch.Tensor):
            raise RuntimeError("episode must be source-token finalized before setting training targets")

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
        return self.reward


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
