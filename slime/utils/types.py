from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Episode:
    """A single generated episode, the universal data unit flowing from rollout to training.

    The generate function receives raw dataset examples (dicts) and produces
    a flat list of Episodes. Episodes are ordered by group: for n_samples_per_prompt=K,
    episodes[i*K : (i+1)*K] all come from the same prompt. GRPO-style normalization
    just reshapes rewards by (-1, K).

    Fields marked 'training-essential' are consumed by the training step.
    Fields marked 'metrics/logging' are only used for logging and evaluation.
    """

    # === Training-essential ===
    tokens: list[int] = field(default_factory=list)  # full prompt + response token ids
    response_length: int = 0
    reward: float = 0.0  # raw scalar reward (normalization is done at batch level)
    loss_mask: list[int] | None = None  # len == response_length; None means all-ones

    # === Optional training fields ===
    rollout_log_probs: list[float] | None = None  # per-token log probs from rollout
    multimodal_train_inputs: dict[str, Any] | None = None  # processed tensors (pixel_values, etc.)
    teacher_log_probs: list[float] | None = None  # for on-policy distillation
    rollout_routed_experts: Any | None = None  # MoE routing replay data
    train_metadata: dict | None = None  # per-episode metadata consumed by training

    # === Metrics / logging (not used by training math) ===
    response: str = ""  # decoded response text, for logging (repetition, etc.)
    truncated: bool = False  # whether generation hit max length
    non_generation_time: float = 0.0  # time in non-generation steps (reward, tools, etc.)

    @property
    def effective_response_length(self) -> int:
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def get_reward_value(self, args) -> float:
        """Get scalar reward, selecting by key if reward is a dict."""
        return self.reward


# ---------------------------------------------------------------------------
# Sample: rollout-internal working state.
# This class is used only inside generate functions (e.g. sglang_rollout.py)
# to accumulate state during async generation. The generate function converts
# Samples to Episodes before returning.
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Internal working state during rollout generation.

    Not part of the public interface — generate functions should convert
    Sample → Episode before returning results.
    """

    # prompt / input
    prompt: str | list[dict[str, str]] = ""
    label: str | None = None
    tools: list[dict] | None = None
    multimodal_inputs: dict[str, Any] | None = None  # raw multimodal data (PIL images)
    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    session_id: str | None = None

    # accumulated during generation
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    rollout_routed_experts: Any | None = None
    multimodal_train_inputs: dict[str, Any] | None = None
    teacher_log_probs: list[float] | None = None
    weight_versions: list[str] = field(default_factory=list)
    non_generation_time: float = 0.0
    train_metadata: dict | None = None

    # status tracking
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: str = Status.PENDING

    def update_from_meta_info(self, args, meta_info: dict):
        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED

    def to_episode(self, args) -> Episode:
        """Convert this Sample to an Episode for training/eval consumption."""
        reward_value = self.reward if not getattr(args, "reward_key", None) else self.reward[args.reward_key]
        return Episode(
            tokens=self.tokens,
            response_length=self.response_length,
            reward=float(reward_value) if reward_value is not None else 0.0,
            loss_mask=self.loss_mask,
            rollout_log_probs=self.rollout_log_probs,
            multimodal_train_inputs=self.multimodal_train_inputs,
            teacher_log_probs=self.teacher_log_probs,
            rollout_routed_experts=self.rollout_routed_experts,
            train_metadata=self.train_metadata,
            response=self.response,
            truncated=(self.status == Sample.Status.TRUNCATED),
            non_generation_time=self.non_generation_time,
        )

    def get_reward_value(self, args) -> float:
        return self.reward if not getattr(args, "reward_key", None) else self.reward[args.reward_key]


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
