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
      1. Created from dataset example — fields are Python lists.
      2. Mutated in-place during async generation + reward.
      3. ``freeze()`` converts tokens/loss_mask/rollout_log_probs to tensors.
      4. Consumed by normalization, packing, and training — all tensor ops.

    Edge-aligned invariants (len == len(tokens) - 1):
    - ``loss_mask[i]``: whether predicting ``tokens[i+1]`` contributes to loss.
      Prompt edges are 0, generated edges are 1.
    - ``rollout_log_probs[i]``: log-prob of ``tokens[i+1]`` under the rollout policy.
    - ``rollout_routed_experts[i]``: top-k MoE expert ids selected at the router
    """

    # Raw dataset row — rollout/RM functions read whatever columns they need
    example: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    session_id: str | None = None

    # Sequence state — lists pre-freeze, tensors post-freeze
    tokens: Any = field(default_factory=list)          # [int] → LongTensor
    loss_mask: Any | None = None                       # [int] → IntTensor
    reward: float | None = None
    rollout_log_probs: Any | None = None               # [float] → FloatTensor
    rollout_routed_experts: Any | None = None          # np.ndarray [num_edges, num_layers, top_k] → IntTensor
    multimodal_inputs: dict[str, Any] | None = None
    # Non-token-aligned multimodal inputs from processor (concat dim=0):
    #   pixel_values: [num_vision_tokens, d] - image embeddings (concat dim=0)
    #   image_grid_thw: [num_images, 3] - image metadata (concat dim=0)
    #   pixel_values_videos: [num_vision_tokens, d] - video embeddings (concat dim=0)
    #   video_grid_thw: [num_videos, 3] - video metadata (concat dim=0)
    text: str | None = None                            # decode of all tokens
    generated_text: str | None = None                  # decode of loss_mask==1 tokens
    non_generation_time: float = 0.0
    max_tokens: int = 0
    _sampling_params: dict[str, Any] | None = None     # transient rollout request params; cleared by freeze()

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

    # --- Pre-freeze helpers (list phase) ---

    @property
    def num_edges(self) -> int:
        return max(len(self.tokens) - 1, 0)

    def ensure_edge_alignment(self) -> None:
        """Validate / materialize loss_mask. Call before freeze()."""
        edge_len = self.num_edges
        if self.loss_mask is None:
            self.loss_mask = [1] * edge_len
        if len(self.loss_mask) != edge_len:
            raise ValueError(f"loss_mask length {len(self.loss_mask)} != num_edges {edge_len}")
        if self.rollout_log_probs is not None and len(self.rollout_log_probs) != edge_len:
            raise ValueError(f"rollout_log_probs length {len(self.rollout_log_probs)} != num_edges {edge_len}")
        if self.rollout_routed_experts is not None and len(self.rollout_routed_experts) != edge_len:
            raise ValueError(f"rollout_routed_experts length {len(self.rollout_routed_experts)} != num_edges {edge_len}")

    def freeze(self) -> None:
        """Convert sequence fields to tensors. Call once after generation + RM."""
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        if self.loss_mask is not None:
            self.loss_mask = torch.tensor(self.loss_mask, dtype=torch.int)
        if self.rollout_log_probs is not None:
            self.rollout_log_probs = torch.tensor(self.rollout_log_probs, dtype=torch.float32)
        if self.rollout_routed_experts is not None and not isinstance(self.rollout_routed_experts, torch.Tensor):
            self.rollout_routed_experts = torch.from_numpy(self.rollout_routed_experts).to(torch.int32)
        self._sampling_params = None

    # --- Properties (work on both lists and tensors) ---

    @property
    def response_length(self) -> int:
        """Count of generated edges (loss_mask == 1)."""
        if self.loss_mask is None:
            return self.num_edges
        return int(sum(self.loss_mask))

    def get_generated_token_ids(self) -> list[int]:
        """Extract token ids where loss_mask == 1.

        Returns a plain list regardless of pre/post-freeze state.
        """
        if self.loss_mask is None or not len(self.tokens):
            return []
        tokens, mask = self.tokens, self.loss_mask
        if hasattr(mask, "bool"):  # tensor path
            return tokens[1:][mask.bool()].tolist()
        return [tokens[i + 1] for i, m in enumerate(mask) if m]

    # --- Shared helpers ---

    def update_status_from_finish_reason(self, finish_reason: str):
        match finish_reason:
            case "length":
                self.status = Episode.Status.TRUNCATED
            case "abort":
                self.status = Episode.Status.ABORTED
            case "stop":
                self.status = Episode.Status.COMPLETED

    def get_reward_value(self, args) -> float:
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

