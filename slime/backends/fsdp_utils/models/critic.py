"""Critic model wrapper for PPO value function estimation.

Wraps a pretrained causal LM by replacing its lm_head with a scalar
value head (Linear(hidden_size, 1)), producing per-token value estimates.
"""

import logging

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def create_critic_model(
    checkpoint_path: str,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    init_context=None,
    model_cls=None,
):
    """Create a critic model from a pretrained causal LM checkpoint.

    Loads the base model, removes the language modeling head, and attaches
    a value head that outputs a single scalar per token position.

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory.
        attn_implementation: Attention implementation override.
        trust_remote_code: Whether to trust remote code in HF models.
        init_context: Context manager factory for weight initialization
            (e.g., init_empty_weights for non-rank-0 processes).
        model_cls: Model class to use (e.g., AutoModelForImageTextToText
            for VLMs). Defaults to AutoModelForCausalLM.
    """
    if model_cls is None:
        model_cls = AutoModelForCausalLM

    kwargs = {"trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code)
    hidden_size = config.hidden_size

    if init_context is not None:
        with init_context():
            model = model_cls.from_pretrained(checkpoint_path, **kwargs)
    else:
        model = model_cls.from_pretrained(checkpoint_path, **kwargs)

    # Replace the lm_head with a value head.
    # Zero-init for stable PPO startup (V(s) ≈ 0 before critic warmup).
    old_lm_head = model.lm_head
    backbone_dtype = old_lm_head.weight.dtype
    value_head = nn.Linear(hidden_size, 1, bias=False, dtype=backbone_dtype)
    nn.init.zeros_(value_head.weight)
    model.lm_head = value_head

    logger.info(
        f"Created critic model: replaced lm_head "
        f"({old_lm_head.in_features}x{old_lm_head.out_features}) "
        f"with value_head ({hidden_size}x1)"
    )

    return model
