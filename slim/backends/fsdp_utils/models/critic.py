"""Critic model wrapper for PPO value function estimation.

Wraps a pretrained causal LM by replacing its lm_head with a scalar
value head (Linear(hidden_size, 1)), producing per-token value estimates.
"""

import logging

import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def create_critic_model(
    checkpoint_path: str,
    init_context=None,
    model_cls=None,
    **from_pretrained_kwargs,
):
    """Create a critic model from a pretrained causal LM checkpoint.

    Loads the base model, removes the language modeling head, and attaches
    a value head that outputs a single scalar per token position.

    Args:
        checkpoint_path: Path to HuggingFace checkpoint directory.
        init_context: Context manager factory for weight initialization
            (e.g., init_empty_weights for non-rank-0 processes).
        model_cls: Model class to use (e.g., AutoModelForImageTextToText
            for VLMs). Defaults to AutoModelForCausalLM.
        **from_pretrained_kwargs: Additional kwargs passed to from_pretrained
            (e.g., trust_remote_code, attn_implementation, torch_dtype).
    """
    if model_cls is None:
        model_cls = AutoModelForCausalLM

    if init_context is not None:
        with init_context():
            model = model_cls.from_pretrained(checkpoint_path, **from_pretrained_kwargs)
    else:
        model = model_cls.from_pretrained(checkpoint_path, **from_pretrained_kwargs)

    # Replace the lm_head with a value head.
    # Zero-init for stable PPO startup (V(s) ≈ 0 before critic warmup).
    old_lm_head = model.lm_head
    hidden_size = old_lm_head.in_features
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
