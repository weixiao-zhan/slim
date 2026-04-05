import logging

import torch

from slim.rollout.base_types import RolloutFnTrainOutput
from slim.utils.processing_utils import load_processor, load_tokenizer
from slim.utils.types import Episode

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
SAMPLE_PRINTED = False


def _get_assistant_mask(tokenizer, messages, tools=None):
    """Use tokenizer.apply_chat_template with return_assistant_tokens_mask to get loss mask."""
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tools=tools,
    )
    token_ids = out["input_ids"]
    loss_mask = out["assistant_masks"]
    return token_ids, loss_mask


def _get_assistant_mask_multimodal(processor, messages, multimodal_inputs, tools=None):
    """For VLMs: use processor for tokenization, tokenizer for assistant mask, then align."""
    prompt_text = processor.apply_chat_template(messages, tokenize=False, tools=tools)
    mm = {k: v for k, v in (multimodal_inputs or {}).items() if v}
    processor_output = processor(text=prompt_text, **mm, return_tensors="pt")
    token_ids = processor_output["input_ids"][0].tolist()

    multimodal_train_inputs = {
        k: v
        for k, v in processor_output.items()
        if k not in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor)
    } or None

    # Get assistant mask from text-only tokenizer
    text_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            text_parts = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            text_messages.append({"role": msg["role"], "content": " ".join(text_parts)})
        else:
            text_messages.append(msg)

    _, text_mask = _get_assistant_mask(processor.tokenizer, text_messages, tools=tools)

    # Align: multimodal tokens are longer due to image/video placeholders expanded
    diff = len(token_ids) - len(text_mask)
    assert diff >= 0, (
        f"input_ids (length={len(token_ids)}) is shorter than text loss_mask (length={len(text_mask)}). "
        f"Please check if processor and tokenizer tokenization are consistent."
    )
    loss_mask = [0] * diff + text_mask

    return token_ids, loss_mask, multimodal_train_inputs


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """SFT rollout: tokenize examples and produce Episodes with loss masks.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout
        data_source: the data source to get examples
        evaluation: bool, whether the rollout is for evaluation

    Returns:
        RolloutFnTrainOutput with flat list of Episodes
    """
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    examples = data_source.get_examples(args.rollout_batch_size)

    episodes = []
    for i, example in enumerate(examples):
        ep = Episode.from_example(example)

        if isinstance(ep.prompt, list) and PROCESSOR is not None:
            token_ids, loss_mask, ep.multimodal_train_inputs = _get_assistant_mask_multimodal(
                PROCESSOR, ep.prompt, ep.multimodal_inputs, tools=ep.tools
            )
        else:
            token_ids, loss_mask = _get_assistant_mask(TOKENIZER, ep.prompt, tools=ep.tools)

        ep.tokens = token_ids
        ep.loss_mask = loss_mask[1:] if len(token_ids) > 1 else []
        ep.reward = 0.0
        ep.ensure_edge_alignment()
        episodes.append(ep)

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(f"sft_rollout::generate_rollout example data: {ep=}")
            SAMPLE_PRINTED = True

    return RolloutFnTrainOutput(episodes=episodes)
