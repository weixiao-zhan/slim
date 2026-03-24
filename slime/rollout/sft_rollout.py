import logging

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Episode

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


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

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)

    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    examples = data_source.get_examples(args.rollout_batch_size)

    episodes = []
    for i, example in enumerate(examples):
        messages = example.get("prompt", "")
        tools = example.get("metadata", {}).get("tools", None)

        token_ids, loss_mask = MASK_GENERATOR.get_loss_mask(messages, tools=tools)
        edge_loss_mask = loss_mask[1:] if len(token_ids) > 1 else []

        ep = Episode(
            tokens=token_ids,
            reward=0.0,
            loss_mask=edge_loss_mask,
        )
        ep.ensure_edge_alignment()
        episodes.append(ep)

        if i == 0 and not SAMPLE_PRINTED:
            logger.info(
                f"sft_rollout::generate_rollout example data: {ep=} (raw){messages=} (raw){token_ids=} (raw){loss_mask=}"
            )
            SAMPLE_PRINTED = True

    return RolloutFnTrainOutput(episodes=episodes)
