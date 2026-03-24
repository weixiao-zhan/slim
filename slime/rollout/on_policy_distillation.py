import aiohttp
import torch

from slime.utils.types import Episode


async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, episodes: list[Episode], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response
    2. Trims them to match the response length
    3. Stores them in episode.teacher_log_probs for OPD KL penalty computation
    4. Sets scalar rewards (0.0 for pure distillation)
    """
    raw_rewards = [ep.reward for ep in episodes]
    response_lengths = [ep.response_length for ep in episodes]

    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for ep, t_log_probs in zip(episodes, teacher_log_probs, strict=False):
        ep.teacher_log_probs = t_log_probs

    # Set scalar rewards for GRPO/PPO advantage estimator
    for ep in episodes:
        ep.reward = 0.0
