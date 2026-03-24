import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Episode

__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, episodes: list[Episode], **kwargs):
    """Dynamic filter: drop groups where all rewards are identical (zero std)."""
    rewards = [episode.get_reward_value(args) for episode in episodes]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
