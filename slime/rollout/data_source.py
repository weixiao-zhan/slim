import abc
import logging
import os

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_examples(self, num_prompts: int) -> list[dict]:
        """Return num_prompts raw dataset examples (dicts)."""

    @abc.abstractmethod
    def add_examples(self, examples: list[dict]):
        """Re-queue examples (e.g. aborted prompts) back into the source."""

    @abc.abstractmethod
    def save(self, rollout_id):
        """Save the state of the data source."""

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """Load the state of the data source."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Length of the data source."""


class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_offset = 0

        if args.rollout_global_dataset:
            self.dataset = Dataset(
                args.prompt_data,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_examples(self, num_prompts: int) -> list[dict]:
        if self.dataset is not None:
            if self.sample_offset + num_prompts <= len(self.dataset):
                examples = self.dataset.samples[self.sample_offset : self.sample_offset + num_prompts]
                self.sample_offset += num_prompts
            else:
                examples = self.dataset.samples[self.sample_offset :]
                remaining = num_prompts - len(examples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                examples = examples + self.dataset.samples[:remaining]
                self.sample_offset = remaining
        else:
            examples = [{} for _ in range(num_prompts)]

        return examples

    def add_examples(self, examples: list[dict]):
        raise RuntimeError(f"Cannot add examples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load data source state from {path}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def __len__(self) -> int:
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = _pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_examples(self, num_prompts: int) -> list[dict]:
        examples = self._get_from_buffer(num_prompts)
        remaining = num_prompts - len(examples)
        if remaining > 0:
            examples += super().get_examples(num_prompts=remaining)
        return examples

    def _get_from_buffer(self, num_prompts: int) -> list[dict]:
        if len(self.buffer) == 0 or num_prompts == 0:
            return []
        return self.buffer_filter(self.args, None, self.buffer, num_prompts)

    def add_examples(self, examples: list[dict]):
        if not examples:
            return
        self.buffer.extend(examples)

    def get_buffer_length(self):
        return len(self.buffer)


def _pop_first(args, rollout_id, buffer: list[dict], num_prompts: int) -> list[dict]:
    num_to_pop = min(len(buffer), num_prompts)
    examples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return examples
