# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from slim.utils.types import Episode
from tests.trim_nemo_grad_norm_rollout import trim_record


def test_trim_record_preserves_prompt_and_slices_prediction_fields():
    episode = Episode(
        tokens=torch.arange(8),
        loss_mask=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        rollout_log_probs=torch.arange(7, dtype=torch.float32),
        rollout_routed_experts=torch.arange(14).reshape(7, 2),
        reward=1.0,
        text="full",
        generated_text="full",
        status=Episode.Status.COMPLETED,
    )

    trimmed, token_count = trim_record(episode.__dict__, max_response_tokens=2)

    assert token_count == 6
    assert trimmed["tokens"].tolist() == [0, 1, 2, 3, 4, 5]
    assert trimmed["loss_mask"].tolist() == [0, 0, 0, 1, 1]
    assert trimmed["rollout_log_probs"].tolist() == [0, 1, 2, 3, 4]
    assert trimmed["rollout_routed_experts"].shape == (5, 2)
    assert trimmed["reward"] == 1.0
    assert trimmed["text"] is None
    assert trimmed["generated_text"] is None
    assert trimmed["status"] == Episode.Status.TRUNCATED
