import pytest
import torch

import slim.backends.megatron.batch_layout as batch_layout
from slim.backends.megatron.batch_layout import (
    PackedLayout,
    build_packed_layout,
    packed_sequence_alignment,
)
from slim.utils.types import Episode


def _episode(
    tokens,
    *,
    loss_mask=None,
    rollout_log_probs=None,
    routed_experts=None,
    multimodal_inputs=None,
):
    episode = Episode(
        tokens=torch.tensor(tokens, dtype=torch.long),
        loss_mask=None if loss_mask is None else torch.tensor(loss_mask),
        rollout_log_probs=(
            None if rollout_log_probs is None else torch.tensor(rollout_log_probs, dtype=torch.float32)
        ),
        rollout_routed_experts=routed_experts,
        multimodal_inputs=multimodal_inputs,
    )
    return episode


def _named_ranges(ranges):
    return {item.name: (item.span.start, item.span.stop) for item in ranges}


def test_packs_each_trajectory_to_tp_sp_cp_alignment_and_aligns_edges():
    first = _episode([10, 11, 12], loss_mask=[0, 1], rollout_log_probs=[-1.0, -2.0])
    first._advantages = [1.0, 2.0]
    first._returns = [11.0, 12.0]
    second = _episode([20, 21, 22, 23, 24], loss_mask=[1, 1, 0, 1], rollout_log_probs=[-3, -4, -5, -6])
    second._advantages = [3.0, 4.0, 5.0, 6.0]
    second._returns = [13.0, 14.0, 15.0, 16.0]

    layout = build_packed_layout(
        [first, second],
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        pad_token_id=0,
    )

    assert isinstance(layout, PackedLayout)
    assert layout.alignment == 4
    assert layout.logical_lengths == (3, 5)
    assert layout.physical_lengths == (4, 8)
    assert layout.tokens.tolist() == [[10, 11, 12, 0, 20, 21, 22, 23, 24, 0, 0, 0]]
    assert layout.labels.tolist() == [[11, 12, -100, -100, 21, 22, 23, 24, -100, -100, -100, -100]]
    assert layout.loss_mask.tolist() == [[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]]
    assert layout.token_mask.tolist() == [[True, True, True, False, True, True, True, True, True, False, False, False]]
    assert layout.edge_mask.tolist() == [[True, True, False, False, True, True, True, True, False, False, False, False]]
    assert layout.edge_features["advantages"].tolist() == [
        [1, 2, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0]
    ]
    assert layout.edge_features["returns"].tolist() == [
        [11, 12, 0, 0, 13, 14, 15, 16, 0, 0, 0, 0]
    ]
    assert layout.edge_features["rollout_log_probs"].tolist() == [
        [-1, -2, 0, 0, -3, -4, -5, -6, 0, 0, 0, 0]
    ]
    assert layout.cu_seqlens_q.tolist() == [0, 3, 8]
    assert layout.cu_seqlens_q_padded.tolist() == [0, 4, 12]
    assert layout.total_tokens == 12

    assert [trajectory.episode_index for trajectory in layout.trajectories] == [0, 1]
    assert layout.trajectories[0].logical_tokens == batch_layout.PackedRange(0, 3)
    assert layout.trajectories[0].tokens == batch_layout.PackedRange(0, 3)
    assert layout.trajectories[0].physical_tokens == batch_layout.PackedRange(0, 4)
    assert layout.trajectories[1].logical_tokens == batch_layout.PackedRange(3, 8)
    assert layout.trajectories[1].tokens == batch_layout.PackedRange(4, 9)
    assert layout.trajectories[1].physical_tokens == batch_layout.PackedRange(4, 12)


def test_alignment_uses_lcm_and_rejects_sp_without_tp():
    assert packed_sequence_alignment(
        tensor_model_parallel_size=4,
        context_parallel_size=2,
        sequence_parallel=True,
    ) == 8
    assert packed_sequence_alignment(
        tensor_model_parallel_size=6,
        context_parallel_size=4,
        sequence_parallel=True,
    ) == 24
    assert packed_sequence_alignment(
        tensor_model_parallel_size=8,
        context_parallel_size=3,
        sequence_parallel=False,
    ) == 6
    with pytest.raises(ValueError, match="sequence_parallel requires"):
        packed_sequence_alignment(
            tensor_model_parallel_size=1,
            context_parallel_size=2,
            sequence_parallel=True,
        )


def test_mixed_text_vision_batch_preserves_episode_media_and_placeholder_order():
    image = _episode(
        [1, 99, 99, 2],
        loss_mask=[0, 0, 1],
        multimodal_inputs={
            "pixel_values": torch.tensor([[10.0], [11.0]]),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        },
    )
    text = _episode([3, 4, 5], loss_mask=[0, 1])
    image_and_video = _episode(
        [99, 6, 98, 98, 7],
        loss_mask=[0, 0, 0, 1],
        multimodal_inputs={
            "pixel_values": torch.tensor([[12.0]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
            "pixel_values_videos": torch.tensor([[20.0], [21.0]]),
            "video_grid_thw": torch.tensor([[1, 1, 2]]),
            "second_per_grid_ts": torch.tensor([0.5]),
        },
    )

    layout = build_packed_layout(
        [image, text, image_and_video],
        placeholder_token_ids={"image": 99, "video": 98},
    )

    assert layout.logical_lengths == (4, 3, 5)
    assert layout.physical_lengths == (4, 3, 5)
    assert [trajectory.episode_index for trajectory in layout.trajectories] == [0, 1, 2]
    assert _named_ranges(layout.trajectories[0].media_ranges)["pixel_values"] == (0, 2)
    assert _named_ranges(layout.trajectories[1].media_ranges)["pixel_values"] == (2, 2)
    assert _named_ranges(layout.trajectories[2].media_ranges)["pixel_values"] == (2, 3)
    assert _named_ranges(layout.trajectories[0].media_ranges)["pixel_values_videos"] == (0, 0)
    assert _named_ranges(layout.trajectories[1].media_ranges)["pixel_values_videos"] == (0, 0)
    assert _named_ranges(layout.trajectories[2].media_ranges)["pixel_values_videos"] == (0, 2)

    assert [
        (item.name, item.span.start, item.span.stop)
        for item in layout.trajectories[0].placeholder_ranges
    ] == [("image", 1, 3)]
    assert layout.trajectories[1].placeholder_ranges == ()
    assert [
        (item.name, item.span.start, item.span.stop)
        for item in layout.trajectories[2].placeholder_ranges
    ] == [("image", 7, 8), ("video", 9, 11)]

    visual_kwargs = layout.visual_inputs.normalized_for_model()
    assert visual_kwargs["pixel_values"].tolist() == [[10.0], [11.0], [12.0]]
    assert visual_kwargs["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 1]]
    assert visual_kwargs["pixel_values_videos"].tolist() == [[20.0], [21.0]]
    assert visual_kwargs["video_grid_thw"].tolist() == [[1, 1, 2]]
    assert visual_kwargs["second_per_grid_ts"].tolist() == [0.5]


def test_routing_replay_is_token_aligned_and_padding_has_no_payload():
    first_routes = torch.tensor(
        [
            [[1], [2]],
            [[3], [4]],
        ],
        dtype=torch.int32,
    )
    second_routes = torch.tensor([[[5], [6]]], dtype=torch.int32)
    first = _episode([10, 11, 12], loss_mask=[1, 1], routed_experts=first_routes)
    second = _episode([20, 21], loss_mask=[1], routed_experts=second_routes)

    layout = build_packed_layout([first, second], context_parallel_size=2)

    assert layout.rollout_routed_experts.shape == (1, 8, 2, 1)
    assert layout.rollout_routed_experts[0, 0:2].tolist() == first_routes.tolist()
    assert layout.rollout_routed_experts[0, 2:4].eq(0).all()
    assert layout.rollout_routed_experts[0, 4].tolist() == second_routes[0].tolist()
    assert layout.rollout_routed_experts[0, 5:8].eq(0).all()


def test_visual_inputs_require_provider_placeholder_ids_and_matching_tokens():
    episode = _episode(
        [1, 2],
        loss_mask=[1],
        multimodal_inputs={
            "pixel_values": torch.tensor([[1.0]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        },
    )

    with pytest.raises(ValueError, match="no 'image' placeholder token ID"):
        build_packed_layout([episode])
    with pytest.raises(ValueError, match="no matching 'image' placeholder tokens"):
        build_packed_layout([episode], placeholder_token_ids={"image": 99})


def test_bridge_row_builder_is_the_preferred_packing_entrypoint(monkeypatch):
    calls = []

    def spy_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return batch_layout._fallback_build_mcore_thd_sequence_batch_from_rows(*args, **kwargs)

    monkeypatch.setattr(batch_layout, "_get_sequence_batch_builder", lambda: spy_builder)
    layout = build_packed_layout([_episode([1, 2], loss_mask=[1])])

    assert layout.total_tokens == 2
    assert len(calls) == 1
    assert calls[0][1]["pad_to_multiple_of"] == 1
    assert calls[0][1]["sequence_tensor_pad_values"] == {
        "token_mask": 0,
        "edge_mask": 0,
    }


def test_partially_present_edge_features_are_rejected():
    first = _episode([1, 2], loss_mask=[1], rollout_log_probs=[-1])
    second = _episode([3, 4], loss_mask=[1])

    with pytest.raises(ValueError, match="rollout_log_probs"):
        build_packed_layout([first, second])
