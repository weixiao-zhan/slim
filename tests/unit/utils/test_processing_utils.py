# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch

from slim.utils.processing_utils import (
    decode_tensor_from_b64_envelope,
    encode_tensor_to_b64_envelope,
)


def test_tensor_envelope_round_trip():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    decoded = decode_tensor_from_b64_envelope(encode_tensor_to_b64_envelope(tensor))

    torch.testing.assert_close(decoded, tensor)
