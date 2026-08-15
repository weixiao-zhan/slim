# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import torch
from PIL import Image

from slim.utils.processing_utils import (
    decode_tensor_from_b64_envelope,
    encode_tensor_to_b64_envelope,
    pil_to_data_url,
)


def test_pil_to_data_url_encodes_rgb_png():
    image = Image.new("RGBA", (2, 3), color=(10, 20, 30, 40))

    data_url = pil_to_data_url(image)

    prefix, encoded = data_url.split(",", maxsplit=1)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert prefix == "data:image/png;base64"
    assert decoded.mode == "RGB"
    assert decoded.size == (2, 3)


def test_tensor_envelope_round_trip():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    decoded = decode_tensor_from_b64_envelope(encode_tensor_to_b64_envelope(tensor))

    torch.testing.assert_close(decoded, tensor)
