# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import nvidia
import pytest

from slim.utils.env_utils import get_nvidia_ld_library_path

NUM_GPUS = 0


@pytest.mark.unit
def test_nvidia_library_path_isolates_wheels_from_host_cuda(tmp_path, monkeypatch):
    wheel_lib = tmp_path / "cudnn" / "lib"
    wheel_lib.mkdir(parents=True)
    monkeypatch.setattr(nvidia, "__path__", [str(tmp_path)])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/usr/local/cuda/lib")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")

    assert get_nvidia_ld_library_path() == {"LD_LIBRARY_PATH": str(wheel_lib)}


@pytest.mark.unit
def test_nvidia_library_path_without_wheel_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(nvidia, "__path__", [str(tmp_path)])

    assert get_nvidia_ld_library_path() == {}
