#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch installed dependencies via `uv run python patch_dependencies.py`."""

from dependency_patches import nemo_automodel, runtime, sglang


def main() -> int:
    nemo_automodel.apply()
    sglang.apply()
    runtime.apply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
