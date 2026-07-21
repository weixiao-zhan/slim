# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import os


def get_nvidia_ld_library_path() -> dict[str, str]:
    """Auto-detect nvidia lib paths from pip-installed packages for LD_LIBRARY_PATH.

    Returns a dict suitable for unpacking into Ray runtime_env env_vars.
    """
    # `nvidia` is a namespace package (no __init__.py), so __file__ is None;
    # resolve its location via __path__ instead.
    try:
        nvidia_bases = list(__import__("nvidia").__path__)
    except (ImportError, AttributeError):
        return {}

    lib_dirs = [d for base in nvidia_bases for d in glob.glob(os.path.join(base, "*/lib"))]
    if not lib_dirs:
        return {}

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": ":".join(lib_dirs) + (f":{existing}" if existing else "")}
