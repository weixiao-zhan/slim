# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


def patch_file(
    path: Path,
    replacements: list[tuple[str, str]],
    log_reason: str = "",
) -> bool:
    """Apply exact text replacements and fail if an upstream anchor changed."""
    text = path.read_text()
    original = text
    label = log_reason or path.name
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            print(f"  No match: {label}; anchor not found: {old[:80]!r}")
            raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    if text != original:
        # Replace the file instead of writing through it: uv hardlinks installed files to its cache.
        tmp = path.with_name(f"{path.name}.slim-patch")
        tmp.write_text(text)
        os.replace(tmp, path)
        print(f"  Applied: {label}")
    else:
        print(f"  Skipped (already applied): {label}")
    return True
