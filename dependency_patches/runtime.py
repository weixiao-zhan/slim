# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA IPC process access."""

import os
import subprocess
from pathlib import Path


def apply() -> None:
    """Set kernel.yama.ptrace_scope=0 (needed for Torch CUDA IPC weight sync).

    Skipped with a warning if sudo is unavailable or requires a password.
    """
    if os.environ.get("SKIP_PTRACE_SYSCTL"):
        print("  Skipped: kernel.yama.ptrace_scope (SKIP_PTRACE_SYSCTL set)")
        return
    if Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip() == "0":
        print("  Skipped: kernel.yama.ptrace_scope is already 0")
        return
    try:
        subprocess.run(
            ["sudo", "-n", "sysctl", "-w", "kernel.yama.ptrace_scope=0"],
            check=True,
            capture_output=True,
        )
        print("  Applied: kernel.yama.ptrace_scope=0")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"  Warning: could not set kernel.yama.ptrace_scope=0 ({e}). "
            "Torch CUDA IPC weight sync (pidfd_getfd) may fail. "
            "Set it manually: sudo sysctl -w kernel.yama.ptrace_scope=0"
        )
