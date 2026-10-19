# Source: https://github.com/THUDM/slime
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def install_paths() -> None:
    current = Path(__file__).resolve()
    sys.path.insert(0, str(current.parent))
    sys.path.insert(0, str(current.parents[1]))
    sys.path.insert(0, str(current.parents[2]))


def install_stubs(*, with_sglang_router: bool = False, with_transformers: bool = False) -> None:
    if "ray" not in sys.modules and importlib.util.find_spec("ray") is None:
        ray_mod = types.ModuleType("ray")
        ray_mod._private = types.SimpleNamespace(
            services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        )
        sys.modules["ray"] = ray_mod

    if with_sglang_router and "sglang_router" not in sys.modules and importlib.util.find_spec("sglang_router") is None:
        mod = types.ModuleType("sglang_router")
        mod.__version__ = "0.2.3"
        sys.modules["sglang_router"] = mod

    if with_transformers and "transformers" not in sys.modules and importlib.util.find_spec("transformers") is None:
        mod = types.ModuleType("transformers")
        mod.AutoTokenizer = type(
            "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: object())}
        )
        mod.AutoProcessor = type(
            "AutoProcessor",
            (),
            {"from_pretrained": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))},
        )
        mod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
        mod.ProcessorMixin = type("ProcessorMixin", (), {})
        sys.modules["transformers"] = mod

