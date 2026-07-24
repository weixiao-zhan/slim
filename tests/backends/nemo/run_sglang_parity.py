# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare NeMo target log probabilities with a consolidated SGLang model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta", type=float, default=0.5)
    parser.add_argument("--mean-delta", type=float, default=0.1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    return parser.parse_args()


def _input_log_probs(response: dict) -> list[float]:
    values = response["meta_info"]["input_token_logprobs"]
    if not values or values[0][0] is not None:
        raise RuntimeError("SGLang prompt log probabilities do not contain the expected empty first position")
    return [float(item[0]) for item in values[1:]]


def main() -> None:
    cli = _arguments()
    artifact = torch.load(cli.artifact, map_location="cpu", weights_only=True)
    checkpoint = artifact["consolidated_checkpoint"]
    documents = artifact["documents"]
    nemo_log_probs = artifact["log_probs"].float()

    import sglang

    engine = sglang.Engine(
        model_path=checkpoint,
        tp_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        mem_fraction_static=cli.mem_fraction_static,
        disable_cuda_graph=True,
    )
    try:
        responses = engine.generate(
            input_ids=documents,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
        )
    finally:
        engine.shutdown()

    if isinstance(responses, dict):
        responses = [responses]
    sglang_log_probs = torch.tensor(
        [value for response in responses for value in _input_log_probs(response)],
        dtype=torch.float32,
    )
    if sglang_log_probs.shape != nemo_log_probs.shape:
        raise RuntimeError(
            f"log-probability shape mismatch: SGLang={tuple(sglang_log_probs.shape)} "
            f"NeMo={tuple(nemo_log_probs.shape)}"
        )

    delta = (sglang_log_probs - nemo_log_probs).abs()
    result = {
        "checkpoint": checkpoint,
        "tokens": int(delta.numel()),
        "max_delta": delta.max().item(),
        "mean_delta": delta.mean().item(),
        "nemo_mean": nemo_log_probs.mean().item(),
        "sglang_mean": sglang_log_probs.mean().item(),
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)
    failures = []
    if result["max_delta"] > cli.max_delta:
        failures.append(f"max_delta={result['max_delta']:.7g} exceeds {cli.max_delta:.7g}")
    if result["mean_delta"] > cli.mean_delta:
        failures.append(f"mean_delta={result['mean_delta']:.7g} exceeds {cli.mean_delta:.7g}")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
