# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass/fail checker for a single sweep or test run.

Usage: python tests/sanity_check.py <log_file> <expect_actor:0|1> <expect_critic:0|1>

Exit code 0 = PASS, 1 = FAIL. Prints a one-line verdict with extracted stats.

Criteria (smoke + numerical sanity):
  * job did not fail / no CUDA OOM / no weight-version mismatch
  * >= 1 actor "train step" with finite loss   (when expect_actor)
  * >= 1 "critic step" with finite loss          (when expect_critic)
  * a rollout reward value was logged            (non-degenerate pipeline)
  * >= 2 actor weight updates pushed to rollout  (when expect_actor) so that
    subsequent (not just the first) weight-update cycles are exercised
"""

import ast
import math
import re
import sys


def _finite_losses(text: str, prefix: str, loss_key: str) -> list[float]:
    """Extract loss values from lines like '<prefix> step N: {...}'."""
    losses = []
    for m in re.finditer(rf"{re.escape(prefix)} step \d+: (\{{.*?\}})", text):
        try:
            d = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            continue
        if loss_key in d and isinstance(d[loss_key], (int, float)):
            losses.append(float(d[loss_key]))
    return losses


def main() -> int:
    log_file, expect_actor, expect_critic = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1"
    with open(log_file, errors="replace") as f:
        text = f.read()

    problems = []

    # The ray job's own status line is the authoritative success/failure signal.
    if re.search(r"Job '\S+' failed", text):
        problems.append("job-failed")
    if "CUDA out of memory" in text:
        problems.append("oom")
    # The actor asserts engine == updater weight version after each sync (ci_test).
    if "Weight version mismatch" in text:
        problems.append("weight-version-mismatch")

    actor_losses = _finite_losses(text, "train/actor", "train/actor/loss")
    critic_losses = _finite_losses(text, "train/critic", "train/critic/value_loss")

    # Each actor.update_weights() is @timer-wrapped -> one "Timer update_weights end" per sync.
    weight_updates = len(re.findall(r"Timer update_weights end", text))

    if expect_actor:
        if not actor_losses:
            problems.append("no-actor-steps")
        elif not all(math.isfinite(x) for x in actor_losses):
            problems.append("actor-loss-nonfinite")
        # >= 2 proves subsequent weight-update cycles (not just the initial sync) are sound.
        if weight_updates < 2:
            problems.append(f"too-few-weight-updates({weight_updates})")
    if expect_critic:
        if not critic_losses:
            problems.append("no-critic-steps")
        elif not all(math.isfinite(x) for x in critic_losses):
            problems.append("critic-loss-nonfinite")

    # Rollout reward proves the generate -> reward -> train pipeline ran.
    reward = re.search(r"rollout/raw_reward['\"]?:?\s*([-\d.eE]+)", text)
    if reward is None and "raw_reward" not in text:
        problems.append("no-rollout-reward")

    verdict = "PASS" if not problems else "FAIL"
    stats = (
        f"actor_steps={len(actor_losses)} critic_steps={len(critic_losses)} "
        f"weight_updates={weight_updates} "
        f"actor_loss0={actor_losses[0] if actor_losses else 'na'} "
        f"critic_loss0={critic_losses[0] if critic_losses else 'na'}"
    )
    print(f"{verdict} | {stats}" + (f" | problems={','.join(problems)}" if problems else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
