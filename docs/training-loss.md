# Training Loss

slim trains a policy $\pi_\theta$ using on-policy RL.
Each rollout produces a batch of sequences; for each sequence $i$, the rollout policy $\pi_\text{rollout}$ generates tokens $x_{i,t}$ and records per-token log-probabilities.
A reward function assigns a scalar reward $R_i$ to the sequence.
The training loop then computes advantages $A_{i,t}$ and updates $\pi_\theta$ via a policy gradient objective.

Denote $\pi(x_{i,t})$: shorthand for $\pi(x_{i,t} | x_{i,<t})$;
$\pi_\theta$: the training policy; 
$\pi_\text{old}$: the policy used for generation running on FSDP backend;
$\pi_\text{rollout}$: the policy used for generation running on SGLang; 
$\pi_\text{ref}$: frozen reference.

The RL policy gradient objective writes:

$$\nabla_\theta L_{\text{pg}} = \mathbb{E}_{y \sim \pi_\text{old}} \left[ \sum_{t=1}^{T} f(r_{i,t},\, \hat{A}_{i,t}) \cdot \nabla_\theta \log\pi_\theta(x_{i,t}) \right]$$

where $r_{i,t} = \pi_\theta(x_{i,t}) / \pi_\text{old}(x_{i,t})$ is the importance ratio.

The two design axes are:
1. **Advantage estimator** (`--advantage-estimator`): how $\hat{A}_{i,t}$ is computed from rewards.
2. **Policy surrogate** (`--policy-surrogate`): the function $f(r_{i,t}, \hat{A}_{i,t})$ that combines the ratio and advantage (clipping, truncation, etc.).

## Advantage Estimators

`--advantage-estimator` controls how per-token advantages $A_{i,t}$ are derived from rewards.

### `ppo_gae`

The full form.
Requires a critic model (set via `--critic-num-gpus`, `--critic-load`) that produces per-token value predictions $V(x_{i,\le t})$.
GAE computes advantages with discount $\gamma$ (`--gamma`, default 1.0) and lambda $\lambda$ (`--lambd`, default 1.0):

$$\delta_{i,t} = R_{i,t} + \gamma V_{i,t+1} - V_{i,t}$$
$$A_{i,t} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{i,t+l}$$

The reward is placed at the last response token; all others have $R_{i,t} = 0$.
For long sequences, GAE uses a chunked parallel prefix scan (chunk size 128) to reduce sequential depth.

### `grpo` (default)

Group-Relative Policy Optimization simplifies PPO-GAE by eliminating the critic entirely.
Instead of learning a value baseline, it uses the other sequences from the same prompt group as the baseline.
Rewards are normalized within each prompt group (the $n$ sequences sampled from the same prompt):

$$A_{i,t} = \frac{R_i - \text{mean}(R_{1..n})}{\text{std}(R_{1..n})}$$


### `gspo`

Uses the same advantage as `grpo`, but redefines the importance ratio fed to the policy surrogate.
Instead of the per-token ratio $r_{i,t} = \pi_\theta(x_{i,t}) / \pi_\text{old}(x_{i,t})$, GSPO uses a sequence-level ratio:

$$r_i = \exp\left( \frac{1}{|T_i|} \sum_t (\log \pi_\theta(x_{i,t}) - \log \pi_\text{old}(x_{i,t})) \cdot \mathbf{1}[\text{loss\_mask}_{i,t}] \right)$$

This single scalar is expanded to every token position in sequence $i$.
The effect is that the surrogate's clipping operates on the overall sequence-level policy shift rather than individual token deviations.

## Policy Surrogates

`--policy-surrogate` selects the function $f(r_{i,t}, A_{i,t})$ from the overall objective.

### `ppo_clip` (default)

$$f(r_{i,t}, A_{i,t}) = \max\big(-r_{i,t} \cdot A_{i,t},\ -\text{clip}(r_{i,t},\ 1-\varepsilon,\ 1+\varepsilon_\text{high}) \cdot A_{i,t}\big)$$

Set $\varepsilon$ with `--eps-clip` (default 0.2) and $\varepsilon_\text{high}$ with `--eps-clip-high` (defaults to `--eps-clip`).
Optional dual-clip via `--eps-clip-c` (must be > 1.0): when $A_{i,t} < 0$, an additional lower bound $-\varepsilon_c \cdot A_{i,t}$ is applied.

### `is`

Plain importance sampling:

$$f(r_{i,t}, A_{i,t}) = -r_{i,t} \cdot A_{i,t}$$

### `tis`

Truncated importance sampling: the ratio is clamped before multiplication:

$$f(r_{i,t}, A_{i,t}) = -\text{clip}(r_{i,t},\ 1-\varepsilon,\ 1+\varepsilon_\text{high}) \cdot A_{i,t}$$

### `cis`

Clipped importance sampling.
When $r_{i,t}$ is in the clipped region of `ppo_clip` and `tis`, the gradient is exactly zero, silently drop the tokens.
`cis` avoids this by detaching the clamped ratio so gradient flows through $\log\pi_\theta$ rather than through $r$:

$$f(r_{i,t}, A_{i,t}) = -\text{sg}\big[\text{clip}(r_{i,t},\ 1-\varepsilon,\ 1+\varepsilon_\text{high})\big] \cdot A_{i,t} \cdot \log\pi_\theta(x_{i,t})$$

## Mismatch Correction

$\pi_\text{rollout} \ne \pi_\theta$ due to rollout-trainer policy numerical mismatch, the importance ratio $r_{i,t}$ may be computed against the wrong baseline.
`--old-logprob-source` controls which $\pi_\text{old}$ to use:

| Value | Behavior |
|-------|----------|
| `actor` (default) | Recompute $\log\pi_\text{old}$ with the actor at training start.|
| `rollout` | Use $\log\pi_\text{rollout}$ captured during generation (the ByPass mode). |

`--mismatch-correction custom` loads a user-defined correction function (e.g. Reject Sampling) via `--custom-mismatch-correction-function-path` (see [Customization](customization.md)).

## KL Penalty

KL penalty measures the divergence between $\pi_\theta$ and the frozen reference model $\pi_\text{ref}$ (load via `--ref-load`).
It constrains the trained policy from drifting too far from the reference.

KL is injected as a loss term (`--kl-loss-coef`), added directly to the training loss:
$$L_\text{total} = L_\text{pg} + \alpha_\text{kl} \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})$$

KL is estimated per-token from $\rho_{i,t} = \log\pi_\theta(x_{i,t}) - \log\pi_\text{ref}(x_{i,t})$.
The estimator type (`--kl-loss-type`) selects among `k1` ($\rho$), `k2` ($\rho^2/2$), or `k3`/`low_var_kl` ($e^{-\rho} - 1 + -\rho$).

## Entropy Bonus

`--entropy-coef` (default 0.0) adds an entropy bonus to encourage exploration:

$$L_\text{total} = L_\text{pg} - \alpha_\text{ent} \cdot H(\pi_\theta)$$

Entropy calculation requires realizing full `[seq_len, vocab]` which consumes significant amount of GRAM. Only computed when $\alpha_\text{ent} \ne 0$.

## Critic Value Loss

When using `ppo_gae`, the critic is trained alongside the actor.
Each training step, the critic runs a forward pass to produce per-token value predictions $V_\theta(x_{i,\le t})$, then updates its weights to minimize:
$$L_\text{value} = \frac{1}{2}\max\left((V_\theta - G)^2,\ \left(\text{clip}(V_\theta, V_\text{old} \pm \varepsilon_v) - G\right)^2\right)$$
where $G_{i,t} = A_{i,t} + V_\text{old}(x_{i,\le t})$ are the target returns and $\varepsilon_v$ = `--value-clip` (default 0.2).

The critic has its own optimizer and learning rate (`--critic-lr`).
`--num-critic-only-steps` (default 0) runs N critic-only training steps at the start before the actor begins updating.
This warms up the value baseline so early advantage estimates are more stable.


## Loss Reduction

By default, per-token losses within each sequence are averaged, then summed across the batch.
When `--calculate-per-token-loss` is set, per-token losses are summed (not averaged) within each sequence, weighting longer responses more.

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--advantage-estimator` | `grpo` | `grpo`, `gspo`, `ppo_gae` |
| `--policy-surrogate` | `ppo_clip` | `ppo_clip`, `is`, `tis`, `cis` |
| `--eps-clip` | `0.2` | PPO clip lower bound |
| `--eps-clip-high` | `None` | PPO clip upper bound (defaults to `--eps-clip`) |
| `--eps-clip-c` | `None` | Dual-clip lower bound (> 1.0) |
| `--kl-loss-coef` | `0.0` | KL loss coefficient (non-zero enables the KL loss term) |
| `--kl-loss-type` | `low_var_kl` | KL estimator: `k1`, `k2`, `k3`, `low_var_kl` |
| `--use-unbiased-kl` | `False` | Multiply KL by importance ratio |
| `--entropy-coef` | `0.0` | Entropy bonus coefficient |
| `--gamma` | `1.0` | GAE discount factor |
| `--lambd` | `1.0` | GAE lambda |
| `--normalize-advantages` | `False` | Globally normalize advantages across DP ranks |
| `--value-clip` | `0.2` | Critic value loss clip range |
| `--old-logprob-source` | `actor` | Source of old log-probs: `actor`, `rollout` |
| `--mismatch-correction` | `none` | `none`, `custom` |
| `--calculate-per-token-loss` | `False` | Sum (not average) token losses within each sequence |
| `--loss-type` | `policy_loss` | `policy_loss`, `custom_loss` |
| `--ref-load` | `None` | Reference model checkpoint |

---

**See also:** [Data Layout](data-layout.md) | [Placement & Weight Update](placement.md) | [Customization](customization.md)
