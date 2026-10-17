# Training Loss

slim trains a policy $\pi_\theta$ using on-policy RL.
Each rollout produces a batch of sequences; for each sequence $i$, the rollout policy $\pi_\text{rollout}$ generates tokens $x_{i,t}$ and records per-token log-probabilities.
A reward function assigns a scalar reward $R_i$ to the sequence.
The training loop then computes advantages $A_{i,t}$ and updates $\pi_\theta$ via a policy gradient objective.

Denote $\pi(x_{i,t})$: shorthand for $\pi(x_{i,t} | x_{i,<t})$;
$\pi_\theta$: the training policy; 
$\pi_\text{old}$: the actor policy used to compute the importance-ratio baseline;
$\pi_\text{rollout}$: the policy used for generation running on SGLang; 
$\pi_\text{ref}$: frozen reference.

The RL policy gradient objective writes:

$$\nabla_\theta L_{\text{pg}} = -\mathbb{E}_{x_i \sim \pi_\text{old}} \left[ \sum_{t=1}^{T} \hat{A}_{i,t} \cdot w(r_{i,t}, \hat{A}_{i,t}) \cdot \nabla_\theta \log\pi_\theta(x_{i,t}) \right]$$

where $r_{i,t} = \pi_\theta(x_{i,t}) / \pi_\text{old}(x_{i,t})$ is the importance ratio.

Only positions with a nonzero `loss_mask` enter the loss, so the actor projects only those through the LM head: the logits are `[loss tokens, vocab]`, and the recomputed $\log\pi_\text{old}$ and $\log\pi_\text{ref}$ are zero elsewhere.

The two design axes are:
1. **Advantage estimator** (`--advantage-estimator`): how $\hat{A}_{i,t}$ is computed from rewards.
2. **Policy surrogate** (`--policy-surrogate`): the surrogate $J(r_{i,t}, \hat{A}_{i,t})$ that combines the ratio and advantage (clipping, truncation, etc.), and its weight $w(r_{i,t}, \hat{A}_{i,t})$.

## Advantage Estimators

`--advantage-estimator` controls how per-token advantages $A_{i,t}$ are derived from rewards.
The driver invokes `AdvantageEstimator.compute_training_targets()` after rollout. It regroups the flattened trajectories by `episode_index` so baselines see an attempt as a unit, then writes targets through `Trajectory.set_train_targets()`.

Reward lives on the trajectory. The reward model can score each `Trajectory.reward`, or score `Episode.reward`, which is short-hand for broadcasting to every trajectory.

### `ppo_gae`

The full form.
Requires a critic model (set via `--critic-num-gpus`, `--critic-load`) that produces per-token value predictions $V(x_{i,\le t})$.
GAE computes advantages with discount $\gamma$ (`--gamma`, default 1.0) and lambda $\lambda$ (`--lambd`, default 1.0):

$$\delta_{i,t} = R_{i,t} + \gamma V_{i,t+1} - V_{i,t}$$
$$A_{i,t} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{i,t+l}$$

The reward is placed at the last response token of each trajectory; all others have $R_{i,t} = 0$.
Masked prompt, observation, and terminal source positions do not advance the GAE recurrence and receive zero advantage.

The recurrence runs per trajectory, which is contiguous, so the recurrence within one is sound; it does not cross trajectory boundaries, since bootstrapping $V$ from one into another would require knowing which follows which. Each computes its own return from the reward it received.

### `grpo` (default)

Group-Relative Policy Optimization simplifies PPO-GAE by eliminating the critic entirely.
Instead of learning a value baseline, it uses the other attempts on the same prompt as the baseline.
The raw rewards remain unchanged. Group statistics are computed over **episode-level scalars**, scattered by `group_index`.

For group $g$ with episodes $e \in g$, let $R_e$ be the mean of episode $e$'s trajectory rewards. Then for each trajectory $\tau$ of each episode in the group:

$$A_{\tau} = \frac{R_\tau - \text{mean}_{e \in g}(R_e)}{\text{std}_{e \in g}(R_e)}$$

The baseline and scale come from episode-level statistics, so an episode that made 20 generation calls does not dominate the group mean of one that made a single call.
The numerator is the trajectory's own reward, so within-episode differentiation survives when rewards are genuinely per-trajectory.
For single-trajectory episodes this reduces to the per-sequence form.

`--disable-group-advantage-std-normalization` keeps the mean-centering step but omits division by the group standard deviation.
`--disable-group-advantage-normalization` skips both operations and uses $A_{\tau} = R_\tau$, providing REINFORCE-style reward direction without changing the stored raw reward.


### `gspo`

Uses the same advantage as `grpo`, but redefines the importance ratio fed to the policy surrogate.
Instead of the per-token ratio $r_{i,t} = \pi_\theta(x_{i,t}) / \pi_\text{old}(x_{i,t})$, GSPO uses a sequence-level ratio:

$$r_i = \exp\left( \frac{1}{|T_i|} \sum_t (\log \pi_\theta(x_{i,t}) - \log \pi_\text{old}(x_{i,t})) \cdot \mathbf{1}[\text{loss\_mask}_{i,t}] \right)$$

This single scalar is expanded to every token position in sequence $i$.
The effect is that the surrogate's clipping operates on the overall sequence-level policy shift rather than individual token deviations.

## Policy Surrogates

`--policy-surrogate` selects the surrogate $J(r, A)$.
The naive loss $-J(r, A)$ sends the gradient through $r$.
At the prefix $x_{i,<t}$, write $p = \pi_\theta$, $x = x_{i,t}$ for the sampled token, and $s_x = \nabla_\theta \log p_x$ for its score.
The actor uses a detached loss with the same gradient, because $\nabla r = r s_x$:

$$L_t = -A_t \text{sg}[w_x] \log p_x$$

$$A w = r \frac{\partial J}{\partial r}$$

$w_x = w(r, A_t)$ is the detached weight, so the gradient is $-A_t w_x s_x$.
`pg_loss` logs this form, so its value is not $-J$.
A token in the clipped region of `ppo_clip` or `tis` has $\partial J / \partial r = 0$, so its weight is $0$.
$\text{clip}(r)$ is short for $\text{clip}(r, 1-\varepsilon, 1+\varepsilon_\text{high})$:

| Surrogate | $J(r, A)$ | $w(r, A)$ |
|---|---|---|
| `ppo_clip`, $A > 0$ | $\min(rA, \text{clip}(r) A)$ | $r \cdot \mathbf{1}[r \le 1+\varepsilon_\text{high}]$ |
| `ppo_clip`, $A < 0$ | $\min(rA, \text{clip}(r) A)$ | $r \cdot \mathbf{1}[r \ge 1-\varepsilon]$, and also $r \le \varepsilon_c$ under dual-clip |
| `is` | $rA$ | $r$ |
| `tis` | $\text{clip}(r) A$ | $r \cdot \mathbf{1}[1-\varepsilon \le r \le 1+\varepsilon_\text{high}]$ |
| `cis` | — | $\text{clip}(r)$ |

Under `gspo`, $r$ is the sequence-level ratio, and the sequence mean of $\log\pi_\theta$ replaces $\log p_x$.

### `ppo_clip` (default)

Set $\varepsilon$ with `--eps-clip` (default 0.2) and $\varepsilon_\text{high}$ with `--eps-clip-high` (defaults to `--eps-clip`).
Optional dual-clip via `--eps-clip-c` (must be > 1.0): when $A < 0$, $J$ becomes $\max(J, \varepsilon_c A)$.

### `is`

Plain importance sampling.

### `tis`

Truncated importance sampling: the ratio is clamped before multiplication.

### `cis`

Clipped importance sampling.
When $r$ is in the clipped region of `ppo_clip` and `tis`, the weight is exactly zero, so these surrogates silently drop the token.
`cis` clamps the weight itself, so the token keeps a gradient through $\log\pi_\theta$ with a bounded weight.
It has no standard $J$, and $w$ defines it.

`--eps-clip 0 --eps-clip-high 0` pins the weight to $1$, which is plain REINFORCE, $-A_t \log p_x$, with no token counted as clipped.

## Score Centering

`--use-score-centering K` enables score centering ([Marek & Ryabinin, arXiv 2609.20807](https://arxiv.org/abs/2609.20807)) with the sampler's top-$K$.
Write $q = \pi_\text{rollout}$ at the same prefix, and $s_v = \nabla_\theta \log p_v$ for each token $v$ in the vocabulary.
The expected per-token update splits into a drift term and a signal term, with $\bar s = \sum_v q_v s_v$:

$$\mathbb{E}_q[A s_x] = \mathbb{E}_q[A] \bar s + \mathrm{Cov}_q(A, s_x)$$

On-policy $\bar s = 0$.
Under rollout-trainer mismatch $\bar s \ne 0$, and the drift term distills $p$ toward $q$ whatever the rewards are.
Group centering does not remove it, because the advantage mean at one prefix is not zero.
Score centering subtracts the sampler-expected weighted score.
It adds one term to the [detached loss](#policy-surrogates), so $w_x s_x$ becomes $w_x s_x - \mathbb{E}_{\hat q}[w s]$, and $A_t$ is a fixed value:

$$L_t = -A_t \text{sg}[w_x] \log p_x + A_t \sum_v \text{sg}[\hat q_v w_v] \log p_v$$

$w_v = w(p_v / q_v, A_t)$ is the surrogate weight, and it must be relative to the sampler.
The rollout records the sampler's top-$K$ tokens $H$, and the tail takes the trainer's shape, $\hat q_v = \rho p_v$ for $v \notin H$:

$$\rho = \frac{1 - \sum_H q_v}{1 - \sum_H p_v}$$

A tail token has the ratio $1/\rho$, so its weight is the constant $w(1/\rho, A_t)$.
Because $\sum_v p_v s_v = 0$, the tail folds into the top-$K$, with $\alpha = \rho w(1/\rho, A_t)$:

$$L_t = \underbrace{-A_t \text{sg}[w_x] \log p_x}_{\text{surrogate}} + A_t \sum_{v \in H} \text{sg}[q_v w_v - \alpha p_v] \log p_v$$

| Setting | $w(r)$ | In the paper | Use |
|---|---|---|---|
| `cis --eps-clip 0 --eps-clip-high 0` | $1$ | SC | Score centering alone, for a mismatch that does not grow within a rollout, such as a quantized sampler. |
| e.g. `cis --eps-clip 1 --eps-clip-high 1` | $\min(r, 2)$ | TIS + SC | Score centering with truncated IS, for staleness from several optimizer steps per rollout. |
| e.g. `tis --eps-clip 0.5 --eps-clip-high 4` | $r \cdot \mathbf{1}[0.5 \le r \le 5]$ | MIS + SC | Score centering with masked IS, for staleness or weight noise. |
| `ppo_clip` | $r$ where the token is not clipped for the sign of $A_t$, else $0$ | Not tested | |

`is` has $w(r) = r$, which makes the correction exactly zero, so `--use-score-centering` rejects it.
`gspo` replaces the per-token ratio with one per sequence, so it has no weight per top-$K$ token.
`--use-score-centering` also requires `--old-logprob-source rollout`, so that the ratio is $p / q$.

The actor reads $\log p$ at the label and at the top-$K$ ids in one fused pass over the logits, without a `[seq_len, vocab]` log-softmax.
The top-$K$ travels as `rollout_topk_log_probs` and `rollout_topk_token_ids`, each `[T, K]` (see [Data Layout](data-layout.md)), so the cost is $8K$ bytes per token; $K = 32$ is 256 bytes.
SGLang reports $q$ after temperature scaling and before top-$p$ and top-$k$ truncation, so keep top-$p = 1$ and top-$k = -1$ for $q$ to be the sampling distribution.

`train/actor/score_centering/topk_mass` logs $\sum_H q_v$, the share of sampler mass that the top-$K$ covers, and `train/actor/score_centering/rho` logs $\rho$.
The paper reproduces its setup with `--loss-normalization-unit token` and `--disable-group-advantage-std-normalization`.

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

Entropy calculation realizes fp32 copies of the `[loss tokens, vocab]` logits, which consumes significant amount of GRAM. Only computed when $\alpha_\text{ent} \ne 0$.

## Critic Value Loss

When using `ppo_gae`, the critic is trained alongside the actor.
The critic first produces the old per-token predictions consumed by `AdvantageEstimator`. The estimator writes actor `advantages`, critic `values`, and critic `value_targets` to each trajectory. The actor and critic then train from separate physical packs.

The critic updates its weights to minimize:
$$L_\text{value} = \frac{1}{2}\max\left((V_\theta - G)^2,\ \left(\text{clip}(V_\theta, V_\text{old} \pm \varepsilon_v) - G\right)^2\right)$$
where $G_{i,t} = A_{i,t} + V_\text{old}(x_{i,\le t})$ is stored as `value_targets`, and $\varepsilon_v$ = `--value-clip` (default 0.2).

The actor, critic backbone, and critic value head use `--lr-actor`, `--lr-critic`, and `--lr-critic-value-head`.
Their optimizer updates begin at `--lr-actor-start-step`, `--lr-critic-start-step`, and `--lr-critic-value-head-start-step`.
Before its start step, a component has zero learning rate and does not accumulate optimizer state.


## Loss Reduction

`--loss-normalization-unit` selects the unit the loss denominator counts.

| Value | Weighting | Meaning |
|---|---|---|
| `episode` (default) | $w_d = 1 / k_{e(d)}$ | Each attempt weighted equally. Pairs with the episode-level GRPO baseline. |
| `trajectory` | $w_d = 1$ | Each generation call weighted equally. |
| `token` | — | Token-sum denominator, no per-document weight. |

Both sequence-level units are one formula over the per-document weight $w_d$, where $d$ ranges over packed documents (trajectories) and $m_{d,t}$ is the loss mask:

$$L = \frac{1}{\sum_d w_d} \sum_{d} w_d \cdot \frac{\sum_t v_{d,t} m_{d,t}}{\sum_t m_{d,t}}$$

Under `episode`, $\sum_d w_d$ counts attempts and the loss is a mean over episodes of the mean over each episode's trajectories. Under `trajectory` it is the document mean. The two coincide for single-trajectory episodes.

`episode` is the default because group centering zeroes the mean advantage per *episode*, so the denominator must use the same unit. Otherwise an episode with $k$ trajectories carries $k$ times the gradient weight of a single-trajectory episode and the advantage mean over the gradient is not zero. It is also the setting under which a broadcast episode reward stays credit-neutral: reward $R$ spread over $k$ trajectories each weighing $1/k$ gives the attempt total credit $R$.

Choosing `trajectory` weights a 50-call attempt 50 times as heavily as a single-call one, which is deliberate only when the generation call is genuinely the unit of interest. Under `token`, per-token losses are summed rather than averaged within each document, weighting longer responses more.

`Trajectory.loss_weight` carries $w_d$: the flattener writes $1/k_e$ under `episode`, $1.0$ under `trajectory`, and $0.0$ for padding trajectories. The loss applies $w_d$ and divides by $\sum_d w_d$ with no mode switch, and `count_global_denominators` reduces that one sum over the DP group.

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--advantage-estimator` | `grpo` | `grpo`, `gspo`, `ppo_gae` |
| `--policy-surrogate` | `ppo_clip` | `ppo_clip`, `is`, `tis`, `cis` |
| `--use-score-centering` | `None` | Top-$K$ size; enables score centering |
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
| `--disable-group-advantage-normalization` | `False` | Use raw rewards directly as GRPO or GSPO advantages |
| `--disable-group-advantage-std-normalization` | `False` | Disable group standard-deviation scaling for GRPO or GSPO advantages |
| `--value-clip` | `0.2` | Critic value loss clip range |
| `--old-logprob-source` | `actor` | Source of old log-probs: `actor`, `rollout` |
| `--mismatch-correction` | `none` | `none`, `custom` |
| `--loss-normalization-unit` | `episode` | Loss denominator unit: `episode`, `trajectory`, `token` |
| `--loss-type` | `policy_loss` | `policy_loss`, `custom_loss` |
| `--ref-load` | `None` | Reference model checkpoint |

---

**See also:** [Data Layout](data-layout.md) | [Placement & Weight Update](placement.md) | [Customization](customization.md)
