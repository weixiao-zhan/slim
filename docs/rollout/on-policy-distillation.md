# On-Policy Distillation

On-policy distillation (OPD) enables a student model to learn from a larger teacher model by training on its own rollouts while matching the teacher's token-level log-probabilities. OPD is orthogonal to advantage estimators — it works as an additive KL penalty on top of any estimator (GRPO, PPO, REINFORCE++, etc.).

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `sglang`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). Controls the weight of the distillation signal relative to the RL advantage. |

## How It Works

OPD modifies the advantage computation by subtracting a KL penalty term that encourages the student to match the teacher's output distribution:

$$
\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t
$$

Where $A_t$ is the original advantage from the base estimator (e.g., GRPO), $\lambda_{\text{opd}}$ is `--opd-kl-coef`, and $D_{\text{KL}}$ is the token-level reverse KL divergence.

This means OPD can be combined with any advantage estimator, including GRPO, PPO, REINFORCE++, and GSPO.

## Teacher Mode

### SGLang Mode (`--opd-type sglang`)

The teacher runs on an external SGLang server. Teacher log-probs are obtained during the rollout phase.

**When to use**: The teacher has a different architecture from the student, or the teacher is too large to load alongside the training model.

**How it works**:
1. An external SGLang server runs the teacher model.
2. During rollout, the custom reward function (`slim.rollout.on_policy_distillation.reward_func`) sends each sample to the teacher server to obtain token-level log-probs.
3. The custom post-processing function (`slim.rollout.on_policy_distillation.post_process_rewards`) trims the teacher log-probs to the response span and stores them in `sample.teacher_log_probs`.
4. During training, the KL penalty is computed from the stored teacher log-probs and applied to advantages.

**Configuration**:
```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--custom-rm-path slim.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slim.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

## Running the Examples

```bash
# 1. Download models and data
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2. Launch SGLang teacher server, then run train.py with --use-opd --opd-type sglang
```

## Preliminary Results

Using Qwen3-8B-Base model SFT-ed on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, on-policy distillation with a Qwen3-32B teacher on the remaining data yields:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |
