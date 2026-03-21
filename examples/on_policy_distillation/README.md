# On-Policy Distillation Example

This example shows how to run **on-policy distillation (OPD)** using slime. A small student (Qwen3-8B) is aligned to imitate a larger teacher (Qwen3-32B) by training only on the student's own rollouts and matching the teacher's token-level log-probabilities.

## Key Features

- **OPD is orthogonal to advantage estimators**: OPD works as an additive KL penalty on top of any advantage estimator (GRPO, PPO, REINFORCE++, etc.), not as a separate estimator.
- **Teacher mode**: Teacher runs on an external SGLang server (`--opd-type sglang`), teacher log-probs are obtained during rollout.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `sglang`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). |

## Components

- `slime/rollout/on_policy_distillation.py` implements (for SGLang mode):
  - `reward_func` calls the teacher server (via `args.rm_url`) with every sample to obtain token-level logprobs.
  - `post_process_rewards` trims the teacher logprobs to the generated response span and writes the tensors back to each `Sample` to compute advantages.
- `run-qwen3-8B-opd.sh` launches an SGLang teacher server, then submits a Ray job that runs `train.py`.

## Running the example

### Using SGLang Teacher (External Server)

1. Download or prepare the required checkpoints and data.
```bash
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

2. Run on-policy distillation:
```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

# Preliminary Results
Using Qwen3-8B-Base model sfted on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, we performed on-policy distillation with a Qwen3-32B teacher on the remaining data. Evaluation on Math500 shows:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |





# FAQ
1. **What happens if I set wrong arguments?**
   The system will raise clear errors:
   - `--use-opd` without `--opd-type`: Error asking you to specify type
   - `--opd-type sglang` with `--opd-teacher-load`: Error indicating conflict


# References
1. https://thinkingmachines.ai/blog/on-policy-distillation/
2. https://arxiv.org/abs/2306.13649
3. https://arxiv.org/abs/2306.08543