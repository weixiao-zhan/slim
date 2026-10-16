---
name: add-eval-dataset-config
description: Guide for adding and validating evaluation dataset configuration in slim. Use when user wants to configure eval datasets via --eval-config or --eval-prompt-data, add per-dataset overrides, or customize evaluation rollout behavior.
---

# Add Eval Dataset Config

Configure evaluation datasets in slim with explicit dataset-level overrides and predictable runtime behavior.

## When to Use

Use this skill when:

- User asks to add evaluation datasets for periodic eval
- User asks to migrate from `--eval-prompt-data` to structured `--eval-config`
- User asks for per-dataset eval overrides (sample count, context length, rm_type, metadata)

## Step-by-Step Guide

### Step 1: Choose Config Entry Method

Supported inputs:

- Structured config file: `--eval-config <yaml>`
- Legacy CLI pairs: `--eval-prompt-data <name1> <path1> <name2> <path2> ...`

If `--eval-interval` is set, eval datasets must be configured.

### Step 2: Build YAML with Required Fields

Each dataset needs at least:

- `name`
- `path`

Example:

```yaml
eval:
  defaults:
    eval_n_samples_per_prompt: 1
  datasets:
    - name: aime
      path: /path/to/aime.jsonl
      rm_type: math
      max_context_len: 8192
      metadata_overrides:
        split: test
```

### Step 3: Understand Override Priority

`slim/utils/eval_config.py` resolves fields in this order:

1. Dataset-level values in `eval.datasets[*]`
2. `eval.defaults`
3. CLI args fallback (for example eval_* or rollout_* fields)

Overridable fields are the `EvalDatasetConfig` fields:

- Runtime: `eval_n_samples_per_prompt`, `max_context_len`
- Extra: `rm_type`, `custom_generate_function_path`, `app_service`, `metadata_overrides`

Sampling is not per-dataset: eval samples at `--rollout-temperature`, and a dataset needing
anything else points `custom_generate_function_path` at a generate function that sets it.

### Step 4: Wire Eval Function if Needed

By default, eval uses `--eval-function-path` (defaults to rollout function path).
Use a separate eval function when inference/eval behavior must differ from training rollout.

### Step 5: Validate Parsing and Runtime

- Start with config parsing sanity by running a short launch command.
- Confirm dataset entries are loaded into `args.eval_datasets`.
- Verify output keys match eval logging/metrics expectations.

## Common Mistakes

- Missing `name` in dataset entries
- Odd-length `--eval-prompt-data` pairs
- Setting `--eval-interval` without any eval dataset
- Mixing reward dict outputs without `eval_reward_key` configuration

## Reference Locations

- Eval config model: `slim/utils/eval_config.py`
- Eval config resolution: `slim/utils/arguments.py`
- Eval rollout path: `slim/rollout/sglang_rollout.py`
- Customization docs: `docs/en/get_started/customization.md`
