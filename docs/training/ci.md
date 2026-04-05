# CI (Continuous Integration)

slim uses GitHub Actions for CI. Tests are triggered by **PR labels** — adding a specific label to a PR will run the corresponding test suite.

## How It Works

Note: The workflow files (`.github/workflows/pr-test.yml`, `pr-test.yml.j2`, `generate_github_workflows.py`) and `tests/ci/gpu_lock_exec.py` are not yet present in this repository. The description below documents the intended CI design.

The workflow is defined in `.github/workflows/pr-test.yml` (auto-generated from `pr-test.yml.j2`). Each CI job:

1. Runs on a self-hosted GPU runner inside a Docker container (`slimrl/slim:latest`).
2. Installs slim with `pip install -e . --no-deps`.
3. Acquires the required GPUs via `tests/ci/gpu_lock_exec.py --count <num_gpus>`.
4. Executes the test script: `bash tests/test_<name>.sh`.

Each test is a self-contained shell script that sources `tests/common.sh` for shared helpers (ray start/stop, cleanup, wandb args), then calls `run_train` with the training arguments.

## CI Labels

Add a label to your PR to trigger the corresponding test suite:

| Label | Job | Description |
|---|---|---|
| `run-ci-short` | `e2e-test-short` | Lightweight smoke tests with Qwen2.5-0.5B (4 GPUs). Fast feedback loop. |
| `run-ci-fsdp` | `e2e-test-fsdp` | FSDP backend tests (true on-policy, VL, etc.). |
| `run-ci-precision` | `e2e-test-precision` | Numerical precision validation (parallel check, etc.). |
| `run-ci-ckpt` | `e2e-test-ckpt` | Checkpoint save/load correctness (sync and async-save). |
| `run-ci-image` | `e2e-test-image` | Full test suite run on `slimrl/slim-test:latest` image (for image validation). |
| `run-ci-changed` | `e2e-test-changed` | **Dynamically** detects new/modified test files in the PR and runs only those. |

All labels also run when triggered via `workflow_dispatch` (manual run from the Actions tab).

## Writing a New Test

1. Create `tests/test_<your_test_name>.sh` following the standard pattern:

```bash
#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

MODEL_DIR="${MODEL_DIR:-/home/ubuntu/models/YourModel}"
DATASET_DIR="${DATASET_DIR:-/home/ubuntu/datasets/your_data}"

start_ray
run_train "
    --num-rollout 200
    --rollout-batch-size 32
    ...
    --hf-checkpoint $MODEL_DIR
    $(wandb_args your-test-name)
"
```

2. Make it executable: `chmod +x tests/test_<your_test_name>.sh`

3. **For quick validation**: Just push your test file and add `run-ci-changed` to the PR.

4. **To register in a permanent label group**: Edit `.github/workflows/pr-test.yml.j2`, add an entry to the desired job's `tests` list, then regenerate:

```bash
cd .github/workflows && python generate_github_workflows.py
```

Remember to commit both the `.j2` and the generated `.yml` file.

## Workflow Generation

The workflow file `pr-test.yml` is auto-generated from the Jinja2 template `pr-test.yml.j2`. **Do not edit `pr-test.yml` directly.** To make changes:

1. Edit `.github/workflows/pr-test.yml.j2`.
2. Run `python .github/workflows/generate_github_workflows.py`.
3. Commit both files.

## Customization Contract Tests

For CPU-only contract tests that validate hooks loaded from function paths, run:

```bash
python -m pytest \
  tests/plugin_contracts/test_plugin_rollout_contracts.py \
  tests/plugin_contracts/test_plugin_generate_contracts.py \
  tests/plugin_contracts/test_plugin_path_loading_contracts.py \
  tests/plugin_contracts/test_plugin_runtime_hook_contracts.py
```
