# Usage Guide


## Introduction to slime Parameters

When using slime, parameters are primarily passed for the following purposes:

1.  To allocate a portion of the GPUs in the cluster for training and another portion for inference.
2.  To configure the FSDP training backend.
3.  To load SGLang for the inference portion.
4.  To configure the hyperparameters required for RL training.

Following this order, we need to configure these parameters:

### Cluster Resource Allocation

There are four main parameters for cluster resource allocation:

  - `--actor-num-nodes`: The number of nodes required for RL actor training.
  - `--actor-num-gpus-per-node`: The number of GPUs per node for RL actor training.
  - `--rollout-num-gpus`: The total number of GPUs required for rollout (inference).
  - `--rollout-num-gpus-per-engine`: The number of GPUs per inference engine. This parameter is similar to SGLang's `tp_size`. When performing multi-node serving, this value should be the total number of GPUs. For example, if serving one model with 2 nodes and 16 GPUs, this value should be 16.
    The reason for not using a parameter like `--sglang-tp-size` is that we might consider supporting SGLang's `dp_size` parameter in the future, which means an engine could contain multiple SGLang servers (currently, only `--sglang-dp-size` under the `--sglang-enable-dp-attention` condition is supported).

With the default configuration, we use these parameters to allocate `actor_num_nodes * actor_num_gpus_per_node` GPUs for training and `rollout_num_gpus` GPUs for inference via Ray, thus achieving a separation of training and inference resources.

For co-located training and inference, you also need to configure:

  - `--colocate`: Enables co-located training and inference. When enabled, it ignores `--rollout-num-gpus` and makes the number of GPUs for training and inference equal.

Additionally, slime supports Prefill and Decode disaggregation (PD Disaggregation). You can set the number of servers used for Prefill by setting the `--prefill-num-servers` argument.

### Training Backend

slime uses PyTorch FSDP2 as its training backend. FSDP loads HuggingFace format weights directly via `AutoModelForCausalLM.from_pretrained()`, so no checkpoint conversion is needed.

#### Model Configuration

FSDP reads model configuration automatically from the HuggingFace checkpoint. No manual model parameter configuration is required.

Note:

  - slime uses data packing (also known as varlen or thd) for training.

#### Performance Configuration

FSDP uses pure data parallelism (1D device mesh). The following parameters control training performance:

  - `--gradient-checkpointing`: Enable gradient checkpointing to reduce GPU memory usage at the cost of additional computation.
  - `--fsdp-cpu-offload`: Offload FSDP parameters to CPU to save GPU memory.
  - `--use-dynamic-batch-size`: Enable dynamic batching for efficient sample packing.
  - `--max-tokens-per-gpu`: Maximum number of tokens processed per GPU when dynamic batching is enabled.

#### Loading Checkpoints

When using slime, there are three parameters for loading and saving checkpoints:

  - `--ref-load`: The HuggingFace checkpoint for the reference model.
  - `--load`: The checkpoint for the actor. If `--load` is not set, or if the specified directory does not exist or does not contain a valid checkpoint, the actor will be initialized from the `--ref-load` checkpoint.
  - `--save`: The path where the actor's checkpoints are saved.

To convert FSDP checkpoints back to HuggingFace format, use `tools/convert_fsdp_to_hf.py`.

### Loading SGLang

Loading SGLang is very simple. You only need:

  - `--hf-checkpoint`: The Hugging Face checkpoint used to initialize SGLang.

Note:

  - Before the first training step, slime will synchronize the parameters from the training backend to SGLang. Therefore, the `--hf-checkpoint` does not need to contain the latest training parameters, and you do not need to change the HF checkpoint when resuming training.
  - By default, SGLang reads the maximum context length from the `config.json` in the Hugging Face checkpoint. You can use the `--sglang-context-length` parameter to override this value to support longer inference.
  - During co-located training and inference, the training backend and SGLang will offload sequentially, but they still need to leave some memory for each other. You need to adjust SGLang's total VRAM usage by reducing `--sglang-mem-fraction-static`.
  - slime supports passing through sgl-router parameters by adding a `router` prefix to the original parameter name. For example, sgl-router's `--balance-abs-threshold` parameter should be set as `--router-balance-abs-threshold`. Since sgl-router uses cache-aware routing by default, it may cause uneven request distribution. You can set `--router-balance-abs-threshold 0` to force balanced distribution, but this may affect prefix cache hit rate in multi-turn conversation scenarios.

For details on some of SGLang's customizations and the principles behind how slime incorporates SGLang, please see the "How to Use SGLang" section.

### Data Format

Currently, slime only supports loading files in `.jsonl` format, where each line of the file is a JSON object. An example of a single data entry (expanded) is as follows:

```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```

This corresponds to the following configuration:

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

Please note that the `step_loss_mask` (default=1) here is for SFT phase. If it is set to 0, the turn will not contibute to the final loss; if it is set to 1, slime will use the normal `loss_mask`.
Additionally, we provide a `metadata_key`, which defaults to `"metadata"`. When read, slime will load the metadata from the data, which can be helpful for custom data generation or creating custom reward models.

### Hyperparameters for RL Training

- `--advantage-estimator`: Specifies the RL algorithm for the training process. Currently supported algorithms include:
    - `grpo` ([https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300))
    - `gspo` ([https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071))
    - `reinforce_plus_plus` and `reinforce_plus_plus_baseline` ([https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262))
    - `ppo` ([https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347))

  Note: On-policy distillation (OPD) is now orthogonal to the advantage estimator. Use `--use-opd` and `--opd-kl-coef` to enable OPD on top of any estimator.
- `--calculate-per-token-loss`: By default, slime calculates loss on a per-sample basis, i.e., `mean(sum(sample_i) / len(sample_i))`. Enable this flag to calculate loss on a per-token basis, i.e., `sum(sum(sample_i)) / sum(len(sample_i))`.
- `--use-tis`: Enable this setting to use TIS (Truncated Importance Sampling) (https://fengyao.notion.site/off-policy-rl).

#### GRPO Algorithm

GRPO (Group Relative Policy Optimization) is an RL algorithm proposed in DeepSeek-Math. Its core idea is to compute advantage through intra-group relative comparisons, eliminating the need for a separate critic model.

To use GRPO, set:

```bash
--advantage-estimator grpo
```

Key features of GRPO:

- **No Critic Model Required**: GRPO samples multiple responses for the same prompt and estimates advantage by computing relative rewards within the group, avoiding the overhead of training and maintaining a critic model.
- **Resource Efficient**: Since no critic model is needed, GPU resources can be fully utilized for actor training and inference.
- **Simple to Use**: Easy configuration - just set `--advantage-estimator grpo`.

Related parameters:

- `--n-samples-per-prompt`: Number of responses sampled per prompt for intra-group comparison.
- `--normalize-advantages`: Whether to normalize advantages.
- `--eps-clip`: PPO-style clip range.

#### PPO Algorithm

PPO (Proximal Policy Optimization) is a classic RL algorithm that uses a critic model to estimate the value function for computing advantages.

To use PPO, set:

```bash
--advantage-estimator ppo
```

**Note: In PPO, the Critic and Actor request GPUs in parallel**, which should be considered when allocating resources. Specifically:

- The critic model occupies a separate set of GPUs, independent from the actor's GPU resources.
- You can configure critic resources using `--critic-num-nodes` and `--critic-num-gpus-per-node`.
- If critic resource parameters are not configured, the same resource configuration as the actor will be used by default.

Cluster resource allocation example:

```bash
# Actor uses 1 node, 4 GPUs
--actor-num-nodes 1
--actor-num-gpus-per-node 4

# Critic uses 1 node, 4 GPUs (parallel to Actor)
--critic-num-nodes 1
--critic-num-gpus-per-node 4

# Rollout uses 8 GPUs
--rollout-num-gpus 8
```

With the above configuration, a total of `4 (actor) + 4 (critic) + 8 (rollout) = 16` GPUs are required.

PPO-related parameters:

- `--critic-load`: Checkpoint path for the critic model.
- `--critic-save`: Save path for the critic model.
- `--critic-lr`: Learning rate for the critic model.
- `--critic-lr-warmup-iters`: Number of warmup steps for the critic model.
- `--num-critic-only-steps`: Number of steps to train only the critic at the beginning of training.
- `--eps-clip`: PPO clip range.
- `--value-clip`: Clip range for value loss.
- `--kl-coef`: KL penalty coefficient for reward shaping.

## Custom Rollout Function

slime supports customizing data generation (rollout) to various degrees.

  - By default, it uses the `generate_rollout` function from [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py) for data generation. This file implements an asynchronous (asyncio) data generation flow based on SGLang and supports features like dynamic sampling and partial rollout.

  - You can completely replace the `generate_rollout` in sglang\_example.py by using the `--rollout-function-path` parameter. You just need to ensure that the function signature passed via `--rollout-function-path` is as follows:

    ```python
    def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
        """
        Args:
            args: the whole args
            rollout_id: int, the id of the rollout, used for deterministic data generation
            data_source: the data source to get and store samples
            evaluation: bool, whether the rollout is for evaluation or not
        
        Returns:
            RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
        """
            ...
            return output
    ```

    Where:

      - `args`: The complete arguments used for the slime run.

      - `rollout_id`: The ID of the current data generation round, used to ensure data order when resuming training.

      - `data_source`: A globally unique data source in slime, which can be used to get initial prompts, data IDs, and store partially generated samples for later use.

      - `evaluation`: A boolean indicating if the rollout is for evaluation. You can configure a separate evaluation function using `--eval-function-path`.

      - The returned `Sample` type is defined in [slime/utils/types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py). When implementing, you need to ensure the following fields are correctly set:

          - `tokens`: The tokens for the prompt + response.
          - `response_length`: The total length of the response. For multi-turn tasks, this is the length of the tokens remaining after the first-turn prompt.
          - `reward`: The reward for this data sample.
        - `status`: The status of this data sample (e.g., `Sample.Status.COMPLETED`, `Sample.Status.TRUNCATED`, `Sample.Status.ABORTED`, `Sample.Status.FAILED`).
          - `loss_mask` should be the same length as `response_length`, with `1` for tokens that should be included in the loss calculation and `0` for those that should be masked out.

  - In some cases, you may only need to replace the data generation logic. You can do this using `--custom-generate-function-path`. A simplified implementation of this function is as follows:

    ```python
    async def generate(args, sample: Sample, sampling_params) -> Sample:
        global TOKENIZER
        if TOKENIZER is None:
            TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # send request to router
        output = await post(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
            {
                "text": sample.prompt,
                "sampling_params": sampling_params,
            }
        )

        prompt_tokens_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
        response_token_ids = TOKENIZER(output["text"], add_special_tokens=False)["input_ids"]

        # set sample
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
        else:
            sample.status = Sample.Status.COMPLETED
        sample.response = output["text"]

        return sample
    ```

    For a more complete version, please refer to [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py).

  - Sometimes, you may also need to support a custom reward model. This can be configured by setting `--custom-rm-path`.

## How to Use SGLang

slime implements a server-based engine using SGLang via the `HttpServerEngineAdapter` as an intermediary.

### Parameter Configuration

slime incorporates almost all SGLang parameters by using SGLang's `ServerArgs.add_cli_args`. When setting an SGLang parameter, you need to add the `--sglang-` prefix. For example:

  - In co-located training and inference, you often need to limit `--mem-fraction-static`. This parameter should be changed to `--sglang-mem-fraction-static`.
  - During training, if you want SGLang to infer beyond the maximum context length specified in the Hugging Face checkpoint's `config.json`, you need to use `--context-length`, which becomes `--sglang-context-length` in slime.
  - For multi-node large EP inference, you might need `--ep-size`, `--enable-dp-attention`, `--dp-size`, `--moe-a2a-backend deepep`, etc. These can be passed as `--sglang-ep-size`, `--sglang-enable-dp-attention`, `--sglang-dp-size`, and `--sglang-moe-a2a-backend deepep` respectively.

Some parameters related to slime's resource scheduling are configured by slime itself, for example:

  - `--tp-size` in slime is set using `--rollout-num-gpus-per-engine`.
  - `--model-path` in slime is set using `--hf-checkpoint`.

The way SGLang parameters are integrated into slime can be found in [slime/backends/sglang_utils/arguments.py](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/arguments.py).

### How to Use the Router

slime uses [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) to manage the SGLang servers during the training process. You can configure the address of the [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) using `--sglang-router-ip` and `--sglang-router-port`. If not configured, a router will be started by default within the cluster.

After starting, all SGLang servers will register with the router via the `/add_worker` endpoint. When actually generating data, you only need to send HTTP requests to the router, which will perform load balancing and forward the requests to the servers.

When you configure an external router using `--sglang-router-ip` and `--sglang-router-port`, slime will not start an internal router. Instead, it will register all its servers with this external router. You can then use this external router's address to implement more complex data generation workflows. Note that the router supports OpenAI-compatible APIs.

### Advanced Engine Configuration (--sglang-config)

For advanced deployments, you can use `--sglang-config` with a YAML file to configure server groups, multi-model serving, and selective weight updates.

**Multi-model deployment** allows serving multiple models simultaneously (e.g., an actor model that receives weight updates and a frozen reference/reward model):

```yaml
sglang:
  - name: actor
    update_weights: true          # receives weight updates from training (default)
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4
  - name: ref
    model_path: /path/to/ref_model
    update_weights: false          # frozen, no weight updates
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

Each model gets its own router. The per-model router info is accessible via `args.sglang_model_routers` (a dict mapping model name to `(ip, port)` tuples). Custom rollout functions can use `get_model_url(args, "ref")` from `slime.rollout.sglang_rollout` to route requests to a specific model.

**Server group features:**
- `worker_type`: `regular`, `prefill`, `decode`, or `placeholder` (reserves GPU slots without creating engines)
- `overrides`: Dict of SGLang `ServerArgs` field overrides applied on top of `--sglang-*` CLI args
- `num_gpus_per_engine`: Per-group TP size override

