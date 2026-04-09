import asyncio
import inspect
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import sglang_router
import torch
from packaging.version import parse
from tqdm import tqdm

from slim.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slim.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slim.utils.async_utils import run
from slim.utils.eval_config import EvalDatasetConfig
from slim.utils.http_utils import get, post
from slim.utils.misc import SingletonMeta, load_function
from slim.utils.types import Episode

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


@dataclass
class RolloutGroup:
    index: int
    example: dict
    episodes: list[Episode]
    completed: bool = False


class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args: Namespace) -> None:
        from slim.utils.processing_utils import load_processor, load_tokenizer

        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        self.semaphore = asyncio.Semaphore(concurrency)
        self.chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_tokens=args.rollout_max_context_len,  # total context budget
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0
        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = candidates[torch.randint(len(candidates), (1,)).item()]
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings: set[asyncio.Task] = set()
        self.aborted = False

    def submit_generate_tasks(self, groups: list[RolloutGroup]) -> None:
        for group in groups:
            self.pendings.add(
                asyncio.create_task(
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(groups)


def _episode_full_text(args: Namespace, episode: Episode) -> str:
    tokens = episode.tokens
    if not len(tokens):
        return ""
    ids = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
    return GenerateState(args).tokenizer.decode(ids, skip_special_tokens=args.rollout_skip_special_tokens)


def _examples_to_rollout_groups(examples: list[dict], args) -> list[RolloutGroup]:
    groups = []
    for index, example in enumerate(examples):
        groups.append(
            RolloutGroup(
                index=index,
                example=example,
                episodes=[Episode.from_example(example) for _ in range(args.n_samples_per_prompt)],
            )
        )
    return groups


def _prepare_episode_tokens(args: Namespace, episode: Episode, max_context_tokens: int) -> None:
    """Tokenize prompt into episode.tokens if not already set.

    Sets episode._max_tokens (total context budget) on first call.
    """
    state = GenerateState(args)

    if episode.tokens:
        return

    prompt = episode.example.get("prompt", "")
    tools = episode.example.get("tools")
    multimodal_inputs = episode.example.get("multimodal_inputs")

    if episode.has_multimodal and state.processor is None:
        raise RuntimeError("Multimodal examples require a processor, but none could be loaded for this checkpoint.")

    if isinstance(prompt, list) and state.processor:
        # VLM processor + conversation
        prompt_text = state.processor.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **state.chat_template_kwargs,
        )
        mm = {k: v for k, v in (multimodal_inputs or {}).items() if v}
        processor_output = state.processor(text=prompt_text, **mm, return_tensors="pt")
        prompt_ids = processor_output["input_ids"][0].tolist()
        episode.multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor)
        } or None
    elif isinstance(prompt, list):
        # LLM tokenizer + conversation
        prompt_ids = state.tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **state.chat_template_kwargs,
        )
    else:
        # Raw string prompt
        prompt_ids = state.tokenizer.encode(prompt, add_special_tokens=False)

    episode.tokens = prompt_ids
    edge_len = max(len(episode.tokens) - 1, 0)
    episode.loss_mask = [0] * edge_len
    episode.rollout_log_probs = [0.0] * edge_len
    episode._max_tokens = max_context_tokens



async def generate(args: Namespace, episode: Episode, sampling_params: dict[str, Any]) -> Episode:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    from slim.utils.processing_utils import encode_image_for_rollout_engine

    assert episode.status in [Episode.Status.PENDING, Episode.Status.ABORTED], f"Episode status is {episode.status}"

    _prepare_episode_tokens(args, episode, sampling_params["max_tokens"])

    assert episode.rollout_log_probs is not None
    max_new_tokens = episode._max_tokens - len(episode.tokens)

    if max_new_tokens <= 0:
        episode.status = Episode.Status.TRUNCATED
        return episode

    sglang_params = {k: v for k, v in sampling_params.items() if k != "max_tokens"}
    sglang_params["max_new_tokens"] = max_new_tokens

    payload = {
        "input_ids": episode.tokens,
        "sampling_params": sglang_params,
        "return_logprob": True,
    }
    mm = episode.example.get("multimodal_inputs")
    image_inputs = mm.get("images") if mm else None
    if episode.has_multimodal and image_inputs:
        image_data = image_inputs
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in image_data]

    headers = None
    if getattr(args, "router_policy", None) == "consistent_hashing" and episode.session_id:
        headers = {"X-SMG-Routing-Key": episode.session_id}

    output = await post(url, payload, headers=headers)

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []

    old_edge_len = episode.num_edges
    episode.tokens.extend(new_tokens)
    new_edge_len = episode.num_edges
    added_edges = new_edge_len - old_edge_len

    assert episode.loss_mask is not None and episode.rollout_log_probs is not None
    episode.loss_mask.extend([1] * added_edges)
    episode.rollout_log_probs.extend(new_log_probs)

    episode.ensure_edge_alignment()
    episode.update_from_meta_info(args, output["meta_info"])
    return episode


async def generate_and_rm(
    args: Namespace,
    episode: Episode,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Episode:
    if episode.status in [Episode.Status.COMPLETED, Episode.Status.TRUNCATED]:
        return episode

    state = GenerateState(args)
    async with state.semaphore:
        if state.aborted:
            episode.status = Episode.Status.ABORTED
            return episode

        with state.dp_rank_context():
            custom_func_path = episode.generate_function_path or args.custom_generate_function_path
            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    episode = await custom_generate_func(args, episode, sampling_params, evaluation=evaluation)
                else:
                    episode = await custom_generate_func(args, episode, sampling_params)
            else:
                episode = await generate(args, episode, sampling_params)

    if not args.group_rm and episode.status != Episode.Status.ABORTED and episode.reward is None:
        from .rm_hub import async_rm

        episode.reward = await async_rm(args, episode)
    return episode


async def generate_and_rm_group(
    args: Namespace, group: RolloutGroup, sampling_params: dict[str, Any], evaluation: bool = False
) -> RolloutGroup:
    state = GenerateState(args)
    if state.aborted:
        return group

    for episode in group.episodes:
        if episode.session_id is None:
            episode.session_id = str(uuid.uuid4())

    tasks = []
    for idx, episode in enumerate(group.episodes):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        tasks.append(asyncio.create_task(generate_and_rm(args, episode, current_sampling_params, evaluation=evaluation)))

    group.episodes = await asyncio.gather(*tasks)
    if not state.aborted and args.group_rm:
        from .rm_hub import batched_async_rm

        rewards = await batched_async_rm(args, group.episodes)
        for episode, reward in zip(group.episodes, rewards, strict=False):
            episode.reward = reward

    group.completed = all(ep.status != Episode.Status.ABORTED for ep in group.episodes)
    return group


async def abort(args: Namespace) -> list[dict]:
    aborted_examples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        # Recycle aborted/incomplete groups back to the data buffer so they
        # can be retried in a later rollout.  Only groups explicitly rejected
        # by the dynamic filter (e.g. zero-std) are truly discarded.
        for task in done:
            group = task.result()
            if not group.completed:
                aborted_examples.append(group.example)

    return aborted_examples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, get_examples: Callable[[int], list[dict]]
) -> tuple[RolloutFnTrainOutput, list[dict]]:
    assert args.rollout_global_dataset
    state = GenerateState(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None

    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size
    kept_groups: list[RolloutGroup] = []
    all_groups: list[RolloutGroup] = []
    do_print = True
    next_group_index = 0
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")

    while len(kept_groups) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            examples = get_examples(args.over_sampling_batch_size)
            groups = _examples_to_rollout_groups(examples, args)
            for group in groups:
                group.index = next_group_index
                next_group_index += 1
            state.submit_generate_tasks(groups)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: RolloutGroup = task.result()

            if do_print:
                ep = group.episodes[0]
                logger.info(
                    f"First rollout sample: {[_episode_full_text(args, ep)]}, label: {ep.example.get('label')}, reward: {ep.reward}",
                )
                do_print = False

            assert len(group.episodes) == args.n_samples_per_prompt
            all_groups.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group.episodes)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(kept_groups) < target_data_size:
                kept_groups.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    ep = kept_groups[-1].episodes[0]
    logger.info(
        f"Finish rollout: {[_episode_full_text(args, ep)]}, label: {ep.example.get('label')}, reward: {ep.reward}",
    )

    aborted_examples = await abort(args)

    assert len(kept_groups) == args.rollout_batch_size, f"Got {len(kept_groups)} samples, expected {args.rollout_batch_size}"
    kept_groups.sort(key=lambda group: group.index)
    state.reset()

    episodes = [episode for group in kept_groups for episode in group.episodes]
    for episode in episodes:
        episode.ensure_edge_alignment()
        episode.freeze()

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, kept_groups)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, [group.episodes for group in all_groups], get_examples)

    return RolloutFnTrainOutput(episodes=episodes, metrics=metric_gatherer.collect()), aborted_examples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    combined = {}
    for result in results_list:
        combined.update(result)
    return RolloutFnEvalOutput(data=combined)


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, list[Episode]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET
    from slim.utils.data import load_hf_dataset

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint,)
    if cache_key not in EVAL_PROMPT_DATASET:
        EVAL_PROMPT_DATASET[cache_key] = load_hf_dataset(dataset_cfg.path)
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_tokens=dataset_cfg.max_context_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    for raw_row in dataset:
        for j in range(dataset_cfg.eval_n_samples_per_prompt):
            episode = Episode.from_example(raw_row)
            episode.example["metadata"] = dataset_cfg.inject_metadata(episode.example.get("metadata") or {})
            episode.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(asyncio.create_task(generate_and_rm(args, episode, sampling_params=sampling_params, evaluation=True)))

    episodes = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        episode = await coro
        if do_print:
            logger.info(f"eval_rollout_single_dataset example data: {[_episode_full_text(args, episode)]} reward={episode.reward}")
            do_print = False
        episode.ensure_edge_alignment()
        episode.freeze()
        episodes.append(episode)
        pbar.update(1)
    pbar.close()

    return {dataset_cfg.name: episodes}


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))

    output, aborted_examples = run(generate_rollout_async(args, rollout_id, data_source.get_examples))
    data_source.add_examples(aborted_examples)
    return output
