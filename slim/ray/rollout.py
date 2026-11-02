# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import itertools
import logging
import multiprocessing
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import ray
import requests
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang_router.launch_router import RouterArgs

from slim.backends.sglang_utils.sglang_engine import SGLangEngine
from slim.rollout.worker import RolloutWorkerPool
from slim.utils import logging_utils
from slim.utils.health_monitor import RolloutHealthMonitor
from slim.utils.http_utils import _wrap_ipv6, get_host_info, init_http_client, run_router
from slim.utils.logging_utils import build_sglang_metrics_endpoints, configure_logger, init_tracking
from slim.utils.metric_utils import (
    compute_consensus,
    compute_majority_vote,
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
)
from slim.utils.misc import get_free_port, load_function
from slim.utils.processing_utils import load_tokenizer
from slim.utils.profile_utils import profile_rollout
from slim.utils.trajectory_batch import build_dp_batches
from slim.utils.types import BULK_FIELDS, Episode, Trajectory

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Placement, rollout_node_ids

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _load_debug_rollout_episodes(path_template: str, rollout_id: int) -> list[Episode]:
    path = Path(path_template.format(rollout_id=rollout_id))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"debug rollout data at {path} must contain an episodes list")
    episodes = []
    for record in records:
        record = dict(record)
        record["trajectories"] = [Trajectory(**traj) for traj in record["trajectories"]]
        for trajectory in record["trajectories"]:
            for name in BULK_FIELDS:
                if (value := getattr(trajectory, name)) is not None:
                    setattr(trajectory, name, ray.put(value))
        episodes.append(Episode(**record))
    logger.info("Loaded %d debug rollout episodes from %s", len(episodes), path)
    return episodes


@dataclasses.dataclass
class RolloutServer:
    """The SGLang engines of the policy behind one router.

    All engines share one TP size, nodes per replica and placement group.
    """

    args: Any
    placement: Placement
    all_engines: list
    router_ip: str
    router_port: int
    router_prometheus_port: int | None
    engine_urls: list[str | None] = dataclasses.field(init=False)
    num_new_engines: int = 0

    def __post_init__(self):
        self.engine_urls = [None] * len(self.all_engines)

    @property
    def nodes_per_replica(self):
        return max(1, self.args.rollout_num_gpus_per_replica // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_replica]

    @property
    def engine_actor_ranks(self) -> list[list[int] | None]:
        """Actor ranks on the GPUs of each node-0 engine, parallel to ``engines``.

        An entry is None if not all GPUs of the engine are actor GPUs.
        """
        num_gpus = self.args.rollout_num_gpus_per_replica
        return [self.placement.actor_ranks(range(j * num_gpus, (j + 1) * num_gpus)) for j in range(len(self.engines))]

    def start_engines(self) -> list:
        """Create the missing engine actors, allocate their ports, and fire ``engine.init()`` without waiting.

        Returns the init ObjectRefs; ``ray.get`` on them blocks until the engines are healthy.
        """
        num_gpus_per_engine = min(self.args.rollout_num_gpus_per_replica, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.placement.role_pg("rollout")

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        base_gpu_ids = {}
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            num_gpus = 0.2
            num_cpus = num_gpus

            gpu_index = i * num_gpus_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(self.args, rank=i, base_gpu_id=base_gpu_id)

            rollout_engines.append((i, rollout_engine))
            base_gpu_ids[i] = base_gpu_id
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return []

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args, rollout_engines=rollout_engines, base_gpu_ids=base_gpu_ids
            )

        init_handles = [engine.init.remote(**addr_and_ports[rank]) for rank, engine in rollout_engines]
        for i, _ in rollout_engines:
            if i % self.nodes_per_replica != 0:
                continue
            self.engine_urls[i] = f"http://{_wrap_ipv6(addr_and_ports[i]['host'])}:{addr_and_ports[i]['port']}"
            if not self.args.rollout_external:
                # The router admits the worker once its /health passes, so it can be added before init finishes.
                requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers", json={"url": self.engine_urls[i]}
                ).raise_for_status()
        return init_handles

    def kill_engine(self, engine_id: int) -> None:
        """Remove ``engines[engine_id]`` from the router, then kill it; ``start_engines`` restarts it."""
        first = engine_id * self.nodes_per_replica
        url = self.engine_urls[first]
        if not self.args.rollout_external:
            router_url = f"http://{self.router_ip}:{self.router_port}"
            try:
                # DELETE takes the router's worker id, so look it up by URL.
                for worker in requests.get(f"{router_url}/workers").json()["workers"]:
                    if worker["url"] == url:
                        requests.delete(f"{router_url}/workers/{worker['id']}").raise_for_status()
            except Exception as e:
                logger.warning(f"Fail to remove {url} from router (e: {e})")
        for i in range(first, first + self.nodes_per_replica):
            engine = self.all_engines[i]
            try:
                ray.get(engine.shutdown.remote())
            except Exception as e:
                logger.warning(f"Fail to shut down engine at index {i} (e: {e})")
            ray.kill(engine)
            self.all_engines[i] = None

    def recover(self):
        """Restart dead engines and wait until they are healthy."""
        dead_indices = [i for i, engine in enumerate(self.all_engines) if engine is None]
        handles = self.start_engines()
        if handles:
            ray.get(handles)
        logger.info(f"Recovered {self.num_new_engines} dead rollout engines")

        # Post-recovery: offload then onload weights for newly created engines that share training GPUs.
        if self.args.rollout_colocate and dead_indices:
            new_engines = [self.all_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_occupation.remote() for engine in new_engines])
            ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines])

    def offload(self):
        """Release the memory occupation of all engines."""
        return ray.get([engine.release_memory_occupation.remote() for engine in self.engines if engine is not None])

    def onload(self, tags: list[str] | None = None):
        """Resume the memory occupation of all engines."""
        return ray.get(
            [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]
        )

    def onload_weights(self):
        """Restore the weights from the CPU backup; ``update_weights`` overwrites them right after."""
        return self.onload([GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        """Resume KV cache and CUDA graphs."""
        return self.onload([GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, placement):
        configure_logger()

        self.placement = placement
        self.args = args

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        self.server: RolloutServer | None = None
        if not self.args.debug_train_only:
            self.server = start_rollout_server(args, placement)
            init_http_client()
            RolloutWorkerPool(args, rollout_node_ids(placement.role_pg("rollout")))

        # Init W&B after the server starts so the router/engine Prometheus
        # endpoints are known and can be registered for scraping.
        init_tracking(args, primary=False, metrics_endpoints=build_sglang_metrics_endpoints(args, self.server))
        self.rollout_id = -1

        self._health_monitor = None
        if self.server is not None:
            self._health_monitor = RolloutHealthMonitor(self.server, args)
            self._health_monitor.start()
        self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.all_engines and self.server.all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if self._health_monitor is not None:
            self._health_monitor.stop()
        engines = self.server.all_engines if self.server else []
        shutdown_refs = [engine.shutdown.remote() for engine in engines if engine is not None]
        for ref in shutdown_refs:
            try:
                ray.get(ref)
            except Exception as e:
                logger.warning(f"Engine shutdown failed during dispose: {e}")
        logging_utils.finish_tracking(self.args)

    @property
    def rollout_engines(self):
        """The node-0 engines."""
        return self.server.engines if self.server else []

    def recover_engines(self) -> None:
        """Pause health monitoring and restart dead engines."""
        self.health_monitoring_pause()
        if self.rollout_id != -1 and self.server is not None:
            self.server.recover()

    def get_num_rollout_per_epoch(self):
        return len(self.data_source) // self.args.rollout_batch_size

    def get_epoch_id(self):
        return self.data_source.epoch_id

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and rollout_id >= 2:
            self._try_ci_fault_injection()
        episodes, metrics = self._get_rollout_episodes(rollout_id=rollout_id)
        self._save_debug_rollout_data(episodes, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, episodes, metrics, time.time() - start_time)
        for index, episode in enumerate(episodes):
            episode.episode_index = index
        return self._split_trajectories_by_dp(episodes)

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            return
        self.health_monitoring_resume()

        result = self.eval_generate_rollout(self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data  # dict[str, list[Episode]]
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        if self.server is not None:
            self.server.offload()

    def onload_weights(self):
        if self.server is not None:
            self.server.onload_weights()

    def onload_kv(self):
        if self.server is not None:
            self.server.onload_kv()

    def get_engines(self):
        """Return the engines, the count of new engines and the actor ranks on each engine (read-only, no recovery).

        All ranks call this after rank-0 has already called
        ``recover_engines``, so ``num_new_engines`` is stable.
        """
        return self.server.engines, self.server.num_new_engines, self.server.engine_actor_ranks

    def clear_num_new_engines(self):
        self.server.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_episodes(self, rollout_id) -> tuple[list[Episode], dict | None]:
        if self.args.load_debug_rollout_data is not None:
            episodes = _load_debug_rollout_episodes(self.args.load_debug_rollout_data, rollout_id)
            metrics = None
        else:
            with profile_rollout(self.args, rollout_id):
                result = self.generate_rollout(self.args, rollout_id, self.data_source, evaluation=False)
            metrics = result.metrics
            episodes = result.episodes

        if not self.args.disable_rollout_trim_samples:
            if len(episodes) % self.args.global_batch_size != 0:
                episodes = self._trim_whole_groups(episodes, self.args.global_batch_size)
            logger.info(f"Final collected {len(episodes)} episodes from rollout to train")

        return episodes, metrics

    @staticmethod
    def _trim_whole_groups(episodes: list[Episode], global_batch_size: int) -> list[Episode]:
        """Drop whole prompt groups so the GRPO baseline keeps complete groups."""
        groups = [
            list(group)
            for _, group in itertools.groupby(episodes, key=lambda episode: episode.group_index)
        ]

        target = (len(episodes) // global_batch_size) * global_batch_size
        kept: list[Episode] = []
        for group in groups:
            if len(kept) + len(group) > target:
                break
            kept.extend(group)
        if len(kept) % global_batch_size:
            raise ValueError(
                f"trimming whole prompt groups left {len(kept)} episodes, which is not a multiple of "
                f"global_batch_size {global_batch_size}; prompt group size must divide it"
            )
        if not kept:
            raise ValueError(
                f"Not enough episodes {len(episodes)} for global_batch_size {global_batch_size}"
            )
        logger.info(f"trim number of episodes from {len(episodes)} to {len(kept)}")
        return kept

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            all_episodes = [ep for episodes in data.values() for ep in episodes] if evaluation else data
            records = [dataclasses.asdict(ep) for ep in all_episodes]
            # Resolve bulk field refs in the saved copy; the live episodes keep their refs.
            ref_fields = [
                (traj, name)
                for record in records
                for traj in record["trajectories"]
                for name in BULK_FIELDS
                if traj[name] is not None
            ]
            values = ray.get([traj[name] for traj, name in ref_fields])
            for (traj, name), value in zip(ref_fields, values, strict=True):
                traj[name] = value

            torch.save(dict(rollout_id=rollout_id, episodes=records), path)

    def _split_trajectories_by_dp(self, episodes: list[Episode]) -> list:
        """Flatten episodes to trajectories and split them across DP ranks."""
        # --critic-train-only runs no actor ranks; otherwise the actor and critic GPU counts are equal.
        train_num_gpus = self.args.critic_num_gpus if self.args.critic_train_only else self.args.actor_num_gpus
        dp_size = train_num_gpus // self.args.context_parallel_size
        num_steps = max(len(episodes) // self.args.global_batch_size, 1)
        batches = build_dp_batches(
            episodes,
            dp_size=dp_size,
            num_steps=num_steps,
            loss_normalization_unit=self.args.loss_normalization_unit,
            pad_token_id=self._pad_token_id,
            balance_data=self.args.balance_data,
            top_k=self.args.use_score_centering,
            routing_replay_shape=self.args.routing_replay_shape,
        )
        logger.info(
            "Split %d episodes into %d trajectories per DP rank across %d optimizer steps",
            len(episodes),
            len(batches[0]),
            num_steps,
        )
        return [ray.put(batch) for batch in batches]

    @functools.cached_property
    def _pad_token_id(self) -> int:
        return load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True).pad_token_id


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, rollout_engines, base_gpu_ids):
    """Allocate the server, NCCL and dist-init ports of each engine on its own node.

    Each engine searches from a block keyed by its first local GPU. Jobs that share a node hold disjoint GPUs, so
    their engines never probe the same free ports before SGLang binds them. The blocks stay below the ephemeral range
    (32768+) and clear of the low Ray worker ports (10002+).
    """
    nodes_per_replica = max(1, args.rollout_num_gpus_per_replica // args.num_gpus_per_node)
    addr_and_ports: dict[int, dict] = {}
    # Engines of one replica come in order, and the replica's node-0 engine owns the dist-init address.
    for rank, engine in rollout_engines:
        owns_dist_init = rank % nodes_per_replica == 0
        # Server and NCCL ports, then the dist-init port followed by the 4 + dp_size dp_attention ports.
        spans = (1, 1, 30 + args.sglang_dp_size) if owns_dist_init else (1, 1)
        cursor = 15000 + 100 * base_gpu_ids[rank]
        ports = []
        for consecutive in spans:
            host, found = ray.get(
                engine._get_current_node_ip_and_free_port.remote(start_port=cursor, consecutive=consecutive)
            )
            ports.append(found)
            cursor = found + consecutive
        if owns_dist_init:
            dist_init_addr = f"{host}:{ports[2]}"
        addr_and_ports[rank] = dict(host=host, port=ports[0], nccl_port=ports[1], dist_init_addr=dist_init_addr)
        logger.info(f"Ports for engine {rank}: {addr_and_ports[rank]}")
    return addr_and_ports


def _start_router(args) -> tuple[str, int, int | None]:
    """Start sgl router and return (router_ip, router_port, prometheus_port).

    If ``args.router_ip`` is already set (e.g. by the user), skip launching and return the existing values.

    ``prometheus_port`` is the port the router exposes ``/metrics`` on; it is
    ``None`` for an externally-managed router (we did not launch it).
    """
    if args.router_ip is not None:
        return args.router_ip, args.router_port, None

    router_ip = _wrap_ipv6(get_host_info()[1])
    router_port = args.router_port
    if router_port is None:
        router_port = get_free_port(start_port=random.randint(3000, 4000))

    prometheus_port = get_free_port(start_port=random.randint(4000, 5000))

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = prometheus_port
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.router_request_timeout_secs

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port} (prometheus_port={prometheus_port})")
    return router_ip, router_port, prometheus_port


def start_rollout_server(args, placement) -> RolloutServer:
    """Start the router, then the SGLang engines behind it, and wait until the engines are healthy."""
    router_ip, router_port, router_prometheus_port = _start_router(args)
    # Rollout functions reach the engines through the router address in args.
    args.router_ip, args.router_port = router_ip, router_port

    num_engines = args.rollout_num_gpus // min(args.rollout_num_gpus_per_replica, args.num_gpus_per_node)
    server = RolloutServer(
        args=args,
        placement=placement,
        all_engines=[None] * num_engines,
        router_ip=router_ip,
        router_port=router_port,
        router_prometheus_port=router_prometheus_port,
    )
    ray.get(server.start_engines())
    return server


def _log_eval_rollout_data(
    rollout_id, args, data: dict[str, list[Episode]], extra_metrics: dict[str, Any] | None = None
):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key, episodes in data.items():
        rewards = [ep.get_reward_value() for ep in episodes]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        log_dict |= dict_add_prefix(_compute_episode_metrics(args, episodes), f"eval/{key}/")
        if (rollout_log_probs := _compute_rollout_log_probs_metric(episodes)) is not None:
            log_dict[f"eval/{key}/rollout_log_probs"] = rollout_log_probs
        if args.eval_log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(flat_rewards=rewards, group_size=args.eval_n_samples_per_prompt),
                f"eval/{key}/sampling/",
            )
            log_dict |= dict_add_prefix(
                compute_majority_vote(flat_rewards=rewards, group_size=args.eval_n_samples_per_prompt),
                f"eval/{key}/sampling/",
            )
            log_dict |= dict_add_prefix(
                compute_consensus(flat_rewards=rewards, group_size=args.eval_n_samples_per_prompt),
                f"eval/{key}/sampling/",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict)

    _save_eval_rollout(rollout_id, args, data)

    return log_dict


def _save_eval_rollout(rollout_id, args, data: dict[str, list[Episode]]):
    if (path_template := args.eval_save_rollout) is None:
        return
    for key, episodes in data.items():
        path = Path(path_template.format(rollout_id=rollout_id, dataset_key=key))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([[t.token_ids for t in ep.trajectories] for ep in episodes], path)
        logger.info(f"Saved eval rollout to {path}")


def _log_rollout_data(rollout_id, args, episodes: list[Episode], rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, episodes, rollout_extra_metrics, rollout_time):
            return

    log_dict = {**(rollout_extra_metrics or {})}
    if episodes:
        rewards = [ep.get_reward_value() for ep in episodes]
        log_dict["rollout/reward"] = sum(rewards) / len(rewards)
    if (rollout_log_probs := _compute_rollout_log_probs_metric(episodes)) is not None:
        log_dict["rollout/rollout_log_probs"] = rollout_log_probs
    log_dict |= dict_add_prefix(_compute_episode_metrics(args, episodes), "rollout/")
    log_dict |= dict_add_prefix(_compute_perf_metrics(episodes, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict)


def _compute_rollout_log_probs_metric(episodes: list[Episode]) -> float | None:
    """Mean over trajectories of each one's mean rollout log-prob across response tokens.

    This belongs to rollout logging because these log-probs are emitted by the
    rollout engine before actor-side training diagnostics are computed.
    """
    per_trajectory_means = []
    for trajectory in (t for ep in episodes for t in ep.trajectories):
        if trajectory.rollout_log_probs is None:
            continue
        log_probs = torch.as_tensor(trajectory.rollout_log_probs, dtype=torch.float32)
        if trajectory.loss_mask is None:
            mask = torch.ones_like(log_probs, dtype=torch.float32)
        else:
            mask = torch.as_tensor(trajectory.loss_mask, dtype=torch.float32)
        denom = mask.sum().clamp_min(1)
        per_trajectory_means.append(((log_probs * mask).sum() / denom).item())
    if not per_trajectory_means:
        return None
    return sum(per_trajectory_means) / len(per_trajectory_means)


def _compute_episode_metrics(args, episodes: list[Episode]):
    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics([ep.token_count for ep in episodes]), "context_len/")
    log_dict |= dict_add_prefix(
        compute_statistics([len(ep.trajectories) for ep in episodes]), "trajectories/"
    )
    log_dict |= _compute_zero_std_metrics(args, episodes)
    log_dict["context_len/repetition_frac"] = sum(
        int(any(has_repetition(t.generated_text or "") for t in ep.trajectories)) for ep in episodes
    ) / max(len(episodes), 1)
    log_dict["context_len/truncated_ratio"] = sum(int(ep.status == Episode.Status.TRUNCATED) for ep in episodes) / max(
        len(episodes), 1
    )
    return log_dict


def _compute_perf_metrics(episodes: list[Episode], rollout_time):
    non_generation_time = [ep.non_generation_time for ep in episodes]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    return log_dict


def _compute_zero_std_metrics(args, episodes: list[Episode]):
    if args.advantage_estimator == "ppo_gae":
        return {}

    groups: dict[Any, list[float]] = {}
    for episode in episodes:
        groups.setdefault(episode.group_index, []).append(episode.get_reward_value())

    counts = Counter(
        str(round(rewards[0], 1))
        for rewards in groups.values()
        if all(reward == rewards[0] for reward in rewards)
    )
    return {f"zero_std/count_{label}": count for label, count in counts.items()}
