# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
from argparse import Namespace

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from slim.rollout.sglang_rollout import GenerateState, RolloutGroup, generate_and_rm_group
from slim.utils.http_utils import get, init_http_client, post
from slim.utils.logging_utils import configure_logger
from slim.utils.misc import SingletonMeta

__all__ = ["RolloutWorker", "RolloutWorkerPool", "offload_bulk_fields"]

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def offload_bulk_fields(group: RolloutGroup) -> None:
    """Put each trajectory's processor outputs into this node's object store."""
    for episode in group.episodes:
        for trajectory in episode.trajectories:
            if trajectory.multimodal_inputs is not None:
                trajectory.multimodal_inputs = ray.put(trajectory.multimodal_inputs)


@ray.remote
class RolloutWorker:
    def __init__(self, args: Namespace, concurrency: int):
        configure_logger()
        init_http_client(concurrency)
        self.args = args
        self.state = GenerateState(args, concurrency)

    async def run_group(self, group: RolloutGroup, evaluation: bool) -> RolloutGroup:
        group = await generate_and_rm_group(self.args, group, evaluation=evaluation)
        offload_bulk_fields(group)
        return group

    async def abort(self) -> None:
        self.state.aborted = True

    async def reset(self) -> None:
        self.state.reset()


class RolloutWorkerPool(metaclass=SingletonMeta):
    """One RolloutWorker per alive node, fed groups by episode load."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) >= 1]
        total = args.rollout_concurrency_per_replica * args.rollout_num_gpus // args.rollout_num_gpus_per_replica
        self.concurrency = math.ceil(total / len(nodes))
        self.workers = [
            RolloutWorker.options(
                num_cpus=1,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                # A worker holds at most `concurrency` groups, plus abort and reset.
                max_concurrency=self.concurrency + 2,
            ).remote(args, self.concurrency)
            for node in nodes
        ]
        ray.get([worker.__ray_ready__.remote() for worker in self.workers])
        logger.info(f"Started {len(self.workers)} rollout workers with concurrency {self.concurrency} each")

        self.loads = [0] * len(self.workers)
        self.condition = asyncio.Condition()
        self.reset()

    def reset(self) -> None:
        self.pendings: set[asyncio.Task] = set()
        self.aborted = False

    async def run_group(self, group: RolloutGroup, evaluation: bool) -> RolloutGroup:
        async with self.condition:
            await self.condition.wait_for(lambda: self.aborted or min(self.loads) < self.concurrency)
            if self.aborted:
                return group
            index = self.loads.index(min(self.loads))
            self.loads[index] += len(group.episodes)
        try:
            return await self.workers[index].run_group.remote(group, evaluation)
        finally:
            async with self.condition:
                self.loads[index] -= len(group.episodes)
                # Each admitted group adds at least one episode, so freed slots admit at most this many waiters.
                self.condition.notify(len(group.episodes))

    def submit_generate_tasks(self, groups: list[RolloutGroup]) -> None:
        for group in groups:
            self.pendings.add(asyncio.create_task(self.run_group(group, evaluation=False)))

    async def abort(self) -> list[dict]:
        aborted_examples = []

        async with self.condition:
            self.aborted = True
            self.condition.notify_all()
        await asyncio.gather(*[worker.abort.remote() for worker in self.workers])

        response = await get(f"http://{self.args.router_ip}:{self.args.router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

        while self.pendings:
            logger.info(f"Abort request for {urls}")
            await asyncio.gather(
                *[post(f"{url}/abort_request", {"abort_all": True}, max_retries=1) for url in urls],
                return_exceptions=True,
            )
            done, self.pendings = await asyncio.wait(self.pendings, return_when=asyncio.ALL_COMPLETED, timeout=1)

            # Recycle aborted/incomplete groups back to the data buffer so they
            # can be retried in a later rollout.  Only groups explicitly rejected
            # by the dynamic filter (e.g. zero-std) are truly discarded.
            for task in done:
                group = task.result()
                if not group.completed:
                    aborted_examples.append(group.example)

        await asyncio.gather(*[worker.reset.remote() for worker in self.workers])
        self.reset()
        return aborted_examples
