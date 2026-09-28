# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from types import SimpleNamespace

import pytest
import ray
import torch

import slim.rollout.sglang_rollout as sglang_rollout
import slim.rollout.worker as worker_module
from slim.ray import rollout
from slim.rollout.sglang_rollout import RolloutGroup
from slim.rollout.worker import RolloutWorkerPool, offload_bulk_fields
from slim.utils import processing_utils
from slim.utils.data import process_rollout_data
from slim.utils.misc import SingletonMeta
from slim.utils.trajectory_batch import build_dp_batches
from slim.utils.types import Episode, Trajectory

NUM_GPUS = 0


class LocalHandle:
    """Exposes an in-process object through the actor call shape `handle.method.remote(...)`."""

    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        method = getattr(self.instance, name)
        return SimpleNamespace(remote=lambda *args, **kwargs: method(*args, **kwargs))


class FakeWorkerClass:
    """Stands in for the `RolloutWorker` actor class and records placement options."""

    def __init__(self, make_worker):
        self.make_worker = make_worker
        self.options_calls = []

    def options(self, **kwargs):
        self.options_calls.append(kwargs)
        return SimpleNamespace(remote=lambda args, concurrency: LocalHandle(self.make_worker(args, concurrency)))


class BlockingWorker:
    def __init__(self, index: int, calls: list, releases: dict):
        self.index = index
        self.calls = calls
        self.releases = releases

    def __ray_ready__(self):
        return True

    async def run_group(self, group, evaluation):
        self.calls.append((self.index, group.index))
        await self.releases.setdefault(group.index, asyncio.Event()).wait()
        group.completed = True
        return group


def _node(index: int, alive: bool = True, cpus: float = 8.0) -> dict:
    return {"NodeID": f"{index + 1:056x}", "Alive": alive, "Resources": {"CPU": cpus} if cpus else {}}


def _args(**overrides):
    values = dict(
        rollout_concurrency_per_replica=4,
        rollout_num_gpus=2,
        rollout_num_gpus_per_replica=1,
        router_ip="127.0.0.1",
        router_port=30000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _group(index: int, num_episodes: int, prompt: str = "") -> RolloutGroup:
    example = {"prompt": prompt or f"prompt-{index}"}
    return RolloutGroup(index=index, example=example, episodes=[Episode.from_example(example) for _ in range(num_episodes)])


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
def fake_cluster(monkeypatch):
    """Patch node discovery and actor creation so the pool builds over in-process workers."""
    monkeypatch.setattr(SingletonMeta, "_instances", {})
    monkeypatch.setattr(worker_module.ray, "get", lambda refs: refs)

    def install(nodes, make_worker):
        monkeypatch.setattr(worker_module.ray, "nodes", lambda: nodes)
        worker_class = FakeWorkerClass(make_worker)
        monkeypatch.setattr(worker_module, "RolloutWorker", worker_class)
        return worker_class

    return install


@pytest.fixture
def fake_router(monkeypatch):
    posts = []

    async def get(url):
        return {"workers": [{"url": "http://engine-0"}, {"url": "http://engine-1"}]}

    async def post(url, payload, max_retries=60, headers=None):
        posts.append((url, payload))
        return {}

    monkeypatch.setattr(worker_module, "get", get)
    monkeypatch.setattr(worker_module, "post", post)
    return posts


@pytest.mark.unit
def test_pool_places_one_worker_per_alive_cpu_node(fake_cluster):
    nodes = [_node(0), _node(1), _node(2), _node(3, alive=False), _node(4, cpus=0)]
    worker_class = fake_cluster(nodes, lambda args, concurrency: BlockingWorker(0, [], {}))

    pool = RolloutWorkerPool(_args(rollout_concurrency_per_replica=5, rollout_num_gpus=4, rollout_num_gpus_per_replica=2))

    assert len(pool.workers) == 3
    assert pool.concurrency == 4  # ceil(5 * 4 // 2 / 3)
    assert [call["scheduling_strategy"].node_id for call in worker_class.options_calls] == [node["NodeID"] for node in nodes[:3]]


@pytest.mark.unit
def test_pool_dispatches_to_least_loaded_worker_below_capacity(fake_cluster):
    calls, releases, workers = [], {}, []

    def make_worker(args, concurrency):
        workers.append(BlockingWorker(len(workers), calls, releases))
        return workers[-1]

    fake_cluster([_node(0), _node(1)], make_worker)

    async def run():
        pool = RolloutWorkerPool(_args())
        assert pool.concurrency == 4

        tasks = [asyncio.create_task(pool.run_group(_group(i, 3), evaluation=False)) for i in range(5)]
        await _settle()
        assert calls == [(0, 0), (1, 1), (0, 2), (1, 3)]
        assert pool.loads == [6, 6]
        assert not tasks[4].done()

        releases[2].set()
        await _settle()
        assert calls[-1] == (0, 4)
        assert pool.loads == [6, 6]

        for index in range(5):
            releases.setdefault(index, asyncio.Event()).set()
        groups = await asyncio.gather(*tasks)
        assert pool.loads == [0, 0]
        assert [group.index for group in groups] == list(range(5))
        assert all(group.completed for group in groups)

    asyncio.run(run())


class _Tokenizer:
    def decode(self, token_ids):
        return ",".join(str(token_id) for token_id in token_ids)


@pytest.mark.unit
def test_generate_rollout_drains_slow_environment_step_on_abort(monkeypatch, fake_cluster, fake_router):
    monkeypatch.setattr(processing_utils, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(processing_utils, "load_processor", lambda *args, **kwargs: None)
    env_steps = []

    async def generate(state, episode):
        episode.trajectories.append(Trajectory(token_ids=[1, 2], loss_mask=[1], rollout_log_probs=[0.0]))
        if episode.example["prompt"] == "fast":
            episode.reward = 1.0
            episode.status = Episode.Status.COMPLETED
            return episode
        while True:
            if state.aborted:
                episode.status = Episode.Status.ABORTED
                return episode
            await asyncio.to_thread(time.sleep, 0.2)
            env_steps.append(episode.example["prompt"])

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    local_worker_class = worker_module.RolloutWorker.__ray_metadata__.modified_class
    fake_cluster([_node(0)], local_worker_class)

    args = _args(
        rollout_num_gpus=1,
        hf_checkpoint="unused",
        apply_chat_template_kwargs=None,
        sglang_dp_size=None,
        use_rollout_routing_replay=False,
        group_rm=False,
        custom_generate_function_path=None,
        rollout_seed=0,
        rollout_global_dataset=True,
        rollout_group_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        rollout_batch_size=1,
        over_sampling_batch_size=3,
        n_samples_per_prompt=2,
        max_context_len=16,
    )
    examples = [{"prompt": "fast"}, {"prompt": "slow"}, {"prompt": "slow-waiting"}]

    async def run():
        pool = RolloutWorkerPool(args)
        output, aborted_examples = await sglang_rollout.generate_rollout_async(args, 0, lambda n: [examples.pop(0) for _ in range(n)])
        return pool, output, aborted_examples

    pool, output, aborted_examples = asyncio.run(run())

    assert [episode.example["prompt"] for episode in output.episodes] == ["fast", "fast"]
    assert all(episode.status == Episode.Status.COMPLETED for episode in output.episodes)
    assert sorted(example["prompt"] for example in aborted_examples) == ["slow", "slow-waiting"]
    assert "slow" in env_steps
    assert fake_router
    assert pool.loads == [0]
    assert not pool.pendings and not pool.aborted
    assert not pool.workers[0].instance.state.aborted


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=1, include_dashboard=False)
    yield
    ray.shutdown()


def _multimodal_episode() -> Episode:
    trajectory = Trajectory(
        token_ids=[1, 2, 3],
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        multimodal_inputs={"pixel_values": torch.arange(8.0).reshape(2, 4), "image_grid_thw": torch.tensor([[1, 1, 2]])},
    )
    return Episode(trajectories=[trajectory], reward=1.0, episode_index=0, group_index=0)


@pytest.mark.unit
def test_multimodal_inputs_round_trip_by_reference(local_ray):
    episode = _multimodal_episode()
    expected = dict(episode.trajectory.multimodal_inputs)
    offload_bulk_fields(RolloutGroup(index=0, example={}, episodes=[episode]))
    assert isinstance(episode.trajectory.multimodal_inputs, ray.ObjectRef)

    episode.finalize_source_token_alignment()
    batches = build_dp_batches([episode], dp_size=2, num_steps=1, loss_normalization_unit="episode", pad_token_id=0, balance_data=False)
    refs = [ray.put(batch) for batch in batches]

    fetched = [trajectory for rank in range(2) for trajectory in process_rollout_data(refs, rank, 2).trajectories]
    real = [trajectory for trajectory in fetched if trajectory.multimodal_inputs is not None]
    assert len(real) == 1 and len(fetched) == 2
    assert set(real[0].multimodal_inputs) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(real[0].multimodal_inputs[key], value)


@pytest.mark.unit
def test_debug_rollout_data_saves_resolved_multimodal_inputs(local_ray, tmp_path):
    episode = _multimodal_episode()
    expected = dict(episode.trajectory.multimodal_inputs)
    offload_bulk_fields(RolloutGroup(index=0, example={}, episodes=[episode]))
    manager = SimpleNamespace(args=SimpleNamespace(save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt")))
    save = rollout.RolloutManager.__ray_metadata__.modified_class._save_debug_rollout_data

    save(manager, [episode], rollout_id=3, evaluation=False)
    save(manager, {"eval-set": [episode]}, rollout_id=3, evaluation=True)

    assert isinstance(episode.trajectory.multimodal_inputs, ray.ObjectRef)
    for name in ("rollout_3.pt", "rollout_eval_3.pt"):
        saved = torch.load(tmp_path / name, weights_only=False)["episodes"][0]["trajectories"][0]["multimodal_inputs"]
        for key, value in expected.items():
            torch.testing.assert_close(saved[key], value)

    loaded = rollout._load_debug_rollout_episodes(str(tmp_path / "rollout_{rollout_id}.pt"), 3)
    assert isinstance(loaded[0].trajectory.multimodal_inputs, ray.ObjectRef)
    for key, value in expected.items():
        torch.testing.assert_close(ray.get(loaded[0].trajectory.multimodal_inputs)[key], value)
