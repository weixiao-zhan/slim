# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import time
import traceback
from pathlib import Path

import torch

from slim.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def profile_rollout(args, rollout_id):
    """
    Trace one rollout step's generation and rm phase with VizTracer.
    """
    if "rollout" not in args.profile_target or not (args.profile_step_start <= rollout_id < args.profile_step_end):
        yield
        return

    from viztracer import VizTracer

    out_dir = Path(args.profile_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    out_path = out_dir / f"rollout_rank{rank}_rollout{rollout_id}.json.gz"

    # min_duration=1000us=1ms drops trivial frames so the trace stays small;
    tracer = VizTracer(log_async=True, min_duration=1000, output_file=str(out_path), verbose=0)
    tracer.start()
    try:
        yield
    finally:
        tracer.stop()
        tracer.save()
        logger.info(f"Wrote rollout trace to {out_path} (open in chrome://tracing or https://ui.perfetto.dev)")


class TrainProfiler:
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profilers = []

        # Performance profiler: enabled by listing the target in --profile-target.
        if "train_overall" in args.profile_target:
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        # Memory recorder: enabled by listing backend(s) in --memory-recorder. Independent of
        # --profile-target; runs for the whole train loop.
        self._memory_profilers = [_BaseMemoryProfiler.create(r, args) for r in args.memory_recorder]
        for mp in self._memory_profilers:
            mp.start()

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (s := self.args.memory_snapshot_num_steps) is not None and rollout_id == s - 1:
            for mp in self._memory_profilers:
                mp.stop()

    def iterate_train_pg(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_pg")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if name not in args.profile_target:
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _on_trace_ready(args, name):
    def handler(prof):
        rank = torch.distributed.get_rank()
        out_dir = Path(args.profile_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / f"{name}_rank_{rank}_time{time.time()}.json.gz"
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Wrote chrome trace to {trace_path}")

    return handler


def _create_torch_profiler(args, name):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=_on_trace_ready(args, name),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(recorder, args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[recorder]
        return c(args)

    # Subclasses set this so the dump filename reflects the backend.
    _suffix = "pickle"

    def __init__(self, args):
        out_dir = Path(args.profile_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()
        self._path_dump = out_dir / f"memory_snapshot_time{time.time()}_rank{rank}.{self._suffix}"

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    _suffix = "pickle"

    def start(self):
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    _suffix = "bin"

    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, (
            "memray requires --memory-snapshot-num-steps (it dumps on stop, not on OOM)."
        )

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
