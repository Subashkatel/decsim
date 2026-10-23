"""A device's execution slots, shared by every pool placed on it.

Decoders on one GPU or CPU host share its compute: CUDA-Q QEC runs N
decoder replicas as N CUDA streams on one GPU, "an interchangeable pool"
(cudaqx libs/qec/lib/realtime/host_side_dispatcher_design.md:27), with
no per-decoder SM partition, and Google decodes in real time only "when
provided exclusive access to a subset of CPUs" (2408.13687 p15). A
simulator models such sharing as a fixed count of slots and a queue:
gem5's FUPool count, shared by every thread, oldest first
(src/cpu/o3/fu_pool.cc:165-190), lifted here from a pool to the device.
FPGA and ASIC instances each have logic of their own (DecoNet
2504.11805 p7-p10), so they need no device: a pool's units are its own.

A decode takes a slot when it starts and gives it back when its compute
ends; one with no free slot waits with its input landed, and the oldest
waiting decode on the device starts when a slot frees. A pool's
unit_count still caps the pool; a device with as many slots as its pools
have units never makes a decode wait. No slowdown with load is modelled:
one measured curve exists (NVIDIA 2604.12841 Table XI), for batches,
not concurrent streams.
"""

import collections
import dataclasses
from collections.abc import Callable

import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry


class ExecutionSlots:
    """The slots one device's decodes run in, across every pool on it."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """slot_count decodes may run at once on the device.

        A device part placed at a site holds this record; every pool at
        that site shares the slots.
        """

        slot_count: int
        name = "execution_slots"

        def __post_init__(self) -> None:
            if self.slot_count < 1:
                raise ValueError(
                    f"device.slot_count is {self.slot_count}; a device runs "
                    "at least one decode at a time"
                )

        def build(self, engine: engine_module.Engine) -> "ExecutionSlots":
            """A fresh device on these settings."""
            return ExecutionSlots(engine, self.slot_count)

    STATISTICS = (
        statistics_registry.Statistic(
            "slot_wait",
            statistic_kinds.Distribution,
            "ticks a decode whose input landed waited for a device slot",
            "ticks",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    def __init__(self, engine: engine_module.Engine, slot_count: int) -> None:
        self.engine = engine
        self.slot_count = slot_count
        # the decodes running in a slot now
        self.running: list = []
        # (job, start, since) of each decode waiting for a slot, oldest
        # first
        self.waiting: collections.deque = collections.deque()

    def take(
        self,
        job: decoding_records.DecodeJob,
        start: Callable[[decoding_records.DecodeJob], None],
    ) -> bool:
        """Give the decode a slot now; else queue it, start called later.

        True when it took a slot and may start at once.
        """
        if len(self.running) < self.slot_count:
            self.running.append(job)
            self.statistics.slot_wait.add(0)
            return True
        self.waiting.append((job, start, self.engine.now))
        return False

    def give_back(self, job: decoding_records.DecodeJob) -> None:
        """The decode's compute ended: its slot goes to the oldest waiting."""
        if not _holds(self.running, job):
            return
        _remove(self.running, job)
        self._start_the_oldest_waiting()

    def withdraw(self, job: decoding_records.DecodeJob) -> None:
        """A waiting decode left its unit before it started."""
        kept = collections.deque()
        for waiting in self.waiting:
            if waiting[0] is not job:
                kept.append(waiting)
        self.waiting = kept

    def check_settled(self) -> None:
        """At the end of a run no decode runs in a slot or waits for one."""
        if self.running or self.waiting:
            raise RuntimeError(
                f"run ended with {len(self.running)} decodes in device slots "
                f"and {len(self.waiting)} waiting for one"
            )

    def _start_the_oldest_waiting(self) -> None:
        """Start waiting decodes, oldest first, while a slot is free.

        A decode cancelled while it waited was withdrawn then
        (DecodeService.evict), so every one here starts.
        """
        while self.waiting and len(self.running) < self.slot_count:
            job, start, since = self.waiting.popleft()
            self.running.append(job)
            waited = self.engine.now - since
            self.statistics.slot_wait.add(waited)
            start(job)


def _holds(jobs: list, job: decoding_records.DecodeJob) -> bool:
    """Whether this very job is among the jobs (jobs compare by identity)."""
    for held in jobs:
        if held is job:
            return True
    return False


def _remove(jobs: list, job: decoding_records.DecodeJob) -> None:
    """Remove this very job from the list."""
    for index, held in enumerate(jobs):
        if held is job:
            del jobs[index]
            return
