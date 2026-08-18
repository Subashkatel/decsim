"""Simulated timing around one functional QEC decoder.

One decode job is one walk through: the configured stages before the
algorithm (for a weak ASIC, reading the window out of the decoder-side memory),
the algorithm itself, priced by the wrapped decoder's own latency model with
its result available when that time ends, then the configured stages after it
(releasing the correction). Every stage is one engine event and one record, so
the trace shows latency at each point inside the decoder.

Stages are data (name, cycles per job, cycles per round, at one clock), not a
fixed vocabulary: a hardware decoder declares its own real stages when it has
them. Precedent for modeling actual units and reporting their cycles per job:
XQsim src/XQ-simulator (https://github.com/SNU-HPCS/XQsim, commit
006c38474c4caf8d51e2065ad3f3a5144b251bfc, MIT); nothing is copied from it.
No pipelining or overlap: one job holds one unit from first stage to release;
the unit count is the decoder manager's pool size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

from .config import us
from .message import DecodeJob, DecodeResult, RunSeedChild, RunSeedPathSegment

ALGORITHM_STAGE = "algorithm"


@dataclass(frozen=True)
class DecoderStage:
    """One named hardware stage priced in cycles."""

    name: str
    cycles_per_job: int = 0
    cycles_per_round: int = 0

    def __post_init__(self) -> None:
        if self.cycles_per_job < 0 or self.cycles_per_round < 0:
            raise ValueError(f"stage {self.name!r} cycles must be nonnegative")

    def cycles_for(self, job: DecodeJob) -> int:
        return self.cycles_per_job + self.cycles_per_round * job.n_rounds


@dataclass(frozen=True)
class DecoderTiming:
    """Stages before and after the algorithm, and the clock that prices them."""

    before: tuple[DecoderStage, ...]
    after: tuple[DecoderStage, ...]
    frequency_mhz: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")

    def ticks_for(self, stage: DecoderStage, job: DecodeJob) -> int:
        return us(stage.cycles_for(job) / self.frequency_mhz)


@dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job: name, cycles charged, start and end ticks."""

    op_id: int
    window_id: int
    stage: str
    cycles: Optional[int]      # None for the algorithm, priced in time
    start_ticks: int
    end_ticks: int


@dataclass
class _RunningDecode:
    job: DecodeJob
    on_done: Callable[[], None]
    result: Optional[DecodeResult] = None


class DecoderEngine:
    """Decoder port plus ``run()``: stages before, the algorithm, stages after."""

    log_name = "DecoderEngine"

    def __init__(self, decoder, timing: DecoderTiming):
        from .decoders import _decoder_fault_model_requirement
        self.decoder = decoder
        self.timing = timing
        self.fault_model_requirement = _decoder_fault_model_requirement(decoder)
        self._completed: dict = {}
        self.stage_records: list[DecoderStageRecord] = []

    def run_seed_children(self):
        return (RunSeedChild((RunSeedPathSegment("field", "decoder"),),
                             self.decoder),)

    @staticmethod
    def _key(job: DecodeJob):
        # Window jobs carry a request key; a merged strong batch only a service key.
        return job.request_key if job.request_key is not None else job.service_key

    def latency(self, job: DecodeJob) -> int:
        stages = self.timing.before + self.timing.after
        return (sum(self.timing.ticks_for(stage, job) for stage in stages)
                + self.decoder.latency(job))

    def run(self, job: DecodeJob, engine, on_done: Callable[[], None]) -> None:
        """Walk the stages as engine events on the unit the manager granted."""
        running = _RunningDecode(job, on_done)
        steps = ([(s.name, s.cycles_for(job), self.timing.ticks_for(s, job))
                  for s in self.timing.before]
                 + [(ALGORITHM_STAGE, None, self.decoder.latency(job))]
                 + [(s.name, s.cycles_for(job), self.timing.ticks_for(s, job))
                    for s in self.timing.after])
        self._enter(running, engine, steps, 0)

    def _enter(self, running: _RunningDecode, engine, steps, index) -> None:
        job = running.job
        if index == len(steps):
            key = self._key(job)          # None for a self-contained external job
            if key is not None and running.result is not None:
                self._completed[key] = running.result
            running.on_done()
            return
        name, cycles, ticks = steps[index]
        start = engine.now
        self.stage_records.append(DecoderStageRecord(
            job.op_id, job.window_id, name, cycles, start, start + ticks))
        engine.log(self.log_name, f"{name} {job.label}"
                   + ("" if cycles is None else f" ({cycles} cycles)"))

        def leave():
            if name == ALGORITHM_STAGE and not job.cancelled:
                running.result = self.decoder.decode(job)   # ready when time ends
            self._enter(running, engine, steps, index + 1)

        engine.schedule(ticks, leave, label=f"{name}({job.label})")

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return the result the algorithm produced; released by run()."""
        result = self._completed.pop(self._key(job), None)
        if result is None:
            raise RuntimeError(f"{job.label!r} has not completed")
        return result

    def stage_records_for(self, op_id: int, window_id: int) -> tuple:
        return tuple(record for record in self.stage_records
                     if (record.op_id, record.window_id) == (op_id, window_id))
