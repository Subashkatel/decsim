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

from ..config import us
from ..message import DecodeJob, DecodeResult, RunSeedChild, RunSeedPathSegment

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
        if any(s.name == ALGORITHM_STAGE for s in self.before + self.after):
            raise ValueError(f"{ALGORITHM_STAGE!r} names the decoder itself, not a hardware stage")

    def stage_ticks(self, job: DecodeJob) -> dict:
        """Ticks per stage, cut from the cumulative cycle count so the stages sum
        to exactly the whole job's cycles at the clock, whatever the partition."""
        ticks, cumulative, previous = {}, 0, 0
        for stage in self.before + self.after:
            cumulative += stage.cycles_for(job)
            end = us(cumulative / self.frequency_mhz)
            ticks[stage.name] = end - previous
            previous = end
        return ticks


@dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job: name, cycles charged, start and end ticks."""

    op_id: int
    window_id: int
    stage: str
    cycles: Optional[int]      # None for the algorithm, priced in time
    start_ticks: int
    end_ticks: int
    measured_ns: Optional[int] = None   # algorithm: wall clock of the real call when measured


@dataclass
class _RunningDecode:
    job: DecodeJob
    on_result: Callable[[Optional[DecodeResult]], None]
    result: Optional[DecodeResult] = None
    aborted: bool = False


class DecoderEngine:
    """Decoder port plus ``run()``: stages before, the algorithm, stages after;
    the result reaches the caller through the ``on_result`` callback when the
    last stage ends (None when the job was cancelled meanwhile)."""

    log_name = "DecoderEngine"

    def __init__(self, decoder, timing: DecoderTiming):
        from .decoders import _decoder_fault_model_requirement
        self.decoder = decoder
        self.timing = timing
        self.fault_model_requirement = _decoder_fault_model_requirement(decoder)
        self._running: dict = {}
        self.stage_records: list[DecoderStageRecord] = []

    def run_seed_children(self):
        return (RunSeedChild((RunSeedPathSegment("field", "decoder"),),
                             self.decoder),)

    @staticmethod
    def _key(job: DecodeJob):
        # Window jobs carry a request key; a merged strong batch only a service key.
        return job.request_key if job.request_key is not None else job.service_key

    @property
    def measures_wall_clock(self) -> bool:
        return bool(getattr(self.decoder, "measures_wall_clock", False))

    def latency(self, job: DecodeJob) -> int:
        return sum(self.timing.stage_ticks(job).values()) + self.decoder.latency(job)

    def run(self, job: DecodeJob, engine,
            on_result: Callable[[Optional[DecodeResult]], None]) -> None:
        """Walk the stages as engine events on the unit the manager granted."""
        running = _RunningDecode(job, on_result)
        ticks = self.timing.stage_ticks(job)
        steps = ([(s.name, s.cycles_for(job), ticks[s.name]) for s in self.timing.before]
                 + [(ALGORITHM_STAGE, None, None)]
                 + [(s.name, s.cycles_for(job), ticks[s.name]) for s in self.timing.after])
        self._running[self._key(job)] = running
        self._enter(running, engine, steps, 0)

    def cancel(self, job: DecodeJob) -> None:
        """Abort a running job: no further stages, no completion callback."""
        running = self._running.pop(self._key(job), None)
        if running is not None:
            running.aborted = True

    def _enter(self, running: _RunningDecode, engine, steps, index) -> None:
        job = running.job
        if running.aborted:
            return
        if index == len(steps):
            self._running.pop(self._key(job), None)
            running.on_result(running.result)
            return
        name, cycles, ticks = steps[index]
        start = engine.now
        measured_ns = None
        if name == ALGORITHM_STAGE:
            if job.decoder_input is not None:       # the read out of this unit's memory
                job.payloads = [fragment for round_input in job.decoder_input.rounds
                                for fragment in round_input.fragments]
            if self.measures_wall_clock:
                # Software decoder on this host: run the real call now, hold the
                # unit for exactly as long as it took, release the result then.
                if not job.cancelled:
                    running.result = self.decoder.decode(job)
                    measured_ns = self.decoder.last_decode_ns
                ticks = us((measured_ns or 0) / 1000.0)
            else:
                ticks = self.decoder.latency(job)
        self.stage_records.append(DecoderStageRecord(
            job.op_id, job.window_id, name, cycles, start, start + ticks, measured_ns))
        engine.log(self.log_name, f"{name} {job.label}"
                   + ("" if cycles is None else f" ({cycles} cycles)")
                   + ("" if measured_ns is None else f" ({measured_ns} ns measured)"))

        def leave():
            if (name == ALGORITHM_STAGE and not job.cancelled
                    and not self.measures_wall_clock):
                running.result = self.decoder.decode(job)   # ready when time ends
            self._enter(running, engine, steps, index + 1)

        engine.schedule(ticks, leave, label=f"{name}({job.label})")

    def stage_records_for(self, op_id: int, window_id: int) -> tuple:
        return tuple(record for record in self.stage_records
                     if (record.op_id, record.window_id) == (op_id, window_id))
