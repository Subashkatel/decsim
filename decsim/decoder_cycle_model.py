"""Staged weak decoder: a memory plus a compute engine stepping five stages.

Shape adapted from XQsim (https://github.com/SNU-HPCS/XQsim, commit
006c38474c4caf8d51e2065ad3f3a5144b251bfc, MIT, full text at
tmp/references/code/xqsim/LICENSE): src/XQ-simulator/error_decode_unit.py
(a decode unit as a state machine whose per-state cycles are counted and
appended per job to unit_stat.edu_cycle_result), quantum_instruction_fetch.py
(fetch reads instruction memory into a buffer before decode), and
xq_simulator.py run_cycle_transfer/update/tick (one unit steps one job at a
time). No XQsim code is copied: XQsim steps every unit every cycle, decsim
schedules one engine event per stage boundary with the stage's cycles at the
card clock, so the trace carries the same per-stage information without a
per-cycle loop.

Stage names and order are the classical five-stage pipeline (Hennessy and
Patterson, IF/ID/EX/MEM/WB, as declared in Ripes rv5s.h:45). Decoder-side
meaning: FETCH reads the window's rounds out of the decoder-side input store
(the LILLIPUT syndrome FIFO, 2108.06569 sec. 4), DECODE turns bits into
decoder work, EXECUTE runs the decoding algorithm, MEMORY writes the
correction, WRITEBACK releases it. No pipelining, overlap, or stalls: one job
occupies one unit from FETCH start to WRITEBACK end (baseline, no
optimizations). Unit count is the decoder manager's pool size.

Timing rule: stage end ticks come from cumulative cycles divided by the clock,
so the job's total equals the sum of its stage cycles at the clock exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Optional

from .config import us
from .message import DecodeJob, DecodeResult, RunSeedChild, RunSeedPathSegment


class DecoderPipelineStage(str, Enum):
    """The five decoder stages, in execution order."""

    FETCH = "fetch"
    DECODE = "decode"
    EXECUTE = "execute"
    MEMORY = "memory"
    WRITEBACK = "writeback"


class CycleQuantityBasis(str, Enum):
    """Whether configured cycles are charged once per job or once per round."""

    PER_JOB = "per_job"
    PER_ROUND = "per_round"


@dataclass(frozen=True)
class StageCycleConfig:
    """One stage's cycle budget, quantity basis, and the source of the number."""

    cycles: int
    basis: CycleQuantityBasis
    source: str

    def __post_init__(self) -> None:
        if type(self.cycles) is not int or self.cycles < 0:
            raise ValueError("cycles must be a nonnegative int")

    def cycles_for(self, job: DecodeJob) -> int:
        if self.basis is CycleQuantityBasis.PER_ROUND:
            return self.cycles * job.n_rounds
        return self.cycles

    def to_json_value(self) -> dict:
        return {"cycles": self.cycles, "basis": self.basis.value,
                "source": self.source}


@dataclass(frozen=True)
class DecoderCycleModel:
    """The card: five stage budgets and the compute engine clock.

    ``include_inner_latency`` adds the wrapped decoder's own latency model to
    EXECUTE, so an algorithm priced in time (PerRoundDecoder and friends) keeps
    its time and the card adds the surrounding stages.
    """

    fetch: StageCycleConfig
    decode: StageCycleConfig
    execute: StageCycleConfig
    memory: StageCycleConfig
    writeback: StageCycleConfig
    frequency_mhz: float
    frequency_source: str
    include_inner_latency: bool
    model_name: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0.0:
            raise ValueError("frequency_mhz must be finite and positive")
        if us(1.0 / self.frequency_mhz) == 0:
            raise ValueError("frequency_mhz too high: one cycle is under one tick")

    def stages(self) -> tuple:
        return (
            (DecoderPipelineStage.FETCH, self.fetch),
            (DecoderPipelineStage.DECODE, self.decode),
            (DecoderPipelineStage.EXECUTE, self.execute),
            (DecoderPipelineStage.MEMORY, self.memory),
            (DecoderPipelineStage.WRITEBACK, self.writeback),
        )

    def ticks_for_cycles(self, cycles: int) -> int:
        return us(cycles / self.frequency_mhz)

    def to_json_value(self) -> dict:
        return {
            "model_name": self.model_name,
            "stages": [dict(stage=stage.value, **config.to_json_value())
                       for stage, config in self.stages()],
            "frequency_mhz": self.frequency_mhz,
            "frequency_source": self.frequency_source,
            "include_inner_latency": self.include_inner_latency,
        }


@dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job on one unit: what XQsim appends per job per state."""

    op_id: int
    window_id: int
    stage: DecoderPipelineStage
    start_ticks: int
    end_ticks: int
    cycles: int
    rounds_read: int      # FETCH: rounds read out of the input store; else 0


@dataclass
class _JobInFlight:
    """The unit's registers while one job walks the stages."""

    job: DecodeJob
    on_done: Callable[[], None]
    start_ticks: int
    inner_latency_ticks: int
    stage_index: int = 0
    result: Optional[DecodeResult] = None


class StagedDecoder:
    """Memory plus compute engine around a real decoder; Decoder port plus run().

    ``run(job, engine, on_done)`` walks FETCH, DECODE, EXECUTE, MEMORY,
    WRITEBACK as engine events and calls ``on_done`` when WRITEBACK ends;
    ``decode(job)`` then returns the result the compute engine produced at
    EXECUTE. ``latency(job)`` reports the same total for callers that price a
    job without running it (escalation planning, switching composition).
    """

    log_name = "StagedDecoder"

    def __init__(self, inner, cycle_model: DecoderCycleModel):
        from .decoders import _decoder_fault_model_requirement
        self.inner = inner
        self.cycle_model = cycle_model
        self.fault_model_requirement = _decoder_fault_model_requirement(inner)
        self._in_flight: dict[int, _JobInFlight] = {}
        self._released: dict[int, DecodeResult] = {}
        self.stage_records: list[DecoderStageRecord] = []

    def run_seed_children(self):
        return (RunSeedChild((RunSeedPathSegment("field", "inner"),), self.inner),)

    # ------------------------------------------------------------ pricing
    def _inner_latency_ticks(self, job: DecodeJob) -> int:
        if self.cycle_model.include_inner_latency:
            return self.inner.latency(job)
        return 0

    def _stage_end_offsets(self, job: DecodeJob, inner_latency_ticks: int) -> tuple:
        """Offset from job start to each stage's end, in stage order."""
        offsets = []
        cumulative_cycles = 0
        extra_ticks = 0
        for stage, config in self.cycle_model.stages():
            cumulative_cycles += config.cycles_for(job)
            if stage is DecoderPipelineStage.EXECUTE:
                extra_ticks = inner_latency_ticks
            offsets.append(
                self.cycle_model.ticks_for_cycles(cumulative_cycles) + extra_ticks)
        return tuple(offsets)

    def latency(self, job: DecodeJob) -> int:
        return self._stage_end_offsets(job, self._inner_latency_ticks(job))[-1]

    # ------------------------------------------------------------ running
    def run(self, job: DecodeJob, engine, on_done: Callable[[], None]) -> None:
        """Start one job on a free unit; the manager guarantees the unit."""
        if id(job) in self._in_flight:
            raise RuntimeError(f"job {job.label!r} is already running")
        flight = _JobInFlight(job, on_done, engine.now,
                              self._inner_latency_ticks(job))
        self._in_flight[id(job)] = flight
        self._enter_stage(flight, engine)

    def _enter_stage(self, flight: _JobInFlight, engine) -> None:
        job = flight.job
        stage, config = self.cycle_model.stages()[flight.stage_index]
        offsets = self._stage_end_offsets(job, flight.inner_latency_ticks)
        start = engine.now
        end = flight.start_ticks + offsets[flight.stage_index]
        rounds_read = 0
        if stage is DecoderPipelineStage.FETCH:
            rounds_read = (0 if job.decoder_input is None
                           else len(job.decoder_input.rounds))
        elif stage is DecoderPipelineStage.EXECUTE:
            flight.result = (DecodeResult(job.op_id, job.window_id)
                             if job.cancelled else self.inner.decode(job))
        cycles = config.cycles_for(job)
        self.stage_records.append(DecoderStageRecord(
            job.op_id, job.window_id, stage, start, end, cycles, rounds_read))
        rounds_note = f", {rounds_read} rounds" if rounds_read else ""
        engine.log(self.log_name,
                   f"{stage.value.upper()} {job.label} ({cycles} cycles{rounds_note})")
        engine.schedule(end - start, lambda: self._leave_stage(flight, engine),
                        label=f"{stage.value}({job.label})")

    def _leave_stage(self, flight: _JobInFlight, engine) -> None:
        flight.stage_index += 1
        if flight.stage_index < len(DecoderPipelineStage):
            self._enter_stage(flight, engine)
            return
        job = flight.job
        del self._in_flight[id(job)]
        # The manager reads a result only for a live window job; cancelled and
        # external jobs finish without decode(), so nothing is held for them.
        if not job.cancelled and job.on_done is None:
            self._released[id(job)] = flight.result
        flight.on_done()

    def decode(self, job: DecodeJob) -> DecodeResult:
        """Return the result the compute engine produced at EXECUTE."""
        result = self._released.pop(id(job), None)
        if result is None:
            raise RuntimeError(f"job {job.label!r} has not been released")
        return result

    def stage_records_for(self, op_id: int, window_id: int) -> tuple:
        return tuple(record for record in self.stage_records
                     if (record.op_id, record.window_id) == (op_id, window_id))
