"""The decoder unit's timing around one algorithm: stages and a pipeline.

One decode job is one walk through the configured stages before the
algorithm (for a weak ASIC, reading the window out of the decoder-side
memory), the algorithm itself, started on its own decoder and delivered
when its time ends, then the configured stages after it (releasing the
correction). Every stage is one engine event and one record, so the
trace shows latency at each point inside the unit.

Stages are data (name, cycles per job, cycles per round, at one clock),
not a fixed vocabulary: a hardware decoder declares its own real stages
when it has them. Precedent for modeling actual units and reporting
their cycles per job: XQsim src/XQ-simulator
(https://github.com/SNU-HPCS/XQsim, commit
006c38474c4caf8d51e2065ad3f3a5144b251bfc, MIT); nothing is copied from
it. A unit may pipeline: with an initiation interval a new job may start
every interval while each result still returns after the whole latency,
at most pipeline_depth in flight (Hennessy and Patterson, Computer
Architecture, App. C; Helios 2301.08419 is the hardware shape). Without
one, a job holds the unit from first stage to release.
"""

import dataclasses
import math
from typing import Callable, Optional

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.observe.trace_source as trace_source
import decsim.records.decoding as decoding_records
import decsim.records.seeds as seed_records

# The decoder unit component's name in the narrator (docs/
# architecture.md's component table).
LOG_SOURCE = "Decoder unit"
ALGORITHM_STAGE = "algorithm"


@dataclasses.dataclass(frozen=True)
class DecoderStage:
    """One named hardware stage priced in cycles."""

    name: str
    cycles_per_job: int = 0
    cycles_per_round: int = 0

    def __post_init__(self) -> None:
        if self.cycles_per_job < 0 or self.cycles_per_round < 0:
            raise ValueError(f"stage {self.name!r} cycles must be nonnegative")

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """The stage's cycles for one job: per job plus per round."""
        round_cycles = self.cycles_per_round * job.round_count
        return self.cycles_per_job + round_cycles


@dataclasses.dataclass(frozen=True)
class UnitTiming:
    """The unit's stages, the clock that prices them, and its pipeline.

    initiation_interval_us is the least time between two starts on the
    unit; None means the unit holds its compute for the whole decode.
    pipeline_depth bounds the decodes in flight; None means the full
    pipeline, ceil(latency / interval), computed per job.
    """

    before: tuple[DecoderStage, ...]
    after: tuple[DecoderStage, ...]
    frequency_mhz: float
    initiation_interval_us: Optional[float] = None
    pipeline_depth: Optional[int] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")
        for stage in self.before + self.after:
            if stage.name == ALGORITHM_STAGE:
                raise ValueError(
                    f"{ALGORITHM_STAGE!r} names the decoder itself, not a "
                    "hardware stage"
                )
        _check_pipeline(self.initiation_interval_us, self.pipeline_depth)

    def stage_ticks(self, job: decoding_records.DecodeJob) -> dict:
        """Ticks per stage, by name.

        Cut from the cumulative cycle count, so the stages sum to exactly
        the whole job's cycles at the clock, whatever the partition.
        """
        ticks = {}
        cumulative_cycles = 0
        previous_end = 0
        for stage in self.before + self.after:
            cumulative_cycles += stage.cycles_for(job)
            cumulative_microseconds = cumulative_cycles / self.frequency_mhz
            end = config.microseconds_to_ticks(cumulative_microseconds)
            ticks[stage.name] = end - previous_end
            previous_end = end
        return ticks

    def initiation_interval_ticks(self) -> Optional[int]:
        """The interval in ticks; None when the unit does not pipeline."""
        if self.initiation_interval_us is None:
            return None
        return config.microseconds_to_ticks(self.initiation_interval_us)


@dataclasses.dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job: name, cycles charged, start and end ticks."""

    operation_id: int
    window_id: int
    stage: str
    cycles: Optional[int]  # None for the algorithm, priced in time
    start_ticks: int
    end_ticks: int


class StagedDecoder(decoder_module.DecoderBase):
    """A decoder on a unit: its stages walked as engine events.

    The wrapped decoder starts the algorithm stage on its own terms
    (priced, or measured on the host clock); the result reaches on_result
    when the last stage ends (None when the job was cancelled meanwhile).
    Trace source: stage_recorded(record), one DecoderStageRecord per
    stage, fired when the stage's end is known (at entry for a priced
    stage, at the result for the algorithm), the lane a hardware model
    fills in for its own stages. The decoder keeps no history; the run's
    StageLedger (observe/stage_records.py) holds what it fires.
    """

    def __init__(self, decoder, timing: UnitTiming):
        self.decoder = decoder
        self.timing = timing
        self.fault_model_requirement = decoder.fault_model_requirement
        self.decoder_evidence = decoder.decoder_evidence
        self._running: dict = {}
        self.stage_recorded = trace_source.TraceSource()

    def run_seed_children(self) -> tuple:
        """The wrapped decoder under the segment decoder."""
        path = (seed_records.RunSeedPathSegment("field", "decoder"),)
        child = seed_records.RunSeedChild(path, self.decoder)
        return (child,)

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The wrapped decoder's result; the stages add no correction."""
        return self.decoder.decode(job)

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The stages' ticks plus the wrapped decoder's latency."""
        stage_ticks = self.timing.stage_ticks(job)
        stage_values = stage_ticks.values()
        stages = sum(stage_values)
        algorithm = self.decoder.latency(job)
        return stages + algorithm

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """The initiation interval when pipelined, else the whole latency.

        None when the wrapped decoder is measured on the host clock: the
        unit cannot say in advance when it frees.
        """
        interval_ticks = self.timing.initiation_interval_ticks()
        if interval_ticks is not None:
            return interval_ticks
        algorithm = self.decoder.occupancy(job)
        if algorithm is None:
            return None
        stage_ticks = self.timing.stage_ticks(job)
        stage_values = stage_ticks.values()
        stages = sum(stage_values)
        return stages + algorithm

    def pipeline_depth(self, job: decoding_records.DecodeJob) -> int:
        """The declared depth, or the full pipeline ceil(latency / interval)."""
        interval_ticks = self.timing.initiation_interval_ticks()
        if interval_ticks is None:
            return 1
        if self.timing.pipeline_depth is not None:
            return self.timing.pipeline_depth
        latency_ticks = self.latency(job)
        decodes_per_latency = latency_ticks / interval_ticks
        full_pipeline = math.ceil(decodes_per_latency)
        return max(1, full_pipeline)

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine,
        on_result: Callable[[Optional[decoding_records.DecodeResult]], None],
    ) -> None:
        """Walk the stages as engine events on the unit the manager granted."""
        running = _RunningDecode(job, on_result)
        steps = self._steps(job)
        key = _key(job)
        self._running[key] = running
        self._enter(running, engine, steps, 0)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Abort a running job: no further stages, no completion callback."""
        key = _key(job)
        running = self._running.pop(key, None)
        if running is not None:
            running.aborted = True
        self.decoder.cancel(job)

    def _steps(self, job: decoding_records.DecodeJob) -> list:
        """(name, cycles, ticks) per stage; the algorithm's time is its own."""
        ticks = self.timing.stage_ticks(job)
        steps = []
        for stage in self.timing.before:
            step = _hardware_step(stage, job, ticks)
            steps.append(step)
        steps.append((ALGORITHM_STAGE, None, None))
        for stage in self.timing.after:
            step = _hardware_step(stage, job, ticks)
            steps.append(step)
        return steps

    def _enter(self, running, engine, steps: list, index: int) -> None:
        job = running.job
        if running.aborted:
            return
        if index == len(steps):
            key = _key(job)
            self._running.pop(key, None)
            running.on_result(running.result)
            return
        name, cycles, ticks = steps[index]
        text = _stage_text(name, job, cycles)
        engine.log(LOG_SOURCE, text)
        if name == ALGORITHM_STAGE:
            self._enter_algorithm(running, engine, steps, index)
            return
        start = engine.now
        end = start + ticks
        record = DecoderStageRecord(
            job.operation_id, job.window_id, name, cycles, start, end
        )
        self.stage_recorded.fire(record)
        next_index = index + 1
        engine.schedule(
            ticks,
            lambda: self._enter(running, engine, steps, next_index),
            label=f"{name}({job.label})",
        )

    def _enter_algorithm(
        self, running, engine, steps: list, index: int
    ) -> None:
        """Start the wrapped decoder; its result closes the stage."""
        job = running.job
        if job.decoder_input is not None:
            # the read out of this unit's memory
            job.payloads = job.decoder_input.fragments()
        start = engine.now

        def on_algorithm_result(
            result: Optional[decoding_records.DecodeResult],
        ) -> None:
            running.result = result
            record = DecoderStageRecord(
                job.operation_id,
                job.window_id,
                ALGORITHM_STAGE,
                None,
                start,
                engine.now,
            )
            self.stage_recorded.fire(record)
            next_index = index + 1
            self._enter(running, engine, steps, next_index)

        self.decoder.start(job, engine, on_algorithm_result)


@dataclasses.dataclass
class _RunningDecode:
    job: decoding_records.DecodeJob
    on_result: Callable[[Optional[decoding_records.DecodeResult]], None]
    result: Optional[decoding_records.DecodeResult] = None
    aborted: bool = False


def _check_pipeline(
    initiation_interval_us: Optional[float], pipeline_depth: Optional[int]
) -> None:
    if initiation_interval_us is None:
        if pipeline_depth is not None:
            raise ValueError(
                "pipeline_depth needs an initiation_interval_us; without "
                "one the unit holds its compute for the whole decode"
            )
        return
    is_positive = initiation_interval_us > 0
    if not math.isfinite(initiation_interval_us) or not is_positive:
        raise ValueError("initiation_interval_us must be positive and finite")
    interval_ticks = config.microseconds_to_ticks(initiation_interval_us)
    if interval_ticks == 0:
        raise ValueError(
            "initiation_interval_us is positive but rounds to zero ticks"
        )
    if pipeline_depth is not None and pipeline_depth < 1:
        raise ValueError("pipeline_depth must be at least 1")


def _key(job: decoding_records.DecodeJob):
    """Window jobs carry a request key; a merged strong batch a service key."""
    if job.request_key is not None:
        return job.request_key
    return job.service_key


def _hardware_step(
    stage: DecoderStage, job: decoding_records.DecodeJob, ticks: dict
) -> tuple:
    cycles = stage.cycles_for(job)
    return stage.name, cycles, ticks[stage.name]


def _stage_text(name: str, job: decoding_records.DecodeJob, cycles) -> str:
    text = f"{name} {job.label}"
    if cycles is not None:
        text += f" ({cycles} cycles)"
    return text
