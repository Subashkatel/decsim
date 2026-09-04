"""Simulated timing around one functional QEC decoder.

One decode job is one walk through: the configured stages before the
algorithm (for a weak ASIC, reading the window out of the decoder-side
memory), the algorithm itself, priced by the wrapped decoder's own
latency model with its result available when that time ends, then the
configured stages after it (releasing the correction). Every stage is
one engine event and one record, so the trace shows latency at each
point inside the decoder.

Stages are data (name, cycles per job, cycles per round, at one clock),
not a fixed vocabulary: a hardware decoder declares its own real stages
when it has them. Precedent for modeling actual units and reporting
their cycles per job: XQsim src/XQ-simulator
(https://github.com/SNU-HPCS/XQsim, commit
006c38474c4caf8d51e2065ad3f3a5144b251bfc, MIT); nothing is copied from
it. No pipelining or overlap: one job holds one unit from first stage
to release; the unit count is the decoder manager's pool size.
"""

import dataclasses
import math
from typing import Callable, Optional

import decsim.config as config
import decsim.message as message

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

    def cycles_for(self, job: message.DecodeJob) -> int:
        """The stage's cycles for one job: per job plus per round."""
        round_cycles = self.cycles_per_round * job.n_rounds
        return self.cycles_per_job + round_cycles


@dataclasses.dataclass(frozen=True)
class DecoderTiming:
    """Stages before and after the algorithm, and the clock that prices them."""

    before: tuple[DecoderStage, ...]
    after: tuple[DecoderStage, ...]
    frequency_mhz: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_mhz) or self.frequency_mhz <= 0:
            raise ValueError("frequency_mhz must be finite and positive")
        for stage in self.before + self.after:
            if stage.name == ALGORITHM_STAGE:
                raise ValueError(
                    f"{ALGORITHM_STAGE!r} names the decoder itself, not a "
                    "hardware stage"
                )

    def stage_ticks(self, job: message.DecodeJob) -> dict:
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


@dataclasses.dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job: name, cycles charged, start and end ticks."""

    op_id: int
    window_id: int
    stage: str
    cycles: Optional[int]  # None for the algorithm, priced in time
    start_ticks: int
    end_ticks: int
    # the algorithm's wall clock, when the real call was measured
    measured_ns: Optional[int] = None


class DecoderEngine:
    """The Decoder port plus run(): stages before, the algorithm, stages after.

    The result reaches the caller through the on_result callback when the
    last stage ends (None when the job was cancelled meanwhile).
    """

    log_name = "DecoderEngine"

    def __init__(self, decoder, timing: DecoderTiming):
        self.decoder = decoder
        self.timing = timing
        self.fault_model_requirement = decoder.fault_model_requirement
        self._running: dict = {}
        self.stage_records: list[DecoderStageRecord] = []

    def run_seed_children(self) -> tuple:
        """The wrapped decoder under the segment decoder."""
        path = (message.RunSeedPathSegment("field", "decoder"),)
        child = message.RunSeedChild(path, self.decoder)
        return (child,)

    @property
    def measures_wall_clock(self) -> bool:
        """Whether the wrapped decoder is timed on the host clock."""
        measures = getattr(self.decoder, "measures_wall_clock", False)
        return bool(measures)

    def latency(self, job: message.DecodeJob) -> int:
        """The stages' ticks plus the wrapped decoder's latency."""
        stage_ticks = self.timing.stage_ticks(job)
        stage_values = stage_ticks.values()
        stages = sum(stage_values)
        algorithm = self.decoder.latency(job)
        return stages + algorithm

    def run(
        self,
        job: message.DecodeJob,
        engine,
        on_result: Callable[[Optional[message.DecodeResult]], None],
    ) -> None:
        """Walk the stages as engine events on the unit the manager granted."""
        running = _RunningDecode(job, on_result)
        steps = self._steps(job)
        key = _key(job)
        self._running[key] = running
        self._enter(running, engine, steps, 0)

    def cancel(self, job: message.DecodeJob) -> None:
        """Abort a running job: no further stages, no completion callback."""
        key = _key(job)
        running = self._running.pop(key, None)
        if running is not None:
            running.aborted = True

    def stage_records_for(self, op_id: int, window_id: int) -> tuple:
        """One window's stage records, in stage order."""
        records = []
        for record in self.stage_records:
            if record.op_id == op_id and record.window_id == window_id:
                records.append(record)
        return tuple(records)

    def _steps(self, job: message.DecodeJob) -> list:
        """(name, cycles, ticks) per stage; the algorithm's are priced later."""
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
        start = engine.now
        measured_ns = None
        if name == ALGORITHM_STAGE:
            ticks, measured_ns = self._enter_algorithm(running)
        end = start + ticks
        record = DecoderStageRecord(
            job.op_id, job.window_id, name, cycles, start, end, measured_ns
        )
        self.stage_records.append(record)
        text = _stage_text(name, job, cycles, measured_ns)
        engine.log(self.log_name, text)
        engine.schedule(
            ticks,
            lambda: self._leave(running, engine, steps, index),
            label=f"{name}({job.label})",
        )

    def _enter_algorithm(self, running) -> tuple:
        """(ticks, measured nanoseconds) of the algorithm stage at its start."""
        job = running.job
        if job.decoder_input is not None:
            # the read out of this unit's memory
            fragments = []
            for round_input in job.decoder_input.rounds:
                fragments.extend(round_input.fragments)
            job.payloads = fragments
        if not self.measures_wall_clock:
            ticks = self.decoder.latency(job)
            return ticks, None
        # Software decoder on this host: run the real call now, hold the
        # unit for exactly as long as it took, release the result then.
        measured_ns = None
        if not job.cancelled:
            running.result = self.decoder.decode(job)
            measured_ns = self.decoder.last_decode_ns
        elapsed_ns = 0
        if measured_ns is not None:
            elapsed_ns = measured_ns
        measured_microseconds = elapsed_ns / 1000.0
        ticks = config.microseconds_to_ticks(measured_microseconds)
        return ticks, measured_ns

    def _leave(self, running, engine, steps: list, index: int) -> None:
        name, _cycles, _ticks = steps[index]
        if name == ALGORITHM_STAGE:
            self._decode_at_time_end(running)
        next_index = index + 1
        self._enter(running, engine, steps, next_index)

    def _decode_at_time_end(self, running) -> None:
        """A priced algorithm's result is ready when its time ends."""
        job = running.job
        if job.cancelled:
            return
        if self.measures_wall_clock:
            return
        running.result = self.decoder.decode(job)


@dataclasses.dataclass
class _RunningDecode:
    job: message.DecodeJob
    on_result: Callable[[Optional[message.DecodeResult]], None]
    result: Optional[message.DecodeResult] = None
    aborted: bool = False


def _key(job: message.DecodeJob):
    """Window jobs carry a request key; a merged strong batch a service key."""
    if job.request_key is not None:
        return job.request_key
    return job.service_key


def _hardware_step(
    stage: DecoderStage, job: message.DecodeJob, ticks: dict
) -> tuple:
    cycles = stage.cycles_for(job)
    return stage.name, cycles, ticks[stage.name]


def _stage_text(name: str, job: message.DecodeJob, cycles, measured_ns) -> str:
    text = f"{name} {job.label}"
    if cycles is not None:
        text += f" ({cycles} cycles)"
    if measured_ns is not None:
        text += f" ({measured_ns} ns measured)"
    return text
