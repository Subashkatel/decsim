"""The decoder unit's timing around one algorithm: its stages.

One decode job walks the configured stages before the algorithm (for a
weak ASIC, reading the window out of the unit's memory), the algorithm
itself, started on its own decoder, then the stages after it (releasing
the correction). Every stage is one engine event and one record, so the
trace shows latency at each point inside the unit. Stages are data
(name, cycles per job, cycles per round, one clock), so a hardware
decoder declares its own, as XQsim models its units and reports their
cycles per job (github.com/SNU-HPCS/XQsim). A job holds the unit from
first stage to release.
"""

import dataclasses
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.records.rounds as round_records
import decsim.records.seeds as seed_records
import decsim.trace_source as trace_source

ALGORITHM_STAGE = "algorithm"


@dataclasses.dataclass(frozen=True)
class DecoderStage:
    """One named hardware stage priced in cycles."""

    name: str
    cycles_per_job: int = 0
    cycles_per_round: int = 0

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """The stage's cycles for one job: per job plus per round."""
        round_cycles = self.cycles_per_round * job.round_count
        return self.cycles_per_job + round_cycles

    def priced_on(self, unit_clock: config.Clock) -> config.Clock:
        """The clock the stage's cycles count on: the unit's own.

        A stage of other logic in front of the unit, a detection event
        former on its own clock (detection_events.py), names that clock
        instead, as gem5 gives each clocked object its own domain
        (src/sim/clocked_object.hh).
        """
        return unit_clock

    def formed_round_keys(self, job: decoding_records.DecodeJob) -> tuple:
        """The rounds this stage forms; none, for a stage that forms none."""
        del job
        return ()


@dataclasses.dataclass(frozen=True)
class MemoryFetchStage(DecoderStage):
    """The fetch out of the unit's own memory: its cycles, then one a word.

    Each round is read in whole words of word_bits, one a cycle, as
    gem5's crossbar charges divCeil(size, width) per packet
    (src/mem/xbar.cc:135) and Helios loads each round a byte a clock
    (control_node_single_FPGA.v lines 35-36 and 152-167). word_bits None
    prices no words.
    """

    word_bits: Optional[int] = None

    def cycles_for(self, job: decoding_records.DecodeJob) -> int:
        """Per job, per round, and a cycle for every word of every round."""
        overhead_cycles = DecoderStage.cycles_for(self, job)
        if self.word_bits is None:
            return overhead_cycles
        word_count = 0
        for round_bits in _round_widths(job):
            whole_words, part_word_bits = divmod(round_bits, self.word_bits)
            word_count += whole_words
            if part_word_bits:
                word_count += 1
        return overhead_cycles + word_count


@dataclasses.dataclass(frozen=True)
class UnitTiming:
    """The unit's stages and the clock that prices them."""

    before: tuple[DecoderStage, ...]
    after: tuple[DecoderStage, ...]
    clock: config.Clock

    def __post_init__(self) -> None:
        if self.clock.period_ticks < 1:
            raise ValueError("the unit's clock period must be at least a tick")

    def stage_ticks(self, job: decoding_records.DecodeJob) -> dict:
        """Ticks per stage, by name.

        A period is a whole number of ticks, so the stages sum to exactly
        the whole job's cycles at the clock, whatever the partition.
        """
        ticks = {}
        for stage in self.before + self.after:
            cycles = stage.cycles_for(job)
            clock = stage.priced_on(self.clock)
            ticks[stage.name] = cycles * clock.period_ticks
        return ticks


@dataclasses.dataclass(frozen=True)
class DecoderStageRecord:
    """One stage of one job: name, cycles charged, start and end ticks."""

    operation_id: int
    window_id: int
    stage: str
    cycles: Optional[int]  # None for the algorithm, priced in time
    start_ticks: int
    end_ticks: int
    # the unit the decode ran on, the lane it belongs to, as each LLVM
    # XRay record carries the thread it ran on
    # (tools/llvm-xray/xray-converter.cc:232-245)
    unit_name: str
    # the (operation, round) identities a formation stage turned into
    # detection events here; empty for every stage that forms none
    round_keys: tuple = ()
    # the run ordinals of the requests this decode serves, so a reader
    # can tell one window's decodes apart: its two forced-class solves,
    # its strong re-decode, and the members of a merged batch all carry
    # the same window key and different ordinals
    run_sequences: tuple = ()
    # the decode was cancelled while this stage was open: the stage ends
    # at the cancel, and no latency point reads it
    cancelled: bool = False
    # the tick a unit took this decode. The window record keeps the last
    # decode's, so a window decoded more than once needs each decode's
    # own here, beside the run ordinals that name them
    dispatch_ticks: Optional[int] = None
    # the tick this decode first may compute: its input landed and its
    # window owed no boundary. What it waited for after this tick is the
    # unit's compute, which is a wait of a different kind
    ready_ticks: Optional[int] = None
    # the rounds the decode read, the job's own count: a strong decode's
    # r_strong is what Toshio's backlog bound divides by (2510.25222
    # lines 1270-1300), and the window record keeps only the last decode
    round_count: int = 0
    # the ticks the decode had waited inside a strong backend when this
    # stage closed (DecodeJob.backend_queue_wait_ticks), all of it by the
    # algorithm stage's end
    backend_queue_wait_ticks: int = 0


class StagedDecoder(decoder_module.DecoderBase):
    """A decoder on a unit: its stages walked as engine events.

    The wrapped decoder starts the algorithm stage on its own terms
    (priced, or measured on the host clock); the result reaches
    on_result when the last stage ends, None when the job was cancelled
    meanwhile. Trace source: stage_recorded(record), one
    DecoderStageRecord per stage, fired when the stage ends. A record
    closed by a cancel says so: gem5 stops a squashed instruction where
    it stands and counts it apart (src/cpu/o3/inst_queue.cc:895-908,
    :294-298 and :1442), and decoder switching halts the strong
    decoder's ongoing computation at the weak decoder's confident
    verdict (Toshio et al. 2510.25222 lines 598-601 and 610-612).
    """

    def __init__(self, decoder, timing: UnitTiming):
        self.decoder = decoder
        self.timing = timing
        self._running: dict = {}
        self.stage_recorded = trace_source.TraceSource()

    @property
    def fault_model_requirement(self):
        """What the wrapped decoder needs of the model; the stages add none."""
        return self.decoder.fault_model_requirement

    @property
    def decoder_evidence(self):
        """The evidence the wrapped decoder reports; the stages add none."""
        return self.decoder.decoder_evidence

    @property
    def missing_evidence_reasons(self) -> dict:
        """Why the wrapped decoder cannot report a piece of evidence."""
        return self.decoder.missing_evidence_reasons

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
        """The whole latency, stages included; None when host-measured."""
        algorithm = self.decoder.occupancy(job)
        if algorithm is None:
            return None
        stage_ticks = self.timing.stage_ticks(job)
        stage_values = stage_ticks.values()
        stages = sum(stage_values)
        return stages + algorithm

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine,
        on_result: Callable[[Optional[decoding_records.DecodeResult]], None],
    ) -> None:
        """Walk the stages as engine events on the unit the manager granted."""
        running = _RunningDecode(job, on_result, engine)
        steps = self._steps(job)
        key = _key(job)
        self._running[key] = running
        self._enter(running, engine, steps, 0)

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Abort a running job: its open stage ends here, marked cancelled.

        The engine never unschedules, so the stage's own timer still
        fires and finds the walk aborted.
        """
        key = _key(job)
        running = self._running.pop(key, None)
        if running is not None:
            running.aborted = True
            self._close_stage(running, True)
        self.decoder.cancel(job)

    def _steps(self, job: decoding_records.DecodeJob) -> list:
        """One step per stage; the algorithm's time is its own."""
        steps = []
        unit_clock = self.timing.clock
        for stage in self.timing.before:
            step = _hardware_step(stage, job, unit_clock)
            steps.append(step)
        algorithm = _Step(ALGORITHM_STAGE, None)
        steps.append(algorithm)
        for stage in self.timing.after:
            step = _hardware_step(stage, job, unit_clock)
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
        step = steps[index]
        text = _stage_text(step, job)
        engine.log(log_sources.DECODER_UNIT, text)
        running.open_stage = step
        running.open_start_ticks = engine.now
        if step.name == ALGORITHM_STAGE:
            self._enter_algorithm(running, engine, steps, index)
            return
        next_index = index + 1
        delay = self._stage_delay(step, engine)
        engine.schedule(
            delay,
            lambda: self._leave(running, engine, steps, next_index),
            label=f"{step.name}({job.label})",
        )

    def _stage_delay(self, step: "_Step", engine) -> int:
        """The ticks to the clock edge the stage's cycles end on."""
        if step.cycles == 0:
            return 0
        now = engine.now
        edge = step.clock.edge(step.cycles, now)
        return edge - now

    def _leave(self, running, engine, steps: list, index: int) -> None:
        """A hardware stage's time ended: close its record, then go on."""
        if running.aborted:
            return
        self._close_stage(running, False)
        self._enter(running, engine, steps, index)

    def _close_stage(self, running, cancelled: bool) -> None:
        """Write the open stage's record, ending it at this instant."""
        step = running.open_stage
        if step is None:
            return
        running.open_stage = None
        job = running.job
        sequences = _run_sequences(job)
        record = DecoderStageRecord(
            job.operation_id,
            job.window_id,
            step.name,
            step.cycles,
            running.open_start_ticks,
            running.engine.now,
            job.decoding_unit_name,
            step.round_keys,
            sequences,
            cancelled,
            job.dispatch_ticks,
            job.ready_ticks,
            round_count=job.round_count,
            backend_queue_wait_ticks=job.backend_queue_wait_ticks,
        )
        self.stage_recorded.fire(record)

    def _enter_algorithm(
        self, running, engine, steps: list, index: int
    ) -> None:
        """Start the wrapped decoder; its result closes the stage."""
        job = running.job
        if job.decoder_input is not None:
            # the read out of this unit's memory
            job.payloads = job.decoder_input.fragments()

        def on_algorithm_result(
            result: Optional[decoding_records.DecodeResult],
        ) -> None:
            if running.aborted:
                return
            running.result = result
            self._close_stage(running, False)
            next_index = index + 1
            self._enter(running, engine, steps, next_index)

        self.decoder.start(job, engine, on_algorithm_result)


@dataclasses.dataclass(frozen=True)
class _Step:
    """One stage of one job as the walk meets it."""

    name: str
    # None for the algorithm, whose time is the wrapped decoder's own
    cycles: Optional[int]
    round_keys: tuple = ()
    # the clock the cycles count on; None for the algorithm
    clock: Optional[config.Clock] = None


@dataclasses.dataclass
class _RunningDecode:
    """One walk in progress: what it decodes, and the stage now open."""

    job: decoding_records.DecodeJob
    on_result: Callable[[Optional[decoding_records.DecodeResult]], None]
    engine: object  # the clock the open stage's record is closed against
    result: Optional[decoding_records.DecodeResult] = None
    aborted: bool = False
    open_stage: Optional["_Step"] = None
    open_start_ticks: int = 0


def _run_sequences(job: decoding_records.DecodeJob) -> tuple:
    """The run ordinals of every request this decode serves."""
    keys = job.service_original_request_keys
    if not keys and job.request_key is not None:
        keys = (job.request_key,)
    sequences = []
    for key in keys:
        sequences.append(key.run_sequence)
    return tuple(sequences)


def _key(job: decoding_records.DecodeJob):
    """Window jobs carry a request key; a merged strong batch a service key."""
    if job.request_key is not None:
        return job.request_key
    return job.service_key


def _hardware_step(
    stage: DecoderStage,
    job: decoding_records.DecodeJob,
    unit_clock: config.Clock,
) -> "_Step":
    cycles = stage.cycles_for(job)
    round_keys = stage.formed_round_keys(job)
    clock = stage.priced_on(unit_clock)
    return _Step(stage.name, cycles, round_keys, clock)


def _stage_text(step: "_Step", job: decoding_records.DecodeJob) -> str:
    text = f"{step.name} {job.label}"
    if step.cycles is not None:
        text += f" ({step.cycles} cycles)"
    if step.round_keys:
        text += f" forming {len(step.round_keys)} rounds"
    return text


def _round_widths(job: decoding_records.DecodeJob) -> list:
    """The bits of each round the job reads, in round order.

    The landed input once it is in the unit's memory, the payloads
    before; a fragment that states no size holds no bits.
    """
    bits_by_round: dict = {}
    for fragment in _fragments_of(job):
        round_key = (fragment.operation_id, fragment.round_index)
        fragment_bits = round_records.stated_bits(fragment.size_bits)
        bits_by_round.setdefault(round_key, 0)
        bits_by_round[round_key] += fragment_bits
    widths = bits_by_round.values()
    return list(widths)


def _fragments_of(job: decoding_records.DecodeJob) -> list:
    decoder_input = job.decoder_input
    if decoder_input is None:
        return list(job.payloads)
    return decoder_input.fragments()
