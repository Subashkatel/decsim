"""The staged decoder: stages before and after the algorithm, as events.

Checked against the closed form of the stage cuts (the stages sum to
the whole job's cycles at the clock) and against the bare decoder's
answers on a Stim memory circuit.
"""

import pytest

from decsim.config import microseconds_to_ticks
from decsim.decoders.decoder import DecoderBase
from decsim.decoders.decoders import PerRoundDecoder, PresetLatencyDecoder
from decsim.decoders.staged_decoder import (
    ALGORITHM_STAGE,
    DecoderStage,
    StagedDecoder,
    UnitTiming,
)
from decsim.engine import Engine
from decsim.observe.stage_records import StageLedger
from decsim.message import (
    DecodeJob,
    DecodeResult,
    DecoderRequestKey,
    DecoderTier,
)

MHZ = 250.0
CYCLE = microseconds_to_ticks(1 / MHZ)


FETCH = DecoderStage("fetch", cycles_per_round=1)
RELEASE = DecoderStage("release", cycles_per_job=1)


def _timing(before=(FETCH,), after=(RELEASE,), frequency_mhz=MHZ):
    return UnitTiming(tuple(before), tuple(after), frequency_mhz)


class _RecordingInner(DecoderBase):
    """Timing-only decoder that records the tick at which decode() ran."""

    def __init__(self, engine, latency_us=2.0):
        self.engine = engine
        self.latency_us = latency_us
        self.decode_ticks = []

    def latency(self, job):
        del job
        return microseconds_to_ticks(self.latency_us)

    def decode(self, job):
        self.decode_ticks.append(self.engine.now)
        return DecodeResult(job.op_id, job.window_id)


def _job(window_id=0, n_rounds=3):
    return DecodeJob(
        op_id=1,
        window_id=window_id,
        n_rounds=n_rounds,
        label=f"W{window_id}",
        request_key=DecoderRequestKey(
            1, window_id, DecoderTier.WEAK, window_id
        ),
    )


def _stages(decoder) -> StageLedger:
    """A ledger hearing this decoder's stages, as the Machine builds one."""
    ledger = StageLedger()
    decoder.stage_recorded.connect(ledger.stage_recorded)
    return ledger


def _run(decoder, engine, job):
    seen = {}

    def on_result(result):
        seen["done_at"] = engine.now
        seen["result"] = result

    decoder.start(job, engine, on_result)
    engine.run()
    return seen


def test_stages_before_algorithm_after_in_order_with_ticks():
    engine = Engine()
    inner = _RecordingInner(engine)
    decoder = StagedDecoder(inner, _timing())
    stages = _stages(decoder)

    seen = _run(decoder, engine, _job(n_rounds=3))

    records = stages.records_for(1, 0)
    algorithm_ticks = microseconds_to_ticks(2.0)
    assert [
        (r.stage, r.cycles, r.start_ticks, r.end_ticks) for r in records
    ] == [
        ("fetch", 3, 0, 3 * CYCLE),
        (ALGORITHM_STAGE, None, 3 * CYCLE, 3 * CYCLE + algorithm_ticks),
        (
            "release",
            1,
            3 * CYCLE + algorithm_ticks,
            4 * CYCLE + algorithm_ticks,
        ),
    ]
    assert seen["done_at"] == 4 * CYCLE + algorithm_ticks
    assert seen["result"].window_id == 0
    assert decoder.latency(_job(n_rounds=3)) == seen["done_at"]


def test_the_result_is_produced_when_the_algorithm_time_ends():
    engine = Engine()
    inner = _RecordingInner(engine, latency_us=5.0)
    decoder = StagedDecoder(inner, _timing())
    stages = _stages(decoder)

    _run(decoder, engine, _job())

    algorithm = stages.records_for(1, 0)[1]
    assert inner.decode_ticks == [algorithm.end_ticks]


def test_algorithm_time_is_the_wrapped_decoder_latency_only():
    inner = PerRoundDecoder(tau_us=0.5)
    decoder = StagedDecoder(inner, _timing(before=(), after=()))
    job = _job(n_rounds=4)
    assert decoder.latency(job) == inner.latency(job)


def test_hardware_stages_are_data_with_free_names():
    engine = Engine()
    timing = _timing(
        before=(
            DecoderStage("syndrome_ingest", cycles_per_round=2),
            DecoderStage("predecode", cycles_per_round=3),
        ),
        after=(DecoderStage("correction_output", cycles_per_job=4),),
    )
    decoder = StagedDecoder(_RecordingInner(engine, 0.0), timing)
    stages = _stages(decoder)

    _run(decoder, engine, _job(n_rounds=2))

    assert [(r.stage, r.cycles) for r in stages.records_for(1, 0)] == [
        ("syndrome_ingest", 4),
        ("predecode", 6),
        (ALGORITHM_STAGE, None),
        ("correction_output", 4),
    ]


def test_cancelled_job_still_holds_the_unit_but_skips_the_algorithm():
    engine = Engine()
    inner = _RecordingInner(engine)
    decoder = StagedDecoder(inner, _timing())
    job = _job()
    job.cancelled = True
    done = []
    decoder.start(job, engine, lambda result: done.append((engine.now, result)))
    engine.run()
    assert inner.decode_ticks == []
    assert done == [(decoder.latency(job), None)]


def test_result_reaches_the_callback():
    engine = Engine()
    decoder = StagedDecoder(_RecordingInner(engine), _timing())
    job = _job()
    seen = _run(decoder, engine, job)
    assert seen["result"].op_id == 1


def test_guards():
    with pytest.raises(ValueError):
        DecoderStage("x", cycles_per_job=-1)
    with pytest.raises(ValueError):
        UnitTiming((), (), 0.0)


def test_an_unpipelined_unit_holds_compute_for_the_whole_decode():
    decoder = StagedDecoder(PresetLatencyDecoder(4.0), _timing())
    job = _job(n_rounds=3)
    assert decoder.occupancy(job) == decoder.latency(job)
    assert decoder.pipeline_depth(job) == 1


def test_a_pipelined_unit_frees_its_intake_after_the_initiation_interval():
    timing = UnitTiming((), (), MHZ, initiation_interval_us=0.5)
    decoder = StagedDecoder(PresetLatencyDecoder(4.0), timing)
    job = _job(n_rounds=3)
    assert decoder.occupancy(job) == microseconds_to_ticks(0.5)
    assert decoder.pipeline_depth(job) == 8
    declared = UnitTiming(
        (), (), MHZ, initiation_interval_us=0.5, pipeline_depth=3
    )
    assert (
        StagedDecoder(PresetLatencyDecoder(4.0), declared).pipeline_depth(job)
        == 3
    )


def test_end_to_end_stim_memory_run_through_the_timed_decoder():
    # A real rotated memory circuit decoded by PyMatching inside the timed
    # unit: same logical answers as the bare decoder, three stages per window
    # in the trace, no two windows overlapping on the single unit.
    stim = pytest.importorskip("stim")
    from decsim.decoders.minimum_weight_perfect_matching.decoder import (
        PyMatchingDecoder,
    )
    from decsim.decoders.settings import DecoderSettings
    from decsim.frontends.settings import WorkloadSettings
    from decsim.machine import Machine, MachineSettings
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.qpu.settings import QpuSettings
    from decsim.qpu.stim_device import StimDevice

    def build(decoder):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=6,
            distance=3,
            after_clifford_depolarization=0.005,
            before_measure_flip_probability=0.005,
            after_reset_flip_probability=0.005,
            before_round_data_depolarization=0.005,
        )
        operation = Operation(
            id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
        )
        settings = MachineSettings(
            workload=WorkloadSettings(
                operations=[operation], rounds_policy=FixedRounds(6)
            ),
            qpu=QpuSettings(distance=3, device=StimDevice()),
            weak_decoder=DecoderSettings(decoder=decoder),
        )
        machine = Machine.build(settings, 11)
        result = machine.run()
        return machine, result

    _, bare_result = build(PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)))
    timed = StagedDecoder(
        PyMatchingDecoder(PerRoundDecoder(tau_us=0.1)), _timing()
    )
    staged, staged_result = build(timed)

    assert staged_result.terminal_status == "complete"
    assert [r.logical_observables for r in staged_result.operation_results] == [
        r.logical_observables for r in bare_result.operation_results
    ]
    stages = staged.observation.stages
    windows = stages.windows()
    assert windows
    for key in windows:
        names = [r.stage for r in stages.records_for(*key)]
        assert names == ["fetch", ALGORITHM_STAGE, "release"]
    spans = sorted(
        (
            stages.records_for(*key)[0].start_ticks,
            stages.records_for(*key)[-1].end_ticks,
        )
        for key in windows
    )
    assert all(
        a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])
    )
    assert any(ALGORITHM_STAGE in line for line in staged.observation.log.lines)


def test_measured_wall_clock_algorithm_holds_the_unit_for_the_real_call():
    # PyMatchingDecoder(latency_model=None) inside the unit: the real
    # matching call runs at algorithm start, the unit stays busy for exactly
    # the measured time, and the result is released only then.
    stim = pytest.importorskip("stim")
    from decsim.decoders.minimum_weight_perfect_matching.decoder import (
        PyMatchingDecoder,
    )
    from decsim.decoders.settings import DecoderSettings
    from decsim.frontends.settings import WorkloadSettings
    from decsim.machine import Machine, MachineSettings
    from decsim.message import Operation
    from decsim.qpu.round_policies import FixedRounds
    from decsim.qpu.settings import QpuSettings
    from decsim.qpu.stim_device import StimDevice

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=6,
        distance=3,
        after_clifford_depolarization=0.005,
        before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005,
        before_round_data_depolarization=0.005,
    )
    operation = Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )

    class RecordingPyMatching(PyMatchingDecoder):
        """The measured row, keeping every call's nanoseconds."""

        def __init__(self):
            PyMatchingDecoder.__init__(self, latency_model=None)
            self.elapsed_ns = []

        def decode_timed(self, job):
            result, elapsed_ns = PyMatchingDecoder.decode_timed(self, job)
            self.elapsed_ns.append(elapsed_ns)
            return result, elapsed_ns

    measured = RecordingPyMatching()
    timing = _timing()
    unit = StagedDecoder(measured, timing)
    settings = MachineSettings(
        workload=WorkloadSettings(
            operations=[operation], rounds_policy=FixedRounds(6)
        ),
        qpu=QpuSettings(distance=3, device=StimDevice()),
        weak_decoder=DecoderSettings(decoder=unit),
    )
    completed = Machine.build(settings, 3)
    result = completed.run()
    assert result.terminal_status == "complete"
    stage_records = completed.observation.stages.records
    algorithm = [r for r in stage_records if r.stage == ALGORITHM_STAGE]
    assert algorithm
    assert len(measured.elapsed_ns) == len(algorithm)
    for record, elapsed_ns in zip(algorithm, measured.elapsed_ns):
        assert elapsed_ns > 0
        held = record.end_ticks - record.start_ticks
        elapsed_microseconds = elapsed_ns / 1000.0
        assert held == microseconds_to_ticks(elapsed_microseconds)
    assert unit.occupancy(_job()) is None
    with pytest.raises(NotImplementedError, match="measured on the host"):
        PyMatchingDecoder(latency_model=None).latency(_job())


def test_cancel_stops_the_remaining_stages_and_never_calls_on_done():
    engine = Engine()
    inner = _RecordingInner(engine, latency_us=5.0)
    decoder = StagedDecoder(inner, _timing())
    stages = _stages(decoder)
    job = _job()
    done = []
    decoder.start(job, engine, lambda _result: done.append(engine.now))
    engine.schedule(1, lambda: decoder.cancel(job))  # during fetch
    engine.run()
    assert done == []
    assert inner.decode_ticks == []
    assert [r.stage for r in stages.records_for(1, 0)] == ["fetch"]


def test_a_hardware_stage_may_not_be_named_algorithm():
    with pytest.raises(ValueError, match="names the decoder itself"):
        UnitTiming((DecoderStage(ALGORITHM_STAGE, cycles_per_job=1),), (), MHZ)


def test_stage_ticks_sum_to_the_whole_job_at_the_clock_for_any_partition():
    job = _job(n_rounds=3)
    one = UnitTiming((DecoderStage("a", cycles_per_job=2),), (), 300.0)
    split = UnitTiming(
        (
            DecoderStage("a", cycles_per_job=1),
            DecoderStage("b", cycles_per_job=1),
        ),
        (),
        300.0,
    )
    assert (
        sum(one.stage_ticks(job).values())
        == sum(split.stage_ticks(job).values())
        == microseconds_to_ticks(2 / 300.0)
    )
