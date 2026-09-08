"""The decoder unit's laws: staged time around one algorithm.

The stages are data priced in cycles of the unit's clock, the shape of
XQsim's modeled units (src/XQ-simulator, github.com/SNU-HPCS/XQsim). The
pipeline laws are Hennessy and Patterson, Computer Architecture,
Appendix C: a unit with an initiation interval frees its intake one
interval after a start and keeps ceil(latency / interval) decodes in
flight, while a unit without one holds its compute for the whole decode.
The stage cuts are checked against their closed form, the stages summing
to the whole job's cycles at the clock whatever the partition, and the
measured algorithm against the host clock of the call it wrapped on a
real Stim memory run.
"""

import pytest

import decsim.config as config
import decsim.decoders.decoder as decoder_module
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.machine as machine_module
import decsim.observe.stage_records as stage_records
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings

MEGAHERTZ = 250.0
CYCLE_MICROSECONDS = 1 / MEGAHERTZ
CYCLE_TICKS = config.microseconds_to_ticks(CYCLE_MICROSECONDS)
MEMORY_ROUNDS = 6
MEMORY_DISTANCE = 3
MEMORY_ERROR_PROBABILITY = 0.005
MEMORY_WINDOW_COUNT = 1
FETCH_STAGE = staged_decoder.DecoderStage("fetch", cycles_per_round=1)
RELEASE_STAGE = staged_decoder.DecoderStage("release", cycles_per_job=1)


class RecordingInner(decoder_module.DecoderBase):
    """A priced row keeping the tick at which its decode ran."""

    def __init__(self, engine, latency_microseconds=2.0):
        self.engine = engine
        self.latency_microseconds = latency_microseconds
        self.decode_ticks = []

    def latency(self, job):
        del job
        return config.microseconds_to_ticks(self.latency_microseconds)

    def decode(self, job):
        self.decode_ticks.append(self.engine.now)
        return decoding_records.DecodeResult(job.operation_id, job.window_id)


class ElapsedRecorder(decoder_module.DecoderBase):
    """A measured row around another, keeping every call's nanoseconds."""

    def __init__(self, inner):
        self.inner = inner
        self.fault_model_requirement = inner.fault_model_requirement
        self.elapsed_nanoseconds = []

    def latency(self, job):
        return self.inner.latency(job)

    def occupancy(self, job):
        return self.inner.occupancy(job)

    def decode(self, job):
        result, _elapsed_nanoseconds = self.decode_timed(job)
        return result

    def decode_timed(self, job):
        result, elapsed_nanoseconds = self.inner.decode_timed(job)
        self.elapsed_nanoseconds.append(elapsed_nanoseconds)
        return result, elapsed_nanoseconds


def fetch_and_release_timing():
    """One cycle of fetch per round, one cycle of release per job."""
    before = (FETCH_STAGE,)
    after = (RELEASE_STAGE,)
    return staged_decoder.UnitTiming(before, after, MEGAHERTZ)


def decode_job(window_id=0, round_count=3):
    """A weak-tier window job with nothing to decode."""
    request_key = window_records.DecoderRequestKey(
        1, window_id, window_records.DecoderTier.WEAK, window_id
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=window_id,
        round_count=round_count,
        label=f"W{window_id}",
        request_key=request_key,
    )


def ledger_hearing(decoder):
    """A stage ledger connected to the decoder as the Machine connects one."""
    ledger = stage_records.StageLedger()
    ledger_listener = ledger.stage_recorded
    decoder.stage_recorded.connect(ledger_listener)
    return ledger


def run_to_completion(decoder, engine, job):
    """Every (tick, result) the decoder's callback delivered."""
    delivered = []

    def on_result(result):
        delivered.append((engine.now, result))

    decoder.start(job, engine, on_result)
    engine.run()
    return delivered


def ticks_for_nanoseconds(elapsed_nanoseconds):
    """The ticks a measured unit is held for one host call."""
    elapsed_microseconds = elapsed_nanoseconds / 1000.0
    return config.microseconds_to_ticks(elapsed_microseconds)


def memory_circuit(stim):
    """A six-round distance-three rotated memory circuit."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=MEMORY_ROUNDS,
        distance=MEMORY_DISTANCE,
        after_clifford_depolarization=MEMORY_ERROR_PROBABILITY,
        before_measure_flip_probability=MEMORY_ERROR_PROBABILITY,
        after_reset_flip_probability=MEMORY_ERROR_PROBABILITY,
        before_round_data_depolarization=MEMORY_ERROR_PROBABILITY,
    )


def memory_machine(decoder, circuit):
    """The Stim-device memory run of one operation on this decoder row."""
    operation = program_records.Operation(
        id=1, name="memory", qubits=(0,), patches=(0,), circuit=circuit
    )
    rounds_policy = round_policies.FixedRounds(MEMORY_ROUNDS)
    workload = workload_settings.WorkloadSettings(
        operations=[operation], rounds_policy=rounds_policy
    )
    device = stim_device.StimDevice()
    qpu = qpu_settings.QpuSettings(distance=MEMORY_DISTANCE, device=device)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    settings = machine_settings.MachineSettings(
        workload=workload, qpu=qpu, weak_decoder=weak_decoder
    )
    return machine_module.Machine.build(settings, 3)


def test_the_result_is_produced_when_the_algorithm_stage_time_ends():
    """The inner decode runs once, at the algorithm record's end.

    The wrapped decoder is asked for the correction when its own priced
    time ends, not when the algorithm stage opened: the contract in
    decsim/decoders/staged_decoder.py, the algorithm "started on its own
    decoder and delivered when its time ends".
    """
    engine = engine_module.Engine()
    inner = RecordingInner(engine, latency_microseconds=5.0)
    timing = fetch_and_release_timing()
    decoder = staged_decoder.StagedDecoder(inner, timing)
    ledger = ledger_hearing(decoder)
    job = decode_job(round_count=3)
    run_to_completion(decoder, engine, job)
    records = ledger.records_for(1, 0)
    algorithm = records[1]
    algorithm_ticks = config.microseconds_to_ticks(5.0)
    fetch_ticks = 3 * CYCLE_TICKS
    algorithm_end = fetch_ticks + algorithm_ticks
    assert algorithm.stage == staged_decoder.ALGORITHM_STAGE
    assert algorithm.start_ticks == fetch_ticks
    assert algorithm.end_ticks == algorithm_end
    assert inner.decode_ticks == [algorithm_end]


def test_the_wrapper_adds_no_time_when_it_has_no_hardware_stages():
    """A unit with no stages is priced by its algorithm alone.

    The unit's latency is the stages plus the algorithm
    (staged_decoder.py, StagedDecoder.latency), so an empty stage list
    leaves the wrapped decoder's own service time untouched.
    """
    inner = decoders.PerRoundDecoder(tau_us=0.5)
    timing = staged_decoder.UnitTiming((), (), MEGAHERTZ)
    decoder = staged_decoder.StagedDecoder(inner, timing)
    job = decode_job(round_count=4)
    inner_ticks = inner.latency(job)
    assert inner_ticks == config.microseconds_to_ticks(2.0)
    assert decoder.latency(job) == inner_ticks


def test_the_hardware_stages_run_in_order_with_their_names_and_cycles():
    """Stages are declared data, not a fixed vocabulary.

    A hardware decoder names its own real stages and prices each in
    cycles per job plus cycles per round (staged_decoder.py's module
    docstring, after XQsim's per-unit cycle reports), and the unit walks
    the before-stages, the algorithm, then the after-stages.
    """
    engine = engine_module.Engine()
    ingest = staged_decoder.DecoderStage("syndrome_ingest", cycles_per_round=2)
    predecode = staged_decoder.DecoderStage("predecode", cycles_per_round=3)
    output = staged_decoder.DecoderStage("correction_output", cycles_per_job=4)
    before = (ingest, predecode)
    after = (output,)
    timing = staged_decoder.UnitTiming(before, after, MEGAHERTZ)
    inner = RecordingInner(engine, latency_microseconds=0.0)
    decoder = staged_decoder.StagedDecoder(inner, timing)
    ledger = ledger_hearing(decoder)
    job = decode_job(round_count=2)
    run_to_completion(decoder, engine, job)
    records = ledger.records_for(1, 0)
    named_cycles = [(record.stage, record.cycles) for record in records]
    assert named_cycles == [
        ("syndrome_ingest", 4),
        ("predecode", 6),
        ("algorithm", None),
        ("correction_output", 4),
    ]


def test_a_job_cancelled_before_it_starts_holds_the_unit_but_never_decodes():
    """A cancelled job is charged its whole unit time and yields None.

    The unit time is committed at dispatch, so the manager frees the
    unit on the same schedule whether or not the correction is wanted
    (decoder.py, DecoderBase.start: "their unit time is charged, no
    correction is computed").
    """
    engine = engine_module.Engine()
    inner = RecordingInner(engine, latency_microseconds=2.0)
    timing = fetch_and_release_timing()
    decoder = staged_decoder.StagedDecoder(inner, timing)
    job = decode_job(round_count=3)
    job.cancelled = True
    delivered = run_to_completion(decoder, engine, job)
    algorithm_ticks = config.microseconds_to_ticks(2.0)
    stage_ticks = 4 * CYCLE_TICKS
    whole_job_ticks = stage_ticks + algorithm_ticks
    assert decoder.latency(job) == whole_job_ticks
    assert delivered == [(whole_job_ticks, None)]
    assert inner.decode_ticks == []


def test_a_cancel_during_a_hardware_stage_stops_the_remaining_stages():
    """A cancel mid-walk ends the job silently, with the started stage kept.

    cancel is the manager's abort of work whose answer is no longer
    wanted (staged_decoder.py, StagedDecoder.cancel: "no further stages,
    no completion callback"), so the stage that had already opened stays
    in the trace and nothing after it is charged or reported.
    """
    engine = engine_module.Engine()
    inner = RecordingInner(engine, latency_microseconds=5.0)
    timing = fetch_and_release_timing()
    decoder = staged_decoder.StagedDecoder(inner, timing)
    ledger = ledger_hearing(decoder)
    job = decode_job(round_count=3)
    delivered = []

    def on_result(result):
        delivered.append((engine.now, result))

    decoder.start(job, engine, on_result)
    engine.schedule(CYCLE_TICKS, lambda: decoder.cancel(job))
    engine.run()
    records = ledger.records_for(1, 0)
    stage_names = [record.stage for record in records]
    assert delivered == []
    assert inner.decode_ticks == []
    assert stage_names == ["fetch"]


def test_a_hardware_stage_may_not_be_named_algorithm():
    """The algorithm stage is the wrapped decoder, never a unit stage.

    Its time is measured or priced by that decoder, so a hardware stage
    of the same name would be charged twice and would collide in the
    stage ledger (staged_decoder.py, UnitTiming.__post_init__).
    """
    stage = staged_decoder.DecoderStage(
        staged_decoder.ALGORITHM_STAGE, cycles_per_job=1
    )
    before = (stage,)
    with pytest.raises(ValueError, match="names the decoder itself"):
        staged_decoder.UnitTiming(before, (), MEGAHERTZ)


def test_a_negative_stage_cost_and_a_zero_frequency_are_refused():
    """A card that cannot be a unit is refused where the card is built.

    Negative cycles and a zero clock have no reading as time, and a card
    reaches decsim from a yaml section, so both raise at construction
    (staged_decoder.py, DecoderStage and UnitTiming).
    """
    with pytest.raises(ValueError, match="cycles must be nonnegative"):
        staged_decoder.DecoderStage("fetch", cycles_per_job=-1)
    with pytest.raises(ValueError, match="must be finite and positive"):
        staged_decoder.UnitTiming((), (), 0.0)


def test_the_unpipelined_unit_holds_its_compute_for_the_whole_decode():
    """Without an initiation interval the unit is busy for its latency.

    Hennessy and Patterson, Computer Architecture, Appendix C: an
    unpipelined functional unit accepts a new operation only after the
    previous one leaves, so its occupancy is its latency and one
    operation is in flight.
    """
    inner = decoders.PresetLatencyDecoder(4.0)
    timing = fetch_and_release_timing()
    decoder = staged_decoder.StagedDecoder(inner, timing)
    job = decode_job(round_count=3)
    algorithm_ticks = config.microseconds_to_ticks(4.0)
    stage_ticks = 4 * CYCLE_TICKS
    whole_job_ticks = stage_ticks + algorithm_ticks
    assert decoder.latency(job) == whole_job_ticks
    assert decoder.occupancy(job) == whole_job_ticks
    assert decoder.pipeline_depth(job) == 1


def test_a_pipelined_unit_frees_its_intake_after_the_initiation_interval():
    """The interval is the occupancy, and the depth fills the latency.

    Hennessy and Patterson, Computer Architecture, Appendix C: a
    pipelined unit accepts one operation per initiation interval and
    holds ceil(latency / interval) of them in flight; a declared depth
    is a shallower unit than the full pipeline and wins over it.
    """
    timing = staged_decoder.UnitTiming(
        (), (), MEGAHERTZ, initiation_interval_us=0.5
    )
    inner = decoders.PresetLatencyDecoder(4.0)
    decoder = staged_decoder.StagedDecoder(inner, timing)
    job = decode_job(round_count=3)
    assert decoder.occupancy(job) == config.microseconds_to_ticks(0.5)
    assert decoder.pipeline_depth(job) == 8
    declared_timing = staged_decoder.UnitTiming(
        (), (), MEGAHERTZ, initiation_interval_us=0.5, pipeline_depth=3
    )
    declared_unit = staged_decoder.StagedDecoder(inner, declared_timing)
    assert declared_unit.pipeline_depth(job) == 3


def test_the_stage_partition_does_not_change_the_whole_job_time():
    """Two cycles are two cycles, in one stage or in two.

    The stages are cut from the cumulative cycle count
    (staged_decoder.py, UnitTiming.stage_ticks), so a finer stage list
    describes the same unit rather than a slower one, and no partition
    accumulates a rounding error at the clock.
    """
    job = decode_job(round_count=3)
    whole = staged_decoder.DecoderStage("whole", cycles_per_job=2)
    first = staged_decoder.DecoderStage("first", cycles_per_job=1)
    second = staged_decoder.DecoderStage("second", cycles_per_job=1)
    one_stage = staged_decoder.UnitTiming((whole,), (), 300.0)
    two_stages = staged_decoder.UnitTiming((first, second), (), 300.0)
    one_stage_ticks = one_stage.stage_ticks(job)
    two_stage_ticks = two_stages.stage_ticks(job)
    one_stage_values = one_stage_ticks.values()
    two_stage_values = two_stage_ticks.values()
    two_cycles_microseconds = 2 / 300.0
    expected_ticks = config.microseconds_to_ticks(two_cycles_microseconds)
    assert sum(one_stage_values) == expected_ticks
    assert sum(two_stage_values) == expected_ticks


def test_every_measured_algorithm_holds_the_unit_for_its_own_wall_clock():
    """A measured row's stage is exactly the host call it wrapped.

    A decoder with no latency model is priced by running it: the unit
    stays busy for the nanoseconds the real matching call took on this
    host, rounded to ticks (decoder.py, DecoderBase._start_measured,
    "hold the unit for as long as it took"). Checked over a real
    six-round distance-three Stim memory run through PyMatching.
    """
    stim = pytest.importorskip("stim")
    pytest.importorskip("pymatching")
    matching_row = adapter.PyMatchingDecoder(latency_model=None)
    measured = ElapsedRecorder(matching_row)
    timing = fetch_and_release_timing()
    unit = staged_decoder.StagedDecoder(measured, timing)
    probe_job = decode_job()
    circuit = memory_circuit(stim)
    machine = memory_machine(unit, circuit)
    result = machine.run()
    records = machine.observation.stages.records
    algorithm = [
        record
        for record in records
        if record.stage == staged_decoder.ALGORITHM_STAGE
    ]
    held_ticks = [record.end_ticks - record.start_ticks for record in algorithm]
    measured_ticks = [
        ticks_for_nanoseconds(elapsed)
        for elapsed in measured.elapsed_nanoseconds
    ]
    assert result.terminal_status == "complete"
    assert len(algorithm) == MEMORY_WINDOW_COUNT
    assert held_ticks == measured_ticks
    assert unit.occupancy(probe_job) is None


def test_the_pipeline_parameters_are_refused_where_the_card_is_built():
    """A card that cannot describe a pipeline is refused at construction.

    A unit's card reaches decsim from a yaml section, so the four shapes
    that have no reading as a pipeline stop there: an interval that is
    not positive, one positive interval that rounds to zero ticks at the
    engine's resolution, a depth below one decode, and a depth on a unit
    that declared no interval and therefore holds its compute for the
    whole decode (staged_decoder.py, _check_pipeline).
    """
    with pytest.raises(ValueError, match="positive"):
        staged_decoder.UnitTiming((), (), MEGAHERTZ, initiation_interval_us=0.0)
    with pytest.raises(ValueError, match="rounds to zero"):
        staged_decoder.UnitTiming(
            (), (), MEGAHERTZ, initiation_interval_us=1e-9
        )
    with pytest.raises(ValueError, match="at least 1"):
        staged_decoder.UnitTiming(
            (),
            (),
            MEGAHERTZ,
            initiation_interval_us=1.0,
            pipeline_depth=0,
        )
    with pytest.raises(ValueError, match="needs an initiation_interval_us"):
        staged_decoder.UnitTiming((), (), MEGAHERTZ, pipeline_depth=2)
