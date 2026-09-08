"""The service's laws: the start, the park, the overlap, the pipeline.

Smith 1982 decoupled access-execute (rowD2, compare_overlap_laws.py):
with two slots the next window's transfer overlaps the compute, one
window per max(T, C). Tomasulo's rule at the boundary hazard: a landed
job whose window owes a boundary keeps its slot and never the compute.
Hennessy and Patterson App. C (rowD4, compare_pipeline_law.py): a
pipelined unit starts one decode per initiation interval, at most depth
in flight, each result a fixed latency after its start. The laws are
computed inside the tests. The strong-primary law needs the whole
declared fabric (tests/declared_run.py), because only a run builds the
request whose tier the pipelined model must accept.
"""

import pytest

import decsim.config as config
import decsim.decoders.decoders as decoders
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import tests.declared_run as declared_run
from decsim.decoders.decoder_manager import DecoderManager


class _Gate:
    """A gate the test opens by hand."""

    def __init__(self):
        self.is_open = True
        self.masked = []

    def may_stage(self, job):
        del job
        return True

    def may_start(self, job):
        del job
        return self.is_open

    def mask_input(self, job):
        self.masked.append(job.label)


def _job(index, gate=None, deps_remaining=0):
    payload = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id="p",
        round_index=index,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )
    request_key = window_records.DecoderRequestKey(
        1, index, window_records.DecoderTier.WEAK, index
    )
    window = window_records.Window(
        operation_id=1,
        window_index=index,
        commit_lo=1,
        commit_hi=1,
        buffer_hi=1,
        round_count=1,
        deps_remaining=deps_remaining,
    )
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=index,
        round_count=1,
        payloads=[payload],
        label=f"w{index}",
        request_key=request_key,
        window=window,
        gate=gate,
    )


def _send_after(engine, ticks):
    def send(on_landed):
        engine.schedule(ticks, on_landed)
        return ticks

    return send


def _manager(engine, decoder, dispatch_ticks=0):
    router = decoders.CodeRouter(decoder)
    scheduler = schedulers.FifoScheduler()
    policy = escalation_policies.Baseline()
    return DecoderManager(
        engine,
        router=router,
        scheduler=scheduler,
        num_units=1,
        escalation_policy=policy,
        dispatch_ticks=dispatch_ticks,
    )


def _input_landing_ticks(dispatch_ticks: int) -> list:
    """The tick each of two jobs' inputs was asked for, in order."""
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = _manager(engine, decoder, dispatch_ticks)
    asked = []
    for index in (0, 1):
        job = _job(index)
        send = _recording_send(engine, asked, 5)
        on_decoded = _resolving(manager)
        manager.enqueue(job, send, on_decoded)
    engine.run()
    return asked


def _recording_send(engine, asked: list, ticks: int):
    """A send that records the tick the manager asked for the input."""

    def send(on_landed):
        asked.append(engine.now)
        engine.schedule(ticks, on_landed)
        return ticks

    return send


def test_the_managers_dispatch_cost_delays_every_input_send_by_itself():
    """decoder_manager.dispatch_cycles, charged once per dispatch.

    Caune et al. arXiv:2410.05202 lines 519-526 and 636-641 measure 250
    to 370 control cycles per decode between a decode's arrival and its
    dispatch. The cost sits between the placement and the input send, so
    every job's input is asked for exactly one charge later than it is
    with no cost at all, whether the job was placed at once or waited.
    """
    free_ticks = _input_landing_ticks(0)
    charged = config.microseconds_to_ticks(2.0)
    charged_ticks = _input_landing_ticks(charged)
    assert len(free_ticks) == 2
    shifts = []
    for index, tick in enumerate(charged_ticks):
        shift = tick - free_ticks[index]
        shifts.append(shift)
    assert shifts == [charged, charged]


def _resolving(manager):
    """An on_decoded that closes the request, as the window side does."""
    keep = decoding_records.Verdict.KEEP

    def on_decoded(job, result):
        manager.resolve_weak_request(job, result, keep)

    return on_decoded


def _ending_in(ends, engine, manager):
    """An on_decoded that keeps the tick each job's result arrived."""
    keep = decoding_records.Verdict.KEEP

    def on_decoded(job, result):
        ends[job.label] = engine.now
        manager.resolve_weak_request(job, result, keep)

    return on_decoded


def _recording(manager, engine):
    """Compute start ticks by label, through service.begin."""
    starts = {}
    original_begin = manager.service.begin

    def recording_begin(job, gated=True):
        starts[job.label] = engine.now
        original_begin(job, gated)

    manager.service.begin = recording_begin
    return starts


def latency_for_window(job):
    """A row whose response depends on the window it decodes."""
    if job.window_id == 0:
        return 100.0
    return 50.0


def strong_primary_run(decoder):
    """One three-round patch decoded by this row alone, on declared ticks.

    StrongOnly makes the single row the primary tier, so its window
    reads Buffer 1 and rides the strong-buffer path.
    """
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 3)
    strong_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    policy = escalation_policies.StrongOnly()
    escalation = escalation_settings.EscalationSettings(policy=policy)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    frame = declared_run.declared_frame()
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        strong_decoder=strong_decoder,
        escalation=escalation,
        links=links,
        controller=controller,
        pauli_frame=frame,
    )
    return declared_run.run_machine(settings, 0)


def test_a_job_starts_when_its_input_landed_and_the_gate_allows():
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    done = {}
    gate = _Gate()
    transfer_ticks = config.microseconds_to_ticks(2.0)
    job = _job(0, gate)
    send_input = _send_after(engine, transfer_ticks)
    on_decoded = _ending_in(done, engine, manager)
    manager.enqueue(job, send_input, on_decoded)
    engine.run()
    assert starts["w0"] == transfer_ticks
    assert gate.masked == ["w0"]
    assert done["w0"] == transfer_ticks + config.microseconds_to_ticks(1.0)


def test_a_parked_job_keeps_its_slot_and_releases_the_compute():
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(1.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    gate = _Gate()
    gate.is_open = False
    job = _job(0, gate)
    on_decoded = _resolving(manager)
    manager.enqueue(job, None, on_decoded)
    (unit,) = manager.pool.units_by_pool["default"]
    assert job.is_parked is True
    assert unit.residents == [job]
    assert unit.holder is None
    assert manager.pool.free_count("default") == 1
    gate.is_open = True
    release_ticks = config.microseconds_to_ticks(3.0)
    window_key = (1, 0)
    engine.schedule(release_ticks, lambda: manager.release_parked(window_key))
    engine.run()
    assert starts["w0"] == release_ticks
    manager.check_decode_work_settled()


def test_the_second_slots_transfer_overlaps_the_compute():
    # T = 1 us, C = 2 us, three windows ready at once (rowD2's two-slot
    # law): w0 lands at 1 and computes 1..3; w1 landed at 1 and waits for
    # the compute until 3; w2 takes w0's slot at 3, lands at 4, starts 5
    engine = engine_module.Engine()
    decoder = decoders.PresetLatencyDecoder(2.0)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    transfer_ticks = config.microseconds_to_ticks(1.0)
    for index in range(3):
        job = _job(index)
        send_input = _send_after(engine, transfer_ticks)
        on_decoded = _resolving(manager)
        manager.enqueue(job, send_input, on_decoded)
    engine.run()
    assert starts == {
        "w0": config.microseconds_to_ticks(1.0),
        "w1": config.microseconds_to_ticks(3.0),
        "w2": config.microseconds_to_ticks(5.0),
    }


def test_a_pipelined_unit_issues_at_its_initiation_interval():
    # II = 0.5 us, C = 4 us, full depth: three windows landing at once
    # start 0.5 us apart and each returns 4 us after its start
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=0.5)
    algorithm = decoders.PresetLatencyDecoder(4.0)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    ends = {}
    on_decoded = _ending_in(ends, engine, manager)
    for index in range(3):
        job = _job(index)
        manager.enqueue(job, None, on_decoded)
    engine.run()
    interval = config.microseconds_to_ticks(0.5)
    latency = config.microseconds_to_ticks(4.0)
    assert starts == {"w0": 0, "w1": interval, "w2": 2 * interval}
    assert ends == {
        "w0": latency,
        "w1": interval + latency,
        "w2": 2 * interval + latency,
    }
    manager.check_decode_work_settled()


def test_an_initiation_interval_stays_a_lower_bound_below_the_response():
    """A response shorter than the interval never opens the intake early.

    The intake rate is the initiation interval's alone: a new decode may
    start every interval while each result still returns after the whole
    latency (staged_decoder.py's module docstring, after Hennessy and
    Patterson App. C). A one-microsecond response inside a
    ten-microsecond interval therefore frees the result state without
    erasing the intake cooldown, so the second start is one interval
    after the first and not one response.
    """
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=10.0)
    algorithm = decoders.PresetLatencyDecoder(1.0)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    ends = {}
    on_decoded = _ending_in(ends, engine, manager)
    first = _job(0)
    second = _job(1)
    manager.enqueue(first, None, on_decoded)
    manager.enqueue(second, None, on_decoded)
    engine.run()
    assert starts == {
        "w0": 0,
        "w1": config.microseconds_to_ticks(10.0),
    }
    assert ends == {
        "w0": config.microseconds_to_ticks(1.0),
        "w1": config.microseconds_to_ticks(11.0),
    }
    manager.check_decode_work_settled()


def test_a_declared_pipeline_depth_bounds_the_decodes_in_flight():
    """Depth two lets the third start wait for the first completion.

    A pipelined unit holds at most its declared depth in flight
    (Hennessy and Patterson App. C; decode_service.py,
    _initiation_complete keeps the compute claim on a full pipeline), and
    every in-flight decode stays resident because its input lives in the
    unit's memory until its result emerges (resident_capacity). So with
    a 5 us transfer, a 1 us interval, a 100 us response and four windows
    ready at once: w0 and w1 start 5 and 6, the intake is free at 7 but
    w2 waits until w0 completes at 105, and w3 cannot even take a slot
    before that completion frees the memory it needs.
    """
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming(
        (), (), 1.0, initiation_interval_us=1.0, pipeline_depth=2
    )
    algorithm = decoders.PresetLatencyDecoder(100.0)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    manager = _manager(engine, decoder)
    starts = _recording(manager, engine)
    ends = {}
    dispatches = {}

    def note_dispatch(job, _unit):
        dispatches[job.label] = engine.now

    manager.service.trace.job_dispatched.connect(note_dispatch)
    on_decoded = _ending_in(ends, engine, manager)
    transfer_ticks = config.microseconds_to_ticks(5.0)
    for index in range(4):
        job = _job(index)
        send_input = _send_after(engine, transfer_ticks)
        manager.enqueue(job, send_input, on_decoded)
    engine.run()
    assert starts == {
        "w0": config.microseconds_to_ticks(5.0),
        "w1": config.microseconds_to_ticks(6.0),
        "w2": config.microseconds_to_ticks(105.0),
        "w3": config.microseconds_to_ticks(110.0),
    }
    assert ends == {
        "w0": config.microseconds_to_ticks(105.0),
        "w1": config.microseconds_to_ticks(106.0),
        "w2": config.microseconds_to_ticks(205.0),
        "w3": config.microseconds_to_ticks(210.0),
    }
    assert dispatches["w2"] == 0  # the third window's slot was free at once
    assert starts["w2"] == ends["w0"]
    assert dispatches["w3"] == ends["w0"]
    manager.check_decode_work_settled()


def test_a_pipelined_unit_refuses_two_in_flight_response_times():
    """One pipelined unit takes one latency, so mixed responses refuse.

    A hardware pipeline retires in issue order (Hennessy and Patterson
    App. C), so a second decode declaring a different latency while the
    first is in flight would complete out of order; decoder_unit.py's
    add_flight refuses loudly instead of reordering the results.
    """
    engine = engine_module.Engine()
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=1.0)
    algorithm = decoders.FunctionLatencyDecoder(latency_for_window)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    manager = _manager(engine, decoder)
    first = _job(0)
    second = _job(1)
    on_decoded = _resolving(manager)
    manager.enqueue(first, None, on_decoded)
    manager.enqueue(second, None, on_decoded)
    with pytest.raises(RuntimeError, match="completes in order"):
        engine.run()


def test_a_pipelined_unit_serves_a_strong_primary_window():
    """A strong-primary window is a plain decode, so the pipeline takes it.

    Only an escalation carries strong_decode_for, and the pipelined
    model serves plain window decodes (decode_service.py, _pipeline_of):
    a strong-primary run decodes each window once, like the weak tier,
    so its window is priced by its own arithmetic on declared_run's
    fabric. Three rounds end at 3, readout classification and the wire
    publish Buffer 1 at 15, the 6 us strong-buffer transfer lands the
    input at 21, and the 100 us row returns at 121.
    """
    timing = staged_decoder.UnitTiming((), (), 1.0, initiation_interval_us=1.0)
    algorithm = decoders.PresetLatencyDecoder(100.0)
    decoder = staged_decoder.StagedDecoder(algorithm, timing)
    machine = strong_primary_run(decoder)
    window = machine.observation.windows.windows[(1, 0)]
    assert window.t_data_complete == config.microseconds_to_ticks(15.0)
    assert window.t_done == config.microseconds_to_ticks(121.0)
