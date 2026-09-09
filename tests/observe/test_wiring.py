"""The wiring: every listener connected once, and no source heard twice.

decsim/observe/wiring.py is the one place that connects a listener to
the sources it hears, so the census below is the whole picture: walk the
built machine, find every trace source, and read who hears it. A source
with two listeners of one class is a connection made twice, which would
double a count without failing anything.
"""

import dataclasses

import decsim.config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.machine as machine_module
import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.flight_recorder as flight_recorder_module
import decsim.observe.log_writers as log_writers
import decsim.observe.observation as observation_module
import decsim.observe.queue_depth as queue_depth_module
import decsim.observe.referee_audit as referee_audit_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_events as round_events_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.sampled_shots as sampled_shots_module
import decsim.observe.settings as observe_settings
import decsim.observe.stage_records as stage_records_module
import decsim.observe.window_ledger as window_ledger_module
import decsim.observe.wiring as wiring
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import decsim.trace_source as trace_source
import tests.declared_run as declared_run
import tests.observe.gate_point as gate_point

EVERY_KNOB = {
    "record_switching_windows": True,
    "round_store_occupancy": True,
    "backlog_trace": True,
    "decoder_utilization": True,
    "decoder_memory_occupancy": True,
    "data_movement": True,
    "trace": "chrome",
}


def _listener_class(listener):
    """The class that hears, through a partial or a bound method."""
    bound = getattr(listener, "func", listener)
    owner = getattr(bound, "__self__", None)
    if owner is None:
        return type(bound)
    return type(owner)


def _walk(root) -> list:
    """(owner, attribute, source) for every trace source the machine holds."""
    found = []
    seen = set()
    pending = [root]
    while pending:
        owner = pending.pop()
        identity = id(owner)
        if identity in seen:
            continue
        seen.add(identity)
        children = _children(owner)
        pending.extend(children)
        sources = _sources_on(owner)
        found.extend(sources)
    return found


def _children(owner) -> list:
    """The decsim values one object holds, directly or in a plain container."""
    state = getattr(owner, "__dict__", None)
    if state is None:
        return []
    children = []
    for value in state.values():
        held = _decsim_values(value)
        children.extend(held)
    return children


def _decsim_values(value) -> list:
    """The value itself when it is a decsim object, or the ones it holds.

    A container may hold containers (the pool keeps its units in a list
    per pool name), so the walk goes through them until it reaches an
    object of this package or something it does not follow.
    """
    if isinstance(value, (list, tuple, set)):
        return _from_items(value)
    if isinstance(value, dict):
        items = value.values()
        return _from_items(items)
    if _is_decsim(value):
        return [value]
    return []


def _from_items(items) -> list:
    """The decsim objects a container holds, however deeply."""
    found = []
    for item in items:
        held = _decsim_values(item)
        found.extend(held)
    return found


def _is_decsim(value) -> bool:
    """Whether the value is one of this package's own objects."""
    kind = type(value)
    module = kind.__module__
    return module.startswith("decsim.")


def _sources_on(owner) -> list:
    """The trace sources one object declares, by attribute name."""
    state = getattr(owner, "__dict__", None)
    if state is None:
        return []
    sources = []
    for name, value in state.items():
        if isinstance(value, trace_source.TraceSource):
            kind = type(owner)
            sources.append((kind.__name__, name, value))
    return sources


def test_no_source_is_heard_twice_by_one_listener_class():
    """One connection per (source, listener class), with every knob on."""
    point = gate_point.settings(**EVERY_KNOB)
    machine = machine_module.Machine.build(point, gate_point.SEED)

    census = _walk(machine)
    doubled = []
    for owner_name, source_name, source in census:
        classes = []
        for listener in source.listeners:
            heard_by = _listener_class(listener)
            classes.append(heard_by)
        if len(classes) != len(set(classes)):
            doubled.append((owner_name, source_name, classes))

    assert doubled == []
    assert len(census) > 30


def test_every_source_has_a_listener_when_every_knob_is_on():
    """A source nothing hears is a fire into an empty list for every run.

    With every knob on, each source a component declares is wired here
    or it is dead (rule 5); a new source added to a component without a
    connection in wiring.py fails this test by name.
    """
    point = gate_point.settings(**EVERY_KNOB)
    machine = machine_module.Machine.build(point, gate_point.SEED)

    census = _walk(machine)
    unheard = []
    for owner_name, source_name, source in census:
        if not source.has_listeners:
            unheard.append((owner_name, source_name))

    assert unheard == []


def test_every_listener_the_section_asks_for_is_built_and_heard():
    """A knob on builds its listener; a knob off leaves the field None."""
    asked_point = gate_point.settings(**EVERY_KNOB)
    silent_point = gate_point.settings()
    asked = machine_module.Machine.build(asked_point, gate_point.SEED)
    silent = machine_module.Machine.build(silent_point, gate_point.SEED)

    for machine in (asked, silent):
        assert machine.observation.log is not None
        assert machine.observation.round_events is not None
        assert machine.observation.stages is not None
    assert asked.observation.trace_writer is not None
    assert asked.observation.data_movement is not None
    assert asked.observation.decode_records is not None
    assert asked.observation.round_store_occupancy is not None
    assert asked.observation.decode_backlog is not None
    assert asked.observation.decoder_utilization is not None
    assert asked.observation.decoder_memory_occupancy is not None
    assert silent.observation.trace_writer is None
    assert silent.observation.data_movement is None
    assert silent.observation.decode_records is None
    assert silent.observation.round_store_occupancy is None
    assert silent.observation.decode_backlog is None
    assert silent.observation.decoder_utilization is None
    assert silent.observation.decoder_memory_occupancy is None


class PortOnlyDecoder:
    """A decoder row written outside decsim: the port, and no base class.

    It answers every method and every attribute of ports.Decoder,
    including the three observation sources, and inherits nothing of
    decsim's, so it is the row the wiring must recognise by the port.
    """

    fault_model_requirement = None
    decoder_evidence = frozenset()
    missing_evidence_reasons: dict = {}

    def __init__(self, latency_ticks: int) -> None:
        self.latency_ticks = latency_ticks
        self.stage_recorded = trace_source.TraceSource()
        self.window_checked = trace_source.TraceSource()
        self.forced_solve_unavailable = trace_source.TraceSource()

    def decode(self, job):
        """A correction of nothing; the trace is what this row is for."""
        return decoding_records.DecodeResult(job.operation_id, job.window_id)

    def latency(self, job) -> int:
        del job
        return self.latency_ticks

    def start(self, job, engine, on_result) -> None:
        """One stage per job, fired on the port's own source."""
        result = self.decode(job)
        end_tick = engine.now + self.latency_ticks
        record = staged_decoder.DecoderStageRecord(
            job.operation_id,
            job.window_id,
            "algorithm",
            None,
            engine.now,
            end_tick,
        )
        self.stage_recorded.fire(record)
        engine.schedule(self.latency_ticks, lambda: on_result(result))

    def cancel(self, job) -> None:
        del job

    def occupancy(self, job) -> int:
        del job
        return self.latency_ticks

    def pipeline_depth(self, job) -> int:
        del job
        return 1


def _machine_on(decoder, **observation):
    """The declared weak-only run with this row as the weak tier."""
    operation = declared_run.memory_operation(1)
    workload = declared_run.declared_workload([operation], 6)
    weak_decoder = decoder_settings.DecoderSettings(decoder=decoder)
    qpu = declared_run.declared_qpu()
    links = declared_run.declared_profile()
    controller = declared_run.declared_controller()
    pauli_frame = declared_run.declared_frame()
    watched = observe_settings.ObservationSettings(**observation)
    settings = machine_settings.MachineSettings(
        workload=workload,
        qpu=qpu,
        weak_decoder=weak_decoder,
        links=links,
        controller=controller,
        pauli_frame=pauli_frame,
        observation=watched,
    )
    return machine_module.Machine.build(settings, 0)


def test_a_decoder_row_that_only_fills_the_port_reaches_the_observers():
    """The wiring recognises a decoder by the port, not by a base class.

    A row that inherits nothing of decsim's decodes every window; before
    the walk asked the port it was invisible to the stage ledger, the
    referee audit and the trace.
    """
    ticks = decsim.config.microseconds_to_ticks(1.0)
    decoder = PortOnlyDecoder(ticks)
    machine = _machine_on(
        decoder, log="off", trace=observe_settings.CHROME_TRACE
    )
    result = machine.run()

    assert result.terminal_status == "complete"
    assert machine.observation.stages.records != []
    assert decoder.window_checked.has_listeners
    stage_events = []
    for event in machine.observation.trace_writer.events:
        if event.get("cat") == "stage":
            stage_events.append(event)
    assert stage_events != []


def _bare_observe(observation, engine, **components):
    """The narrator and the three listeners the run result reads, no more.

    No trace writer, no data movement, no flight recorder input, no stage
    ledger, no referee audit, no metrics, no window ledger, no round
    events, no command events, no queue depth, no occupancy. The narrator
    hears both of the engine's line sources, as the real wiring does when
    observation.log_component_io is on, because a source with no listener
    is never fired at all (decsim/engine.py log_io).
    """
    del observation
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    engine.io_line.connect(log.write)
    traffic_ledger = components["traffic_ledger"]
    links = components["links"]
    links.trace.transfer_delivered.connect(traffic_ledger.on_transfer)
    result_ledger = result_ledger_module.ResultLedger()
    window_manager = components["window_manager"]
    results = window_manager.results
    results.trace.operation_result_delivered.connect(
        result_ledger.operation_result_delivered
    )
    runtime_stamps = runtime_stamps_module.RuntimeStamps()
    execution_runtime = components["execution_runtime"]
    execution_runtime.trace.body_finished.connect(runtime_stamps.body_finished)
    return _bare_observation(
        engine, log, traffic_ledger, result_ledger, runtime_stamps
    )


def _bare_observation(
    engine, log, traffic_ledger, result_ledger, runtime_stamps
):
    """The record those three listeners hang on, every other field empty."""
    windows = window_ledger_module.WindowLedger()
    recorder = _bare_flight_recorder(engine, runtime_stamps)
    corrections = flight_recorder_module.FrameCorrections()
    queue_depth = queue_depth_module.QueueDepthLog()
    counters = controller_counters_module.ControllerCounters()
    commands = command_events_module.CommandEvents()
    stages = stage_records_module.StageLedger()
    rounds = round_events_module.RoundEventRecorder(engine)
    audit = referee_audit_module.RefereeAudit()
    shots = sampled_shots_module.SampledShots()
    return observation_module.Observation(
        log=log,
        windows=windows,
        results=result_ledger,
        traffic=traffic_ledger,
        flight_recorder=recorder,
        frame_corrections=corrections,
        trace_writer=None,
        data_movement=None,
        decode_records=None,
        runtime_stamps=runtime_stamps,
        queue_depth=queue_depth,
        controller_counters=counters,
        command_events=commands,
        stages=stages,
        decode_backlog=None,
        decoder_utilization=None,
        decoder_memory_occupancy=None,
        round_events=rounds,
        round_store_occupancy=None,
        referee_audit=audit,
        sampled_shots=shots,
    )


def _bare_flight_recorder(engine, runtime_stamps):
    """A recorder over empty ledgers, so no component is heard through it."""
    rounds = round_events_module.RoundEventRecorder(engine)
    windows = window_ledger_module.WindowLedger()
    commands = command_events_module.CommandEvents()
    corrections = flight_recorder_module.FrameCorrections()
    return flight_recorder_module.FlightRecorder(
        rounds,
        windows,
        runtime_stamps,
        commands,
        corrections,
        (),
    )


def test_the_narrator_log_is_byte_identical_with_and_without_the_observers(
    monkeypatch,
):
    """The log is a pure component product, on a real gate point.

    STYLE.md rule 7: observation is reached through callbacks a component
    fires, so a component runs with no observer at all. The log the
    frozen gate hashes is therefore written entirely by components; if an
    observer wrote one line into it, that line would be missing here.
    I7 Part 1 slice 4(f) moved the last such line, "NO FORCED SOLVE",
    out of the wiring and into the decoder manager that fires it.
    """
    wired_machine, _wired_result = gate_point.run()
    wired_lines = list(wired_machine.observation.log.lines)
    monkeypatch.setattr(wiring, "observe", _bare_observe)
    bare_machine, _bare_result = gate_point.run()
    bare_lines = list(bare_machine.observation.log.lines)

    assert bare_lines == wired_lines
    assert len(wired_lines) > 100


def test_a_run_with_only_those_listeners_gives_the_same_result_record(
    monkeypatch,
):
    """Every field of the gate point's result, between the two wirings."""
    wired_machine, wired = gate_point.run(**EVERY_KNOB)
    monkeypatch.setattr(wiring, "observe", _bare_observe)
    _bare_machine, bare = gate_point.run(**EVERY_KNOB)

    assert bare.terminal_status == wired.terminal_status
    assert bare.operation_results == wired.operation_results
    assert bare.link_traffic == wired.link_traffic
    assert bare.fully_done_ticks == wired.fully_done_ticks
    assert wired_machine.observation.data_movement is not None


def test_the_data_movement_report_is_the_only_field_that_needs_a_listener(
    monkeypatch,
):
    wired_machine, wired = gate_point.run(**EVERY_KNOB)
    monkeypatch.setattr(wiring, "observe", _bare_observe)
    _bare_machine, bare = gate_point.run(**EVERY_KNOB)
    wired_without = dataclasses.replace(wired, data_movement=None)

    assert wired.data_movement is not None
    assert bare.data_movement is None
    assert bare == wired_without
    assert wired_machine.observation.stages.records != []


def test_every_optional_listener_off_gives_the_same_result_as_every_one_on():
    """The observation section is a knob on what is recorded, not on the run."""
    _every_machine, everything = gate_point.run(**EVERY_KNOB)
    _no_machine, nothing = gate_point.run()

    assert nothing.operation_results == everything.operation_results
    assert nothing.fully_done_ticks == everything.fully_done_ticks
    assert nothing.link_traffic == everything.link_traffic
