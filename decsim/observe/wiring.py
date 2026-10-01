"""Every listener of one run, built from the observation section and wired.

The Machine builds the parts; this module builds what watches them
and connects each listener to the sources it hears, in pipeline order.
It is the only place that knows which yaml key builds which listener, so
a new listener is one class in observe/ and one connection here
(STYLE.md rule 7: observation is reached through the callbacks a
component fires, never through a port, so a component runs with nothing
connected).

The wiring runs after every component is built and before the workload
is loaded, because the program's first operation is issued while it
loads: the narrator's first line, the QPU's first command and the
runtime's first stamps are all fired there, and a listener connected
afterwards would miss them.
"""

import functools
from typing import Any, Optional

import decsim.decoders.decoder_pool as decoder_pool
import decsim.engine as engine_module
import decsim.observe.burst_flags as burst_flags_module
import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.data_movement as data_movement_module
import decsim.observe.decode_records as decode_records_module
import decsim.observe.frame_corrections as frame_corrections_module
import decsim.observe.link_traffic as link_traffic
import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics
import decsim.observe.observation as observation_module
import decsim.observe.queue_depth as queue_depth_module
import decsim.observe.referee_audit as referee_audit_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_events as round_events_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.sampled_shots as sampled_shots_module
import decsim.observe.settings as observe_settings
import decsim.observe.stage_records as stage_records_module
import decsim.observe.trace_writer as trace_writer_module
import decsim.observe.window_ledger as window_ledger_module


def observe(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    *,
    links: Any,
    qpu: Any,
    control: Any,
    readout: Any,
    windows: Any,
    decoders: Any,
    process_name: str,
    traffic_ledger: link_traffic.TrafficLedger,
) -> observation_module.Observation:
    """Every listener of the run, built and connected to what it hears.

    A listener reaches the component it hears through the machine's part
    that holds it (decsim/build); the parts sit above this package in the
    package order, so they arrive untyped. A component a run may not have
    reads as None, and the listeners that hear it take None; the decoder
    managers, one per side, are heard as one tuple by every listener of
    the decode path. It stays whole past the size prompt: it is the run's
    one list of listeners, in pipeline order, one built or connected per
    line, and a split would only hand the list from one half to the
    other.
    """
    window_manager = windows.window_manager
    decoder_managers = _decoder_managers(decoders)
    log = _connect_log(observation, engine)
    links.trace.transfer_delivered.connect(traffic_ledger.on_transfer)
    round_events = _connect_round_events(engine, qpu, control, readout)
    window_ledger = _connect_window_ledger(window_manager)
    result_ledger = _connect_result_ledger(window_manager)
    runtime_stamps = runtime_stamps_module.RuntimeStamps()
    _connect_runtime_stamps(
        control.execution_runtime, qpu.factory, runtime_stamps
    )
    queue_depth = _connect_queue_depth(decoder_managers)
    controller_counters = _connect_controller_counters(control.idle_rounds)
    command_events = _connect_command_events(qpu.device)
    stages = _connect_stage_records(decoders)
    referee_audit = _connect_referee_audit(decoders)
    sampled_shots = _connect_sampled_shots(qpu.syndrome_source)
    decode_records = _decode_records(observation)
    _connect_decode_records(decoder_managers, decode_records)
    confidence = _connect_confidence(
        windows.escalation_policy, decoder_managers
    )
    trace_writer = _trace_writer(observation, engine, process_name)
    data_movement = _data_movement(observation)
    if data_movement is not None:
        _connect_data_movement(
            data_movement, links, qpu, readout, windows, decoders
        )
    if trace_writer is not None:
        _connect_trace_writer(
            trace_writer, links, qpu, control, readout, windows, decoders
        )
    decode_backlog = _decode_backlog(
        observation, engine, window_manager, decoder_managers
    )
    decoder_utilization = _decoder_utilization(engine, decoder_managers)
    decoder_memory_occupancy = _decoder_memory_occupancy(
        observation, engine, decoder_managers
    )
    frame_corrections = _frame_corrections(control.pauli_frame)
    burst_flags = _connect_burst_flags(windows.burst_detector)
    return observation_module.Observation(
        log=log,
        windows=window_ledger,
        results=result_ledger,
        traffic=traffic_ledger,
        frame_corrections=frame_corrections,
        trace_writer=trace_writer,
        data_movement=data_movement,
        decode_records=decode_records,
        runtime_stamps=runtime_stamps,
        queue_depth=queue_depth,
        controller_counters=controller_counters,
        command_events=command_events,
        stages=stages,
        referee_audit=referee_audit,
        sampled_shots=sampled_shots,
        decode_backlog=decode_backlog,
        decoder_utilization=decoder_utilization,
        decoder_memory_occupancy=decoder_memory_occupancy,
        round_events=round_events,
        burst_flags=burst_flags,
        confidence=confidence,
    )


def _connect_window_ledger(window_manager) -> window_ledger_module.WindowLedger:
    """The window records: the plan's at build, each stream's as it grows."""
    ledger = window_ledger_module.WindowLedger()
    planned = window_manager.planned_windows()
    ledger.load_planned(planned)
    sources = window_manager.window_sources()
    sources.window_planned.connect(ledger.window_planned)
    sources.window_committed.connect(ledger.window_committed)
    return ledger


def _connect_result_ledger(window_manager) -> result_ledger_module.ResultLedger:
    """The logical results, heard once per operation as they are delivered."""
    ledger = result_ledger_module.ResultLedger()
    results = window_manager.results
    results.trace.operation_result_delivered.connect(
        ledger.operation_result_delivered
    )
    return ledger


def _decoder_managers(decoders) -> tuple:
    """The chip's manager and, when the run escalates, the host's."""
    managers = [decoders.decoder_manager]
    strong = decoders.strong_decoder_manager
    if strong is not None:
        managers.append(strong)
    return tuple(managers)


def _decoder_rows(decoders) -> list:
    """Every decoder row of the run's two decoder units, either absent."""
    units = (decoders.primary_decoder, decoders.strong_decoder)
    return decoder_pool.decoder_rows(units)


def _connect_queue_depth(decoder_managers) -> queue_depth_module.QueueDepthLog:
    """The waiting jobs, sampled at every change of either queue's depth."""
    depth_log = queue_depth_module.QueueDepthLog()
    for manager in decoder_managers:
        pool_name = manager.pool.name
        depth_changed = functools.partial(depth_log.depth_changed, pool_name)
        manager.queue.trace.depth_changed.connect(depth_changed)
    return depth_log


def _connect_controller_counters(
    idle_rounds,
) -> controller_counters_module.ControllerCounters:
    """The controller's idle rounds, counted as they are emitted."""
    counters = controller_counters_module.ControllerCounters()
    idle_rounds.trace.idle_round_emitted.connect(counters.idle_round_emitted)
    return counters


def _connect_command_events(qpu) -> command_events_module.CommandEvents:
    """Every command the QPU received and started."""
    events = command_events_module.CommandEvents()
    qpu.trace.command_event.connect(events.command_event)
    return events


def _trace_writer(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    process_name: str,
) -> Optional[trace_writer_module.TraceWriter]:
    """The Chrome trace writer, only when the section asks for a trace."""
    if not observation.writes_trace:
        return None
    return trace_writer_module.TraceWriter(engine, process_name)


def _data_movement(
    observation: observe_settings.ObservationSettings,
) -> Optional[data_movement_module.DataMovement]:
    """The copy, reference and move counters, only when the section asks."""
    if not observation.data_movement:
        return None
    return data_movement_module.DataMovement()


def _connect_data_movement(
    data_movement: data_movement_module.DataMovement,
    links,
    qpu,
    readout,
    windows,
    decoders,
) -> None:
    """The counters hear every copy, reference and residence.

    The hops and residences of the data path's hop table, each from the
    component where it happens.
    """
    qpu.device.trace.round_emitted.connect(data_movement.round_emitted)
    links.trace.transfer_delivered.connect(data_movement.transfer_delivered)
    _connect_store_counts(data_movement, readout.weak_syndrome_buffer)
    strong_syndrome_buffer = readout.strong_syndrome_buffer
    if strong_syndrome_buffer is not None:
        _connect_store_counts(data_movement, strong_syndrome_buffer)
    for manager in _decoder_managers(decoders):
        for source in manager.reference_sources():
            source.connect(data_movement.hold_registered)
    for source in _copy_sources(readout, windows, decoders):
        source.connect(data_movement.copy_made)
    formation = readout.detection_events
    formation.trace.state_held.connect(data_movement.formation_state_held)


def _connect_trace_writer(
    trace_writer: trace_writer_module.TraceWriter,
    links,
    qpu,
    control,
    readout,
    windows,
    decoders,
) -> None:
    """The trace hears every hop and residence, in the order of the path."""
    device = qpu.device
    device.trace.round_emitted.connect(trace_writer.round_emitted)
    device.trace.command_event.connect(trace_writer.command_event)
    links.trace.transfer_delivered.connect(trace_writer.transfer_delivered)
    links.trace.frame_landed.connect(trace_writer.frame_landed)
    for source in _copy_sources(readout, windows, decoders):
        source.connect(trace_writer.copy_made)
    _connect_controller_trace(trace_writer, readout)
    _connect_store_trace(
        trace_writer, readout.weak_syndrome_buffer, "weak syndrome buffer"
    )
    strong_syndrome_buffer = readout.strong_syndrome_buffer
    if strong_syndrome_buffer is not None:
        _connect_store_trace(
            trace_writer, strong_syndrome_buffer, "strong syndrome buffer"
        )
    decoder_managers = _decoder_managers(decoders)
    for index, manager in enumerate(decoder_managers):
        _connect_decoder_trace(trace_writer, index, manager)
    decoder_rows = _decoder_rows(decoders)
    for decoder in decoder_rows:
        decoder.stage_recorded.connect(trace_writer.stage_recorded)
    _connect_window_trace(
        trace_writer, windows.window_manager, decoder_managers
    )
    _connect_frame_trace(trace_writer, control.pauli_frame)


def _connect_controller_trace(
    trace_writer: trace_writer_module.TraceWriter, readout
) -> None:
    """The packing stage, the line in front of it and the one after it."""
    assembler = readout.assembler
    capacity = assembler.settings.packing_rounds_in_flight
    in_assembly = functools.partial(trace_writer.round_in_assembly, capacity)
    assembler.trace.round_event.connect(in_assembly)
    waiting = functools.partial(
        trace_writer.round_waiting_for_a_place, capacity
    )
    readout.packing_line.trace.round_event.connect(waiting)
    held_rounds = readout.held_rounds
    held_rounds.trace.round_event.connect(trace_writer.round_held_for_room)


def _connect_frame_trace(
    trace_writer: trace_writer_module.TraceWriter, pauli_frame
) -> None:
    """The frame's corrections; a run without a frame has none to hear."""
    if pauli_frame is None:
        return
    pauli_frame.trace.correction_accepted.connect(
        trace_writer.correction_accepted
    )
    pauli_frame.trace.correction_committed.connect(
        trace_writer.correction_committed
    )


def _connect_store_counts(
    data_movement: data_movement_module.DataMovement, store
) -> None:
    """One store's references: registered, transferred and released."""
    store.trace.hold_registered.connect(data_movement.hold_registered)
    store.trace.hold_transferred.connect(data_movement.hold_transferred)
    store.trace.hold_released.connect(data_movement.hold_released)


def _copy_sources(readout, windows, decoders) -> list:
    """Every copy_made source of the data path, in hop order."""
    sources = [
        readout.controller.trace.copy_made,
        readout.assembler.trace.copy_made,
        readout.weak_syndrome_round_receiver.trace.copy_made,
    ]
    strong_syndrome_round_receiver = readout.strong_syndrome_round_receiver
    if strong_syndrome_round_receiver is not None:
        sources.append(strong_syndrome_round_receiver.trace.copy_made)
    for manager in _decoder_managers(decoders):
        for source in manager.copy_sources():
            sources.append(source)
    for source in windows.window_manager.copy_sources():
        sources.append(source)
    return sources


def _connect_store_trace(
    trace_writer: trace_writer_module.TraceWriter, store, store_name: str
) -> None:
    """One store's residences, its occupancy, its holds and its accesses."""
    capacity_bits = store.capacity_bits()
    stored = functools.partial(
        trace_writer.round_stored, store_name, capacity_bits
    )
    store.trace.round_stored.connect(stored)
    published = functools.partial(trace_writer.round_published, store_name)
    store.trace.round_published.connect(published)
    released = functools.partial(trace_writer.round_released, store_name)
    store.trace.round_released.connect(released)
    registered = functools.partial(trace_writer.hold_registered, store_name)
    store.trace.hold_registered.connect(registered)
    transferred = functools.partial(trace_writer.hold_transferred, store_name)
    store.trace.hold_transferred.connect(transferred)
    hold_released = functools.partial(trace_writer.hold_released, store_name)
    store.trace.hold_released.connect(hold_released)
    access_served = functools.partial(trace_writer.access_served, store_name)
    store.trace.access_served.connect(access_served)


def _connect_decoder_trace(
    trace_writer: trace_writer_module.TraceWriter,
    index: int,
    decoder_manager,
) -> None:
    """One manager's ready queue, its units' services and their memories."""
    queue = decoder_manager.queue
    queue.trace.job_enqueued.connect(trace_writer.job_enqueued)
    queue.trace.job_withdrawn.connect(trace_writer.job_withdrawn)
    depth_changed = functools.partial(trace_writer.depth_changed, index)
    queue.trace.depth_changed.connect(depth_changed)
    service = decoder_manager.service
    service.trace.job_dispatched.connect(trace_writer.job_dispatched)
    service.trace.input_landed.connect(trace_writer.input_landed)
    service.trace.job_started.connect(trace_writer.job_started)
    service.trace.job_finished.connect(trace_writer.job_finished)
    for unit in decoder_manager.pool.units:
        memory = unit.memory
        deposited = functools.partial(
            trace_writer.memory_deposited, memory.name
        )
        memory.trace.deposited.connect(deposited)
        taken = functools.partial(trace_writer.memory_taken, memory.name)
        memory.trace.taken.connect(taken)


def _connect_window_trace(
    trace_writer: trace_writer_module.TraceWriter,
    window_manager,
    decoder_managers,
) -> None:
    """The windows a stream lays, their verdicts, commits and absorptions."""
    sources = window_manager.window_sources()
    sources.window_planned.connect(trace_writer.window_planned)
    sources.window_data_complete.connect(trace_writer.window_ready)
    sources.solve_held.connect(trace_writer.solve_held)
    sources.window_committed.connect(trace_writer.window_committed)
    sources.window_absorbed.connect(trace_writer.window_absorbed)
    for manager in decoder_managers:
        manager.outcomes.trace.verdict_given.connect(trace_writer.verdict_given)
    strong_redecode = window_manager.strong_redecode
    if strong_redecode is None:
        return
    strong_redecode.trace.strong_window_held.connect(
        trace_writer.strong_window_held
    )
    strong_redecode.trace.strong_window_left.connect(
        trace_writer.strong_window_left
    )


def _decode_records(
    observation: observe_settings.ObservationSettings,
) -> Optional[decode_records_module.DecodeRecordLedger]:
    """The switching study's record ledger, only when the section asks."""
    if not observation.record_switching_windows:
        return None
    return decode_records_module.DecodeRecordLedger()


def _connect_decode_records(
    decoder_managers: tuple,
    decode_records: Optional[decode_records_module.DecodeRecordLedger],
) -> None:
    """The ledger hears every request's end on either side's manager."""
    if decode_records is None:
        return
    for manager in decoder_managers:
        outcomes = manager.outcomes
        outcomes.trace.request_ended.connect(decode_records.request_ended)


def _connect_confidence(
    escalation_policy, decoder_managers: tuple
) -> Optional[decode_records_module.ConfidenceLedger]:
    """The confidence ledger, only when a confidence decides the verdict."""
    if not escalation_policy.decides_on_a_confidence:
        return None
    confidence = decode_records_module.ConfidenceLedger()
    for manager in decoder_managers:
        outcomes = manager.outcomes
        outcomes.trace.request_ended.connect(confidence.request_ended)
    return confidence


def _connect_runtime_stamps(
    execution_runtime, factory, stamps: runtime_stamps_module.RuntimeStamps
) -> None:
    """The stamps hear every tick of an operation's life and its waits."""
    execution_runtime.trace.operation_issued.connect(stamps.operation_issued)
    execution_runtime.trace.operation_started.connect(stamps.operation_started)
    execution_runtime.trace.body_finished.connect(stamps.body_finished)
    execution_runtime.trace.decode_released.connect(stamps.decode_released)
    execution_runtime.trace.result_returned.connect(stamps.result_returned)
    factory.trace.state_delivered.connect(stamps.magic_state_delivered)


def _connect_round_events(
    engine: engine_module.Engine, qpu, control, readout
) -> round_events_module.RoundEventRecorder:
    """The recorder hears every round event, output and strong landing."""
    round_events = round_events_module.RoundEventRecorder(engine)
    components = (
        qpu.device,
        readout.assembler,
        readout.held_rounds,
        readout.transmitter,
        readout.weak_syndrome_round_receiver,
    )
    for component in components:
        component.trace.round_event.connect(round_events.record)
    instruction_output = control.instruction_output
    instruction_output.trace.output_event.connect(round_events.output)
    strong_syndrome_buffer = readout.strong_syndrome_buffer
    strong_syndrome_round_receiver = readout.strong_syndrome_round_receiver
    if strong_syndrome_buffer is not None:
        strong_syndrome_buffer.trace.round_stored.connect(
            round_events.round_stored
        )
    if strong_syndrome_round_receiver is not None:
        strong_syndrome_round_receiver.trace.round_event.connect(
            round_events.record
        )
    return round_events


def _connect_stage_records(decoders) -> stage_records_module.StageLedger:
    """The run's stage history, heard from every decoder row."""
    stages = stage_records_module.StageLedger()
    decoder_rows = _decoder_rows(decoders)
    for decoder in decoder_rows:
        decoder.stage_recorded.connect(stages.stage_recorded)
    return stages


def _connect_referee_audit(decoders) -> referee_audit_module.RefereeAudit:
    """The referee's checks, heard from every decoder row."""
    audit = referee_audit_module.RefereeAudit()
    decoder_rows = _decoder_rows(decoders)
    for decoder in decoder_rows:
        decoder.window_checked.connect(audit.window_checked)
    return audit


def _connect_sampled_shots(
    syndrome_source,
) -> sampled_shots_module.SampledShots:
    """Every shot the source draws, with the circuit it was drawn from."""
    shots = sampled_shots_module.SampledShots()
    syndrome_source.shot_sampled.connect(shots.shot_sampled)
    return shots


def _frame_corrections(
    pauli_frame,
) -> frame_corrections_module.FrameCorrections:
    """The frame's landed corrections."""
    corrections = frame_corrections_module.FrameCorrections()
    if pauli_frame is None:
        return corrections
    pauli_frame.trace.correction_committed.connect(
        corrections.correction_committed
    )
    return corrections


def _connect_burst_flags(
    burst_detector,
) -> Optional[burst_flags_module.BurstFlags]:
    """The rounds the detector fired on; a run without one has none."""
    if burst_detector is None:
        return None
    flags = burst_flags_module.BurstFlags()
    burst_detector.trace.round_flagged.connect(flags.round_flagged)
    return flags


def _decoder_utilization(
    engine: engine_module.Engine, decoder_managers
) -> metrics.DecoderUtilization:
    """The busy-unit integral, stepping at every pool's claims and returns.

    Always built: every run's pool columns read each tier's busy
    fraction off it.
    """
    units_by_pool = {}
    for manager in decoder_managers:
        pool = manager.pool
        units_by_pool[pool.name] = len(pool.units)
    utilization = metrics.DecoderUtilization(engine, units_by_pool)
    for manager in decoder_managers:
        pool = manager.pool
        pool.trace.unit_busy.connect(utilization.unit_busy)
        pool.trace.unit_freed.connect(utilization.unit_freed)
    return utilization


def _decode_backlog(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    window_manager,
    decoder_managers: tuple,
) -> Optional[metrics.DecodeBacklog]:
    """The backlog sampler, after every action, only when asked for."""
    if not observation.backlog_trace:
        return None
    decode_backlog = metrics.DecodeBacklog(window_manager, decoder_managers)
    engine.action_done.connect(decode_backlog.observe)
    return decode_backlog


def _decoder_memory_occupancy(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    decoder_managers: tuple,
) -> Optional[metrics.DecoderMemoryOccupancy]:
    """The held-bit integral of every unit memory, only when asked for."""
    if not observation.decoder_memory_occupancy:
        return None
    units = []
    for manager in decoder_managers:
        units.extend(manager.pool.units)
    capacity_by_unit = {}
    for unit in units:
        capacity_by_unit[unit.name] = unit.memory.capacity_bits
    occupancy = metrics.DecoderMemoryOccupancy(engine, capacity_by_unit)
    for unit in units:
        deposited = functools.partial(occupancy.deposited, unit.name)
        unit.memory.trace.deposited.connect(deposited)
        taken = functools.partial(occupancy.taken, unit.name)
        unit.memory.trace.taken.connect(taken)
    return occupancy


def _connect_log(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
) -> log_writers.LogWriter:
    """The narrator's listeners: the record always, the console when asked.

    Connecting a listener to io_line is what turns the component I/O
    lines on, gem5's named debug flag (src/base/debug.hh Flag).
    """
    log = log_writers.LogWriter()
    engine.line.connect(log.write)
    if observation.log_component_io:
        engine.io_line.connect(log.write)
    if observation.prints_log:
        printer = log_writers.ConsolePrinter()
        engine.line.connect(printer.write)
        if observation.log_component_io:
            engine.io_line.connect(printer.write)
    return log
