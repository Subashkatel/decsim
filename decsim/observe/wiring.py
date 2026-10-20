"""Every listener of one run, built from the observation settings and wired.

The only place that knows which setting builds which listener, so a new
listener is one class in observe/ and one connection here. Observation
is reached through the sources a component fires, never through a port,
so a component runs with nothing connected. The wiring runs before the
workload is loaded, because the first operation is issued while it
loads and a later listener would miss its lines and stamps.
"""

import functools
from typing import Optional

import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.qpu as qpu_part
import decsim.build.readout as readout_part
import decsim.build.windows as windows_part
import decsim.decoders.decoder_pool as decoder_pool
import decsim.engine as engine_module
import decsim.links.settings as link_settings
import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.data_movement as data_movement_module
import decsim.observe.decode_records as decode_records_module
import decsim.observe.frame_corrections as frame_corrections_module
import decsim.observe.link_traffic as link_traffic
import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics
import decsim.observe.observation as observation_module
import decsim.observe.referee_audit as referee_audit_module
import decsim.observe.region_of_interest as region_of_interest_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_events as round_events_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.sampled_shots as sampled_shots_module
import decsim.observe.settings as observe_settings
import decsim.observe.stage_records as stage_records_module
import decsim.observe.trace_writer as trace_writer_module
import decsim.observe.window_ledger as window_ledger_module
import decsim.observe.window_outcomes as window_outcomes_module
import decsim.ports as ports
import decsim.statistics.registry as statistics_registry
import decsim.syndrome_buffer.syndrome_round_receiver as receiver_module


def observe(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    *,
    links: ports.Link,
    qpu: qpu_part.Qpu,
    control: control_part.Control,
    readout: readout_part.Readout,
    windows: windows_part.Windows,
    decoders: decoders_part.Decoders,
    process_name: str,
    traffic_ledger: link_traffic.TrafficLedger,
    run_statistics: statistics_registry.Registry,
    links_card: link_settings.FabricSettings,
) -> observation_module.Observation:
    """Every listener of the run, built and connected to what it hears.

    A component a run lacks reads as None. It is the run's one list of
    listeners in pipeline order, so it stays whole past the size prompt.
    """
    window_manager = windows.window_manager
    decoder_managers = _decoder_managers(decoders)
    keeps_records = not observation.folds_windows
    log = _connect_log(observation, engine, keeps_records)
    links.trace.transfer_delivered.connect(traffic_ledger.on_transfer)
    round_events = _connect_round_events(
        engine, qpu, control, readout, keeps_records
    )
    window_ledger = _connect_window_ledger(window_manager)
    result_ledger = _connect_result_ledger(window_manager)
    runtime_stamps = runtime_stamps_module.RuntimeStamps()
    _connect_runtime_stamps(
        control.execution_runtime, qpu.factory, runtime_stamps
    )
    controller_counters = _connect_controller_counters(control.idle_rounds)
    command_events = _connect_command_events(qpu.device)
    stages = _connect_stage_records(decoders, keeps_records)
    referee_audit = _connect_referee_audit(decoders)
    sampled_shots = _connect_sampled_shots(qpu.syndrome_source)
    window_outcomes = _window_outcomes(observation)
    if window_outcomes is not None:
        _connect_window_outcomes(
            window_outcomes, qpu.syndrome_source, window_manager
        )
    decode_records = _decode_records(observation, keeps_records)
    _connect_decode_records(decoder_managers, decode_records)
    gap_join = windows.gap_join
    confidence = _connect_confidence(gap_join, decoder_managers, keeps_records)
    store_names = _store_names(readout)
    trace_writer = _trace_writer(
        observation, engine, process_name, links_card, store_names
    )
    data_movement = _data_movement(observation, links_card, store_names)
    if data_movement is not None:
        _connect_data_movement(
            data_movement, links, qpu, readout, windows, decoders
        )
    if trace_writer is not None:
        _connect_trace_writer(
            trace_writer, links, qpu, control, readout, windows, decoders
        )
    decode_backlog = _decode_backlog(
        observation, engine, window_manager, decoder_managers, round_events
    )
    if decode_backlog is not None:
        run_statistics.on_reset(decode_backlog.reset)
    region_of_interest = _region_of_interest(observation, run_statistics)
    if region_of_interest is not None:
        qpu.device.trace.round_emitted.connect(
            region_of_interest.round_read_out
        )
    frame_corrections = _frame_corrections(control.pauli_frame, keeps_records)
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
        controller_counters=controller_counters,
        command_events=command_events,
        stages=stages,
        referee_audit=referee_audit,
        sampled_shots=sampled_shots,
        window_outcomes=window_outcomes,
        decode_backlog=decode_backlog,
        statistics=run_statistics,
        round_events=round_events,
        confidence=confidence,
        region_of_interest=region_of_interest,
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
    """Every pool's manager, the entry pool's first."""
    return decoders.managers


def _decoder_rows(decoders) -> list:
    """Every decoder row of the run's decoder units, in graph order."""
    units = [pool.unit for pool in decoders.pools]
    return decoder_pool.decoder_rows(units)


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


def _store_names(readout) -> tuple:
    """The name each store narrates and reports its copies under."""
    return tuple(parts.plan.intake.name for parts in readout.stores)


def _trace_writer(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    process_name: str,
    links_card: link_settings.FabricSettings,
    store_names: tuple,
) -> Optional[trace_writer_module.TraceWriter]:
    """The Chrome trace writer, only when the section asks for a trace."""
    if not observation.writes_trace:
        return None
    threads = trace_writer_module.thread_order(links_card, store_names)
    return trace_writer_module.TraceWriter(engine, process_name, threads)


def _data_movement(
    observation: observe_settings.ObservationSettings,
    links_card: link_settings.FabricSettings,
    store_names: tuple,
) -> Optional[data_movement_module.DataMovement]:
    """The copy, reference and move counters, only when the section asks."""
    if not observation.data_movement:
        return None
    class_by_hop = {}
    for path in links_card.paths:
        class_by_hop[path.name] = data_movement_module.memory_class_of_hop(path)
    store_name_set = frozenset(store_names)
    classes = data_movement_module.MemoryClasses(class_by_hop, store_name_set)
    return data_movement_module.DataMovement(classes)


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
    for parts in readout.stores:
        _connect_store_counts(data_movement, parts.store)
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
    for parts in readout.stores:
        _connect_store_trace(trace_writer, parts.store, parts.plan.label)
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
    ]
    for parts in readout.stores:
        sources.append(parts.receiver.trace.copy_made)
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
    sources.window_final.connect(trace_writer.window_final)
    sources.window_absorbed.connect(trace_writer.window_absorbed)
    for manager in decoder_managers:
        manager.outcomes.trace.verdict_given.connect(trace_writer.verdict_given)
    for edge_trace in sources.edge_traces:
        edge_trace.strong_window_held.connect(trace_writer.strong_window_held)
        edge_trace.strong_window_left.connect(trace_writer.strong_window_left)


def _decode_records(
    observation: observe_settings.ObservationSettings, keeps_records: bool
) -> Optional[decode_records_module.DecodeRecordLedger]:
    """The decode record ledger, when the records that read it ask.

    The switching records and the window outcomes both read each
    decode's syndrome weight off it; a stream that lets its windows go
    reads neither (observation.folds_windows).
    """
    is_asked = (
        observation.record_switching_windows
        or observation.record_window_outcomes
    )
    if not is_asked or not keeps_records:
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
    gap_join, decoder_managers: tuple, keeps_records: bool
) -> Optional[decode_records_module.ConfidenceLedger]:
    """The confidence ledger, only on a run that computes a confidence.

    It keeps every window's verdict result, which a stream that lets its
    windows go never reads (observation.folds_windows).
    """
    if gap_join is None or not keeps_records:
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
    execution_runtime.trace.operation_ready.connect(stamps.operation_ready)
    execution_runtime.trace.operation_issued.connect(stamps.operation_issued)
    execution_runtime.trace.operation_started.connect(stamps.operation_started)
    execution_runtime.trace.body_finished.connect(stamps.body_finished)
    execution_runtime.trace.decode_released.connect(stamps.decode_released)
    execution_runtime.trace.result_returned.connect(stamps.result_returned)
    factory.trace.state_delivered.connect(stamps.magic_state_delivered)


def _connect_round_events(
    engine: engine_module.Engine, qpu, control, readout, keeps_records: bool
) -> round_events_module.RoundEventRecorder:
    """The recorder hears every round event, output and strong landing.

    A written store's receiving end is on the readout's path; a landed
    store's is heard after the outputs, with its stores.
    """
    round_events = _round_event_recorder(engine, keeps_records)
    components = [
        qpu.device,
        readout.assembler,
        readout.held_rounds,
        readout.transmitter,
    ]
    landed = []
    for parts in readout.stores:
        if parts.plan.intake.shape == receiver_module.WRITTEN:
            components.append(parts.receiver)
            continue
        landed.append(parts)
    for component in components:
        component.trace.round_event.connect(round_events.record)
    qpu.device.trace.round_emitted.connect(round_events.round_read_out)
    idle_rounds = control.idle_rounds
    idle_rounds.trace.idle_round_emitted.connect(
        round_events.idle_round_read_out
    )
    instruction_output = control.instruction_output
    instruction_output.trace.output_event.connect(round_events.output)
    for parts in landed:
        parts.store.trace.round_stored.connect(round_events.round_stored)
    for parts in landed:
        parts.receiver.trace.round_event.connect(round_events.record)
    return round_events


def _round_event_recorder(
    engine: engine_module.Engine, keeps_records: bool
) -> round_events_module.RoundEventRecorder:
    """The run's recorder, or a stream's, which keeps no row."""
    if keeps_records:
        return round_events_module.RoundEventRecorder(engine)
    return round_events_module.FoldingRoundEventRecorder(engine)


def _connect_stage_records(
    decoders, keeps_records: bool
) -> stage_records_module.StageLedger:
    """The run's stage history, heard from every decoder row."""
    stages = stage_records_module.StageLedger(keeps_records)
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


def _window_outcomes(
    observation: observe_settings.ObservationSettings,
) -> Optional[window_outcomes_module.WindowOutcomes]:
    """The windows' answers and labels, only when the section asks."""
    if not observation.record_window_outcomes:
        return None
    return window_outcomes_module.WindowOutcomes()


def _connect_window_outcomes(
    window_outcomes: window_outcomes_module.WindowOutcomes,
    syndrome_source,
    window_manager,
) -> None:
    """The source's fired errors and every delivered result's windows."""
    syndrome_source.errors_sampled.connect(window_outcomes.errors_sampled)
    results = window_manager.results
    results.trace.contributions_delivered.connect(
        window_outcomes.contributions_delivered
    )


def _frame_corrections(
    pauli_frame, keeps_records: bool
) -> frame_corrections_module.FrameCorrections:
    """The frame's landed corrections."""
    corrections = frame_corrections_module.FrameCorrections(keeps_records)
    if pauli_frame is None:
        return corrections
    pauli_frame.trace.correction_committed.connect(
        corrections.correction_committed
    )
    return corrections


def _decode_backlog(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    window_manager,
    decoder_managers: tuple,
    round_events: round_events_module.RoundEventRecorder,
) -> Optional[metrics.DecodeBacklog]:
    """The backlog sampler, after every action, only when asked for."""
    if not observation.backlog_trace:
        return None
    decode_backlog = metrics.DecodeBacklog(
        window_manager, decoder_managers, round_events
    )
    at_tick_end = functools.partial(_sample_at_tick_end, engine, decode_backlog)
    engine.action_done.connect(at_tick_end)
    return decode_backlog


def _region_of_interest(
    observation: observe_settings.ObservationSettings,
    run_statistics: statistics_registry.Registry,
) -> Optional[region_of_interest_module.RegionOfInterest]:
    """The warm-up's end, only when the run leaves one out."""
    if observation.warmup_rounds == 0:
        return None
    return region_of_interest_module.RegionOfInterest(
        run_statistics, observation.warmup_rounds
    )


def _sample_at_tick_end(
    engine: engine_module.Engine,
    decode_backlog: metrics.DecodeBacklog,
    tick: int,
) -> None:
    """Sample the backlog after a tick's last action only.

    The peak is over the counts ticks end at, so an action with another
    due at its tick changes nothing the sampler keeps.
    """
    if engine.has_action_due_now():
        return
    decode_backlog.observe(tick)


def _connect_log(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    keeps_records: bool,
) -> log_writers.LogWriter:
    """The narrator's listeners: the record, the console when asked.

    Connecting a listener to io_line is what turns the component I/O
    lines on, gem5's named debug flag (src/base/debug.hh Flag). A stream
    that lets its windows go keeps no record (observation.folds_windows).
    """
    log = log_writers.LogWriter()
    if keeps_records:
        engine.line.connect(log.write)
    if observation.log_component_io:
        engine.io_line.connect(log.write)
    if observation.prints_log:
        printer = log_writers.ConsolePrinter()
        engine.line.connect(printer.write)
        if observation.log_component_io:
            engine.io_line.connect(printer.write)
    return log
