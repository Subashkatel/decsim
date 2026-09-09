"""Every listener of one run, built from the observation section and wired.

The Machine builds the components; this module builds what watches them
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
from typing import Optional

import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.observe.command_events as command_events_module
import decsim.observe.controller_counters as controller_counters_module
import decsim.observe.data_movement as data_movement_module
import decsim.observe.decode_records as decode_records_module
import decsim.observe.flight_recorder as flight_recorder_module
import decsim.observe.link_traffic as link_traffic
import decsim.observe.log_writers as log_writers
import decsim.observe.metrics as metrics
import decsim.observe.observation as observation_module
import decsim.observe.queue_depth as queue_depth_module
import decsim.observe.referee_audit as referee_audit_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_events as round_events_module
import decsim.observe.round_store_occupancy as round_store_occupancy_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.sampled_shots as sampled_shots_module
import decsim.observe.settings as observe_settings
import decsim.observe.stage_records as stage_records_module
import decsim.observe.trace_writer as trace_writer_module
import decsim.observe.window_ledger as window_ledger_module
import decsim.ports as ports
import decsim.records.log_sources as log_sources
import decsim.seeding as seeding


def observe(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    *,
    process_name: str,
    operations: tuple,
    traffic_ledger: link_traffic.TrafficLedger,
    links,
    qpu,
    syndrome_source,
    controller,
    idle_rounds,
    assembler,
    held_rounds,
    round_writer,
    transmitter,
    instruction_output,
    round_store,
    strong_round_store,
    strong_round_writer,
    execution_runtime,
    pauli_frame,
    decoder_manager,
    window_manager,
    pool,
) -> observation_module.Observation:
    """Every listener of the run, built and connected to what it hears."""
    log = _connect_log(observation, engine)
    links.trace.transfer_delivered.connect(traffic_ledger.on_transfer)
    round_events = _connect_round_events(
        engine,
        controller=controller,
        assembler=assembler,
        held_rounds=held_rounds,
        transmitter=transmitter,
        instruction_output=instruction_output,
        strong_round_store=strong_round_store,
    )
    round_store_occupancy = _round_store_occupancy(
        observation, engine, round_store
    )
    window_ledger = _connect_window_ledger(window_manager)
    result_ledger = _connect_result_ledger(window_manager)
    runtime_stamps = runtime_stamps_module.RuntimeStamps()
    _connect_runtime_stamps(execution_runtime, runtime_stamps)
    queue_depth = _connect_queue_depth(decoder_manager)
    controller_counters = _connect_controller_counters(idle_rounds)
    command_events = _connect_command_events(qpu)
    stages = _connect_stage_records(pool)
    referee_audit = _connect_referee_audit(pool)
    sampled_shots = _connect_sampled_shots(syndrome_source)
    _connect_unpinnable_observables(engine, pool)
    decode_records = _decode_records(observation)
    _connect_decode_records(decoder_manager, decode_records)
    trace_writer = _trace_writer(observation, engine, process_name)
    data_movement = _data_movement(observation)
    _connect_data_path(
        trace_writer,
        data_movement,
        links=links,
        qpu=qpu,
        controller=controller,
        assembler=assembler,
        round_writer=round_writer,
        round_store=round_store,
        strong_round_store=strong_round_store,
        strong_round_writer=strong_round_writer,
        decoder_manager=decoder_manager,
        window_manager=window_manager,
        pauli_frame=pauli_frame,
        pool=pool,
    )
    return _assembled(
        observation,
        engine,
        log,
        window_manager,
        decoder_manager,
        window_ledger=window_ledger,
        result_ledger=result_ledger,
        traffic_ledger=traffic_ledger,
        decode_records=decode_records,
        runtime_stamps=runtime_stamps,
        queue_depth=queue_depth,
        controller_counters=controller_counters,
        command_events=command_events,
        stages=stages,
        referee_audit=referee_audit,
        sampled_shots=sampled_shots,
        round_events=round_events,
        round_store_occupancy=round_store_occupancy,
        pauli_frame=pauli_frame,
        operations=operations,
        trace_writer=trace_writer,
        data_movement=data_movement,
    )


def _connect_window_ledger(window_manager) -> window_ledger_module.WindowLedger:
    """The window records: the plan's at build, each stream's as it grows."""
    ledger = window_ledger_module.WindowLedger()
    planned = window_manager.planned_windows()
    ledger.load_planned(planned)
    sources = window_manager.window_sources()
    sources.window_planned.connect(ledger.window_planned)
    sources.window_committed.connect(ledger.window_committed)
    sources.window_absorbed.connect(ledger.window_absorbed)
    return ledger


def _connect_result_ledger(window_manager) -> result_ledger_module.ResultLedger:
    """The logical results, heard once per operation as they are delivered."""
    ledger = result_ledger_module.ResultLedger()
    results = window_manager.results
    results.trace.operation_result_delivered.connect(
        ledger.operation_result_delivered
    )
    return ledger


def _connect_queue_depth(decoder_manager) -> queue_depth_module.QueueDepthLog:
    """The waiting jobs, sampled at every change of the queue's depth."""
    depth_log = queue_depth_module.QueueDepthLog()
    decoder_manager.queue.trace.depth_changed.connect(depth_log.depth_changed)
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


def _round_store_occupancy(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    round_store,
):
    """The L5 listener on Buffer 0, only when the observation asks."""
    if not observation.round_store_occupancy:
        return None
    occupancy = round_store_occupancy_module.RoundStoreOccupancy(engine)
    round_store.trace.round_stored.connect(occupancy.round_stored)
    round_store.trace.round_released.connect(occupancy.round_released)
    return occupancy


def _trace_writer(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    process_name: str,
) -> Optional[trace_writer_module.TraceWriter]:
    """The Chrome trace writer, only when the section names a path."""
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


def _connect_data_path(
    trace_writer: Optional[trace_writer_module.TraceWriter],
    data_movement: Optional[data_movement_module.DataMovement],
    *,
    links,
    qpu,
    controller,
    assembler,
    round_writer,
    round_store,
    strong_round_store,
    strong_round_writer,
    decoder_manager,
    window_manager,
    pauli_frame,
    pool,
) -> None:
    """Hand the trace and the counters every source of the data path.

    The hops and residences of the data path's hop table, each from
    the component where it happens; a run with neither listener connects
    nothing and fires into empty lists.
    """
    if data_movement is not None:
        qpu.trace.round_emitted.connect(data_movement.round_emitted)
        links.trace.transfer_delivered.connect(data_movement.transfer_delivered)
        _connect_store_counts(data_movement, round_store)
        if strong_round_store is not None:
            _connect_store_counts(data_movement, strong_round_store)
        for source in decoder_manager.reference_sources():
            source.connect(data_movement.hold_registered)
        for source in _copy_sources(
            controller,
            assembler,
            round_writer,
            strong_round_writer,
            decoder_manager,
            window_manager,
        ):
            source.connect(data_movement.copy_made)
    if trace_writer is None:
        return
    qpu.trace.round_emitted.connect(trace_writer.round_emitted)
    qpu.trace.command_event.connect(trace_writer.command_event)
    links.trace.transfer_delivered.connect(trace_writer.transfer_delivered)
    for source in _copy_sources(
        controller,
        assembler,
        round_writer,
        strong_round_writer,
        decoder_manager,
        window_manager,
    ):
        source.connect(trace_writer.copy_made)
    in_assembly = functools.partial(
        trace_writer.round_in_assembly,
        assembler.settings.packing_rounds_in_flight,
    )
    assembler.trace.round_event.connect(in_assembly)
    _connect_store_trace(trace_writer, round_store, "Buffer 0")
    if strong_round_store is not None:
        _connect_store_trace(trace_writer, strong_round_store, "Buffer 1")
    _connect_decoder_trace(trace_writer, decoder_manager, pool)
    _connect_window_trace(trace_writer, window_manager, decoder_manager)
    if pauli_frame is not None:
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


def _copy_sources(
    controller,
    assembler,
    round_writer,
    strong_round_writer,
    decoder_manager,
    window_manager,
) -> list:
    """Every copy_made source of the data path, in hop order."""
    sources = [
        controller.trace.copy_made,
        assembler.trace.copy_made,
        round_writer.trace.copy_made,
    ]
    if strong_round_writer is not None:
        sources.append(strong_round_writer.trace.copy_made)
    for source in decoder_manager.copy_sources():
        sources.append(source)
    for source in window_manager.copy_sources():
        sources.append(source)
    return sources


def _connect_store_trace(
    trace_writer: trace_writer_module.TraceWriter, store, store_name: str
) -> None:
    """One store's residences, its occupancy and its holds."""
    capacity = store.capacity_rounds()
    stored = functools.partial(trace_writer.round_stored, store_name, capacity)
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


def _connect_decoder_trace(
    trace_writer: trace_writer_module.TraceWriter,
    decoder_manager,
    pool,
) -> None:
    """The ready queue, the units' services, their memories and stages."""
    queue = decoder_manager.queue
    queue.trace.job_enqueued.connect(trace_writer.job_enqueued)
    queue.trace.depth_changed.connect(trace_writer.depth_changed)
    service = decoder_manager.service
    service.trace.job_dispatched.connect(trace_writer.job_dispatched)
    service.trace.input_landed.connect(trace_writer.input_landed)
    service.trace.job_started.connect(trace_writer.job_started)
    service.trace.job_finished.connect(trace_writer.job_finished)
    for unit in decoder_manager.pool.units():
        memory = unit.memory
        deposited = functools.partial(
            trace_writer.memory_deposited, memory.name
        )
        memory.trace.deposited.connect(deposited)
        taken = functools.partial(trace_writer.memory_taken, memory.name)
        memory.trace.taken.connect(taken)
    for decoder in _routed_decoders(pool):
        decoder.stage_recorded.connect(trace_writer.stage_recorded)


def _connect_unpinnable_observables(engine, pool) -> None:
    """Say once per model that its windows can pin no logical class.

    Compiling the model is where that is known. A confidence built from
    forced-class solves reads no gap on such a model, so without this
    line the run shows one unexplained escalation per window of it.
    """
    report = functools.partial(_log_unpinnable_observable, engine)
    for decoder in _routed_decoders(pool):
        decoder.forced_solve_unavailable.connect(report)


def _log_unpinnable_observable(engine, model, reason: str) -> None:
    """One line naming the model and why no class can be forced."""
    detector_count = len(model.detector_ids)
    engine.log(
        log_sources.DECODER_MANAGER,
        f"NO FORCED SOLVE on a {detector_count}-detector window model: "
        f"{reason}",
    )


def _routed_decoders(pool) -> list:
    """Every decoder row the router can reach, in the order the walk finds.

    A row is anything that answers the runtime-checkable Decoder port,
    which every row of DECODERS does and a row written outside decsim
    does too without inheriting decsim's base class; the port declares
    stage_recorded, so the walk asks nothing further about what a row
    has. The recursion asks the seeding protocol whether a value names
    children of its own: the routers name the tiers and the per-code
    rows, and a row that wraps another (the confidence, staged and check
    wrappers) names its inner decoder the same way.
    """
    found = []
    seen = set()
    pending = [pool.router]
    while pending:
        decoder = pending.pop()
        identity = id(decoder)
        if decoder is None or identity in seen:
            continue
        seen.add(identity)
        if isinstance(decoder, ports.Decoder):
            found.append(decoder)
        if not isinstance(decoder, seeding.RunSeedComposite):
            continue
        children = decoder.run_seed_children()
        for child in children:
            pending.append(child.child)
    return found


def _connect_window_trace(
    trace_writer: trace_writer_module.TraceWriter,
    window_manager,
    decoder_manager,
) -> None:
    """The windows a stream lays, their verdicts, commits and absorptions."""
    sources = window_manager.window_sources()
    sources.window_planned.connect(trace_writer.window_planned)
    sources.window_data_complete.connect(trace_writer.window_ready)
    sources.window_committed.connect(trace_writer.window_committed)
    sources.window_absorbed.connect(trace_writer.window_absorbed)
    decoder_manager.outcomes.trace.verdict_given.connect(
        trace_writer.verdict_given
    )
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
    decoder_manager: decoder_manager_module.DecoderManager,
    decode_records: Optional[decode_records_module.DecodeRecordLedger],
) -> None:
    """The ledger hears both terminal outcomes; the decoder runs without it."""
    if decode_records is None:
        return
    outcomes = decoder_manager.outcomes
    outcomes.trace.request_ended.connect(decode_records.request_ended)
    outcomes.trace.service_ended.connect(decode_records.service_ended)


def _connect_runtime_stamps(
    execution_runtime, stamps: runtime_stamps_module.RuntimeStamps
) -> None:
    """The stamps hear every tick of an operation's life."""
    execution_runtime.trace.operation_issued.connect(stamps.operation_issued)
    execution_runtime.trace.operation_started.connect(stamps.operation_started)
    execution_runtime.trace.body_finished.connect(stamps.body_finished)
    execution_runtime.trace.decode_released.connect(stamps.decode_released)
    execution_runtime.trace.result_returned.connect(stamps.result_returned)


def _connect_round_events(
    engine: engine_module.Engine,
    *,
    controller,
    assembler,
    held_rounds,
    transmitter,
    instruction_output,
    strong_round_store,
) -> round_events_module.RoundEventRecorder:
    """The recorder hears every round event, output and strong landing."""
    round_events = round_events_module.RoundEventRecorder(engine)
    for component in (controller, assembler, held_rounds, transmitter):
        component.trace.round_event.connect(round_events.record)
    instruction_output.trace.output_event.connect(round_events.output)
    if strong_round_store is not None:
        strong_round_store.trace.round_stored.connect(round_events.round_stored)
    return round_events


def _connect_stage_records(
    pool,
) -> stage_records_module.StageLedger:
    """The run's stage history, heard from every routed decoder."""
    stages = stage_records_module.StageLedger()
    for decoder in _routed_decoders(pool):
        decoder.stage_recorded.connect(stages.stage_recorded)
    return stages


def _connect_referee_audit(pool) -> referee_audit_module.RefereeAudit:
    """The referee's checks, heard from every routed decoder row."""
    audit = referee_audit_module.RefereeAudit()
    for decoder in _routed_decoders(pool):
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
) -> flight_recorder_module.FrameCorrections:
    """The frame's accepted and landed corrections, for the recorder."""
    corrections = flight_recorder_module.FrameCorrections()
    if pauli_frame is None:
        return corrections
    pauli_frame.trace.correction_accepted.connect(
        corrections.correction_accepted
    )
    pauli_frame.trace.correction_committed.connect(
        corrections.correction_committed
    )
    return corrections


def _decoder_utilization(
    engine: engine_module.Engine, decoder_manager
) -> metrics.DecoderUtilization:
    """The busy-unit integral, stepping at the pool's claims and returns."""
    pool = decoder_manager.pool
    units_by_pool = {}
    for name, units in pool.units_by_pool.items():
        units_by_pool[name] = len(units)
    utilization = metrics.DecoderUtilization(engine, units_by_pool)
    pool.trace.unit_busy.connect(utilization.unit_busy)
    pool.trace.unit_freed.connect(utilization.unit_freed)
    return utilization


def _decoder_memory_occupancy(
    engine: engine_module.Engine, decoder_manager
) -> metrics.DecoderMemoryOccupancy:
    """The held-round integral of every unit memory, at deposit and take."""
    units = decoder_manager.pool.units()
    capacity_by_unit = {}
    for unit in units:
        capacity_by_unit[unit.name] = unit.memory.capacity_rounds
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


def _assembled(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    log: log_writers.LogWriter,
    window_manager,
    decoder_manager,
    *,
    window_ledger: window_ledger_module.WindowLedger,
    result_ledger: result_ledger_module.ResultLedger,
    traffic_ledger: link_traffic.TrafficLedger,
    decode_records: Optional[decode_records_module.DecodeRecordLedger],
    runtime_stamps: runtime_stamps_module.RuntimeStamps,
    queue_depth: queue_depth_module.QueueDepthLog,
    controller_counters: controller_counters_module.ControllerCounters,
    command_events: command_events_module.CommandEvents,
    stages: stage_records_module.StageLedger,
    referee_audit: referee_audit_module.RefereeAudit,
    sampled_shots: sampled_shots_module.SampledShots,
    round_events,
    round_store_occupancy,
    pauli_frame,
    operations: tuple,
    trace_writer: Optional[trace_writer_module.TraceWriter],
    data_movement: Optional[data_movement_module.DataMovement],
) -> observation_module.Observation:
    """Every listener of the run: the connected ones, and the sampled ones.

    The sampled metrics the section asks for connect to action_done
    here; the rest are already connected to the sources they hear.
    """
    decode_backlog = None
    if observation.backlog_trace:
        decode_backlog = metrics.DecodeBacklog(window_manager, decoder_manager)
        engine.action_done.connect(decode_backlog.observe)
    decoder_utilization = None
    if observation.decoder_utilization:
        decoder_utilization = _decoder_utilization(engine, decoder_manager)
    decoder_memory_occupancy = None
    if observation.decoder_memory_occupancy:
        decoder_memory_occupancy = _decoder_memory_occupancy(
            engine, decoder_manager
        )
    corrections = _frame_corrections(pauli_frame)
    flight_recorder = flight_recorder_module.FlightRecorder(
        round_events,
        window_ledger,
        runtime_stamps,
        command_events,
        corrections,
        operations,
    )
    return observation_module.Observation(
        log=log,
        windows=window_ledger,
        results=result_ledger,
        traffic=traffic_ledger,
        flight_recorder=flight_recorder,
        frame_corrections=corrections,
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
        round_store_occupancy=round_store_occupancy,
    )
