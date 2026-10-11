"""The machine: the parts of one run, built from its settings and connected.

A run is six parts in decsim/build/, in the order a readout travels:
the qpu, the readout path (the controller, the packing stage and the
syndrome buffers), the windows, the decoders, the control side (the
program's execution and the Pauli frame) and the link fabric every hop
rides. Each part builds and wires its own components; Machine.assemble
lists every wire that crosses from one part to another. That is gem5's
standard library, where a board is handed a processor, a memory and a
cache hierarchy that each wire their own insides
(src/python/gem5/components/boards/abstract_board.py:426-459), and
OMNeT++'s compound modules (manual section 3.4). Where two components
refer to each other, the submitting side carries the return path with
the job (SimPy's callback on the event, simpy/core.py step()).

The packages import each other in one direction only, so the top can be
cut off and what is left still runs (Parnas 1972 lines 505-529;
Dijkstra's THE, dijkstra_the.txt 52-57). The levels, leaves first, are
what tools/check_uses_graph.py prints and check.sh enforces:

    0  compiled_libraries, config, records, trace_source
    1  engine, seeding
    2  ports
    3  controller, detector_error_model, links, pauli_frame,
       syndrome_buffer, windows
    4  decoders, qpu
    5  confidence, escalation, frontends, sinter_adapters
    6  build, producers
    7  observe
    8  settings
    9  machine (this file)
    10 experiments
    11 __main__

The twelve priced hops are the line where a call stops being local
(Waldo 1994, waldo1994.txt 302-304 and 852-855): a call across a hop
has a card, a payload a record names, and a send at one end; a call
inside a unit is never priced. A component above a hop never sees a
loss, a duplicate or a reordering on any protocol
(decsim/links/credit_channel.py and reliable_channel.py); when a
reliable link's retry count runs out, the run stops with an error.
"""

import dataclasses
from typing import Optional

import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.escalation as escalation_build
import decsim.build.layout as layout_module
import decsim.build.plan as plan_build
import decsim.build.program as program_build
import decsim.build.qpu as qpu_part
import decsim.build.readout as readout_part
import decsim.build.windows as windows_part
import decsim.engine as engine_module
import decsim.links.fabric as fabric
import decsim.observe.link_traffic as link_traffic
import decsim.observe.observation as observation_module
import decsim.observe.wiring as wiring
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.results as result_records
import decsim.records.seeds as seed_records
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.statistics.registry as statistics_registry
import decsim.windows.built_window_models as built_window_models
import decsim.windows.routers as routers


@dataclasses.dataclass(frozen=True)
class Machine:
    """The wired parts of one run, built from its settings.

    Build one per seed with Machine.build, then run it once; the parts
    stay readable afterwards for the measurements and the views that
    read them.
    """

    settings: machine_settings.MachineSettings
    engine: engine_module.Engine
    plan: plan_build.Plan
    links: ports.Link
    qpu: qpu_part.Qpu
    control: control_part.Control
    readout: readout_part.Readout
    windows: windows_part.Windows
    decoders: decoders_part.Decoders
    # one per escalate edge, in graph order; none on a run with no edge
    switching_parts: tuple
    observation: observation_module.Observation
    layout: layout_module.Layout

    @property
    def switching(self) -> Optional[escalation_build.Switching]:
        """The entry edge's switching part; None on a run with no edge."""
        if not self.switching_parts:
            return None
        return self.switching_parts[0]

    @classmethod
    def build(
        cls,
        settings: machine_settings.MachineSettings,
        seed: Optional[int] = 0,
        built_models: Optional[built_window_models.BuiltWindowModels] = None,
        online_threshold: Optional[ports.ThresholdSource] = None,
        layout: Optional[layout_module.Layout] = None,
    ) -> "Machine":
        """Build every part from the run's settings, then assemble them.

        The seed is the run's root; every stochastic component derives
        its own from it and its path. built_models is the window error
        model cache the shots of one unit share (collect.run_unit), and
        online_threshold the task's calibrator when its threshold
        learns across shots (collect.Task); both are task state, never
        settings. layout is a new-shape graph's (collect.Task); None
        lays out the settings' slots under their own names.
        """
        settings = settings.at_task()
        if built_models is None:
            built_models = built_window_models.BuiltWindowModels()
        if layout is None:
            layout = layout_module.legacy_layout(settings)
        engine = engine_module.Engine()
        switching_parts = escalation_build.build_switching(
            layout, engine, online_threshold, settings.clock
        )
        plan = plan_build.build_plan(
            settings.qpu,
            settings.workload,
            settings.windows,
            settings.idle_policy,
            settings.detection_events,
            settings.switching,
            settings.decoder_manager.bulk_strong,
            switching_parts,
        )
        formation_paths = layout.formation_paths()
        detection_events = readout_part.build_detection_events(
            settings.detection_events,
            settings.clock,
            plan.device,
            formation_paths,
        )
        signal_by_pool = escalation_build.signal_by_pool(switching_parts)
        pools = decoders_part.build_pool_units(
            layout, settings.clock, detection_events, signal_by_pool
        )
        links = build_links(settings, engine)
        qpu = qpu_part.Qpu.build(settings.magic_state_factory, engine, plan)
        control = control_part.Control.build(
            settings.controller,
            settings.pauli_frame,
            settings.clock,
            engine,
            plan,
            links,
        )
        readout = readout_part.Readout.build(
            settings.controller,
            settings.links,
            layout.stores,
            settings.clock,
            engine,
            detection_events,
            links,
        )
        windows = windows_part.Windows.build(
            settings.windows,
            settings.workload,
            layout,
            settings.clock,
            engine,
            plan,
            links,
            built_models,
        )
        target_by_source = layout.target_by_source()
        decoders = decoders_part.Decoders.build(
            settings.decoder_manager,
            settings.clock,
            engine,
            pools,
            target_by_source,
        )
        return cls.assemble(
            settings,
            engine,
            plan,
            links,
            qpu,
            control,
            readout,
            windows,
            decoders,
            layout,
            switching_parts,
            seed,
        )

    @classmethod
    def assemble(
        cls,
        settings: machine_settings.MachineSettings,
        engine: engine_module.Engine,
        plan: plan_build.Plan,
        links: ports.Link,
        qpu: qpu_part.Qpu,
        control: control_part.Control,
        readout: readout_part.Readout,
        windows: windows_part.Windows,
        decoders: decoders_part.Decoders,
        layout: layout_module.Layout,
        switching_parts: tuple = (),
        seed: Optional[int] = 0,
    ) -> "Machine":
        """Connect the parts, start them, seed them, and load the program.

        Nothing is scheduled until the machine runs, so the order the
        parts were built in cannot move a tick (gem5
        src/sim/sim_object.hh lines 194 and 280).
        """
        root_seed = _root_seed(seed)
        edges = escalation_build.escalate_edges_of(switching_parts)
        weak_name, strong_name = layout.entry_edge_stores()
        weak_store = readout.store_named(weak_name)
        strong_store = readout.store_named(strong_name)
        qpu.connect(
            readout_receiver=readout.controller,
            runtime=control.execution_runtime,
            idle_rounds=control.idle_rounds,
            decode_queue=decoders.decoder_manager,
        )
        readout.connect(
            windows=windows.window_manager,
            retention=windows.retention,
        )
        input_fold = decoders.decoder_manager.input_fold()
        routed_sides = _routed_sides(layout, readout, decoders)
        windows.connect(
            weak_store=weak_store,
            strong_store=strong_store,
            store_output=readout.primary_output,
            primary_decoder=decoders.primary_decoder,
            target_decoders=decoders.target_decoders,
            decode_queue=decoders.decoder_manager,
            edges=edges,
            input_fold=input_fold,
            frame=control.pauli_frame,
            conditional_release=control.conditional_release,
            factory=qpu.factory,
            detection_events=readout.detection_events,
            routed_sides=routed_sides,
        )
        control.connect(
            qpu=qpu.device,
            factory=qpu.factory,
            windows=windows.window_manager,
            decode_queue=decoders.decoder_manager,
        )
        for part in switching_parts:
            part.connect(plan, readout, windows, decoders)
        escalation_build.join_the_chain(switching_parts)
        entry_switching = None
        if switching_parts:
            entry_switching = switching_parts[0]
            # a window's solves join on the entry edge's signal alone
            windows.gap_join.signal = entry_switching.confidence_signal
        # the managers hear their rows before the planner compiles a
        # model, and the sender asks the wired window side where it reads
        decoders.start()
        windows.start()
        # the window side's stochastic owners are the edges' policies
        parts = (qpu, control, readout, decoders, *switching_parts)
        seed_roots = _seed_roots(plan, links, parts)
        seeding.bind_run_seed(root_seed, seed_roots)
        run_statistics = statistics_registry.Registry(engine)
        decoders.attach_statistics(run_statistics)
        statistics_parts = (control, readout, *switching_parts)
        _attach_statistics(run_statistics, statistics_parts)
        observation = _observe(
            settings,
            engine,
            links,
            run_statistics,
            qpu=qpu,
            control=control,
            readout=readout,
            windows=windows,
            decoders=decoders,
            switching=entry_switching,
            seed=seed,
        )
        program_build.load_program(plan, control, windows)
        return cls(
            settings=settings,
            engine=engine,
            plan=plan,
            links=links,
            qpu=qpu,
            control=control,
            readout=readout,
            windows=windows,
            decoders=decoders,
            switching_parts=switching_parts,
            observation=observation,
            layout=layout,
        )

    def start(self) -> None:
        """Queue the first event of every component that has one.

        gem5's startup, "the appropriate place to schedule initial
        event(s)" (gem5 src/sim/sim_object.hh lines 194 and 280).
        """
        self.qpu.factory.start()
        self.control.execution_runtime.start()

    def run(self) -> result_records.RunResult:
        """Start every component, run to quiescence, read the result."""
        self.start()
        self.engine.run()
        self.observation.statistics.finish()
        for part in self.switching_parts:
            part.check_settled()
        self.windows.check_settled()
        self.decoders.check_settled()
        self.readout.check_settled()
        return _capture_result(self)


def build_links(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> ports.Link:
    """The link fabric, on the run's links card."""
    return fabric.LinkFabric(settings.links, engine)


def _seed_roots(
    plan: plan_build.Plan, links: ports.Link, parts: tuple
) -> tuple:
    """The seed path of every stochastic owner; the names are results.

    The seeding sorts the roots by path, so the order here moves nothing.
    """
    owners = [
        ("code", plan.code),
        ("scheme", plan.scheme),
        ("device", plan.device),
        ("error_model_provider", plan.error_model_provider),
        ("boundary_policy", plan.boundary_policy),
        ("window_interaction", plan.window_interaction),
        ("idle_policy", plan.idle_policy),
        ("links", links),
    ]
    for part in parts:
        part_roots = part.seed_roots()
        owners.extend(part_roots)
    roots = []
    for name, owner in owners:
        path = (seed_records.RunSeedPathSegment("field", name),)
        roots.append((path, owner))
    return tuple(roots)


def _attach_statistics(
    run_statistics: statistics_registry.Registry, parts: tuple
) -> None:
    """Every component each built part names, at the path it names.

    gem5 binds each SimObject's stats into the hierarchy after the
    objects exist (src/python/m5/stats/__init__.py:277-309); a part not
    built (None) names none.
    """
    for part in parts:
        if part is None:
            continue
        for path, component in part.statistics_parts():
            run_statistics.attach(component, path)


def _observe(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    links: ports.Link,
    run_statistics: statistics_registry.Registry,
    *,
    qpu: qpu_part.Qpu,
    control: control_part.Control,
    readout: readout_part.Readout,
    windows: windows_part.Windows,
    decoders: decoders_part.Decoders,
    switching: Optional[escalation_build.Switching],
    seed: Optional[int],
) -> observation_module.Observation:
    """Connect the run's listeners, before the workload is loaded."""
    traffic_ledger = link_traffic.TrafficLedger(settings.links)
    name = _process_name(settings, seed)
    return wiring.observe(
        settings.observation,
        engine,
        links=links,
        qpu=qpu,
        control=control,
        readout=readout,
        windows=windows,
        decoders=decoders,
        switching=switching,
        process_name=name,
        traffic_ledger=traffic_ledger,
        run_statistics=run_statistics,
    )


def _process_name(
    settings: machine_settings.MachineSettings, seed: Optional[int]
) -> str:
    """The machine the trace is of: its escalation, code distance and seed."""
    kind = settings.escalation_kind
    distance = settings.qpu.distance
    return f"decsim {kind} d{distance} seed{seed}"


def _routed_sides(
    layout: layout_module.Layout,
    readout: readout_part.Readout,
    decoders: decoders_part.Decoders,
) -> tuple:
    """Each entry pool past the first: its manager and its store's end.

    The entry store sends to each over a hop of its own (layout
    StoreReader); none of them has an edge leaving it yet, so none has
    a join.
    """
    entry_pools = layout.entry_pools
    if len(entry_pools) == 1:
        return ()
    entry_store = readout.stores[0]
    sides = []
    for entry in entry_pools[1:]:
        pool = entry.pool
        manager = decoders.manager_of(pool)
        store_output = entry_store.output_for(pool)
        side = routers.EntrySide(pool, manager, store_output)
        sides.append(side)
    return tuple(sides)


def _root_seed(value) -> Optional[int]:
    """The run's root seed: None for entropy, else an unsigned 64-bit int."""
    if value is None:
        return None
    root_seed = int(value)
    if not 0 <= root_seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return root_seed


def _capture_result(machine: Machine) -> result_records.RunResult:
    """Project the finished components into the immutable run result."""
    engine = machine.engine
    execution_runtime = machine.control.execution_runtime
    if not engine.idle or not execution_runtime.workload_complete:
        raise RuntimeError("primary run ended before workload completed")
    truth_for = machine.qpu.syndrome_source.logical_observable_truth
    operation_by_id = {}
    for operation in machine.plan.all_operations:
        operation_by_id[operation.id] = operation
    ordered_ids = sorted(
        operation_by_id, key=identity_records.stable_identity_bytes
    )
    rows = []
    for operation_id in ordered_ids:
        operation = operation_by_id[operation_id]
        row = _operation_result(machine, operation, truth_for)
        rows.append(row)
    link_traffic = machine.observation.traffic.traffic_json_value()
    data_movement = _data_movement_value(machine.observation)
    stamps = machine.observation.runtime_stamps
    waits = stamps.magic_state_wait.values()
    magic_state_stall_ticks = sum(waits)
    return result_records.RunResult(
        terminal_status="complete",
        event_queue_empty=True,
        decode_work_settled=True,
        execution_workload_complete=True,
        execution_done_ticks=stamps.last_finish,
        fully_done_ticks=engine.now,
        magic_state_stall_ticks=magic_state_stall_ticks,
        operation_results=tuple(rows),
        link_traffic=link_traffic,
        data_movement=data_movement,
    )


def _data_movement_value(
    observation: observation_module.Observation,
) -> Optional[dict]:
    """The data-path counters the result carries, when they were built."""
    counters = observation.data_movement
    if counters is None:
        return None
    return counters.json_value()


def _operation_result(
    machine: Machine, operation, truth_for
) -> result_records.LogicalOperationResult:
    """One operation's predicted observables beside the sampled truth."""
    operation_id = operation.id
    logical = machine.observation.results.observables_for(operation_id)
    bits = None
    status = "no_logical_output"
    if logical is not None:
        bits = tuple(logical)
        status = "logical_observables"
    binding = machine.control.issuer.stream_binding_for(operation_id)
    actual = _operation_truth(operation, binding, truth_for)
    if actual is not None:
        actual = tuple(actual)
    failure = _logical_failure(operation_id, bits, actual)
    stream_offset = operation.stream_offset
    if binding is not None:
        stream_offset = binding.stream_offset
    return result_records.LogicalOperationResult(
        operation_id, status, bits, stream_offset, actual, failure
    )


def _operation_truth(operation, binding, truth_for):
    """A segment contribution has no independently sampled logical truth.

    Sinter's _CompiledStimThenDecodeSampler.sample compares predictions
    and actual_obs for the same complete circuit shot. A stream segment
    contributes only its commit interval, so only its owner is scored.
    """
    stream_id = operation.stream_id
    if binding is not None:
        stream_id = binding.stream_id
    if stream_id is not None and stream_id != operation.id:
        return None
    return truth_for(operation.id)


def _logical_failure(operation_id, prediction, truth) -> Optional[bool]:
    if prediction is None or truth is None:
        return None
    if len(prediction) != len(truth):
        raise RuntimeError(
            f"operation {operation_id} predicted {len(prediction)} logical "
            f"observables but the syndrome source sampled {len(truth)}"
        )
    return prediction != truth
