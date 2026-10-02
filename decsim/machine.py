"""The machine: the parts of one run, built from its settings and connected.

A run is six parts, each a record of the components it holds, in the
order a readout travels: the qpu (the device and its readouts), the
readout path (the controller, the packing stage and the syndrome
buffers), the windows (the window manager, the verdict and the strong
re-decode), the decoders (each tier's units and their managers), the
control side (the program's execution and the Pauli frame) and the link
fabric every hop rides. Each part lives in decsim/build/, builds its own
components from their settings and wires them to one another; the
machine builds the parts and connects them, and Machine.assemble lists
every wire that crosses from one part to another.

That is the shape of gem5's standard library and of OMNeT++'s compound
modules. A gem5 board is handed a processor, a memory and a cache
hierarchy, and each wires its own insides through one method
(src/python/gem5/components/boards/abstract_board.py:426-459,
cachehierarchies/classic/private_l1_cache_hierarchy.py:97-130); an
OMNeT++ compound module declares its submodules and connections and
shows its parent only its own gates (manual section 3.4,
samples/routing/node/Node.ned). A component still meets its neighbours
only through the ports of decsim/ports.py, and a pluggable component is
still one class and one row in its package's table, sinter's
`BUILT_IN_DECODERS` (sinter/_decoding/_decoding_all_built_in_decoders.py)
made one dict per pluggable component. Where two components refer to
each other the per-job callback law breaks the cycle (SimPy's callback
on the event, simpy/core.py step()): the submitting side carries the
return path with the job, so the window side names the decoder manager
as its DecodeQueue and the strong redecode asks it to await and accept
a strong selection over the same port.

The packages import each other in one direction only, so the top can be
cut off and what is left still runs (Parnas 1972 lines 505-529;
Dijkstra's THE, dijkstra_the.txt 52-57). The levels, leaves first, are
what tools/check_uses_graph.py prints and check.sh enforces:

    0  compiled_libraries, config, plots, records, tables, trace_source
    1  engine, ports, seeding
    2  controller, detector_error_model, escalation, links, pauli_frame,
       syndrome_buffer, windows
    3  burst_detectors, decoders, qpu
    4  confidence, frontends, observe, sinter_adapters
    5  producers, settings
    6  build
    7  machine (this file)
    8  collect
    9  experiments
    10 __main__, experiment_runner, results

Level 3 and below decode a window on a store with no window manager,
which is what the decoders' own tests run.

The eleven priced hops are the line where a call stops being local
(Waldo 1994, waldo1994.txt 302-304 and 852-855): a call across a hop
has a card, a payload a record names, and a send at one end; a call
inside a unit is never priced. Across that line decsim models latency,
memory access, and partial failure only where a card's protocol names
it. The ideal row, every card's default, drops, duplicates and reorders
nothing and never retries. The credit row cuts a message into frames
that wait for a finite receive buffer's credits and loses nothing. The
reliable row loses frames at the card's bit error rate and resends them
by go-back-N until each message is delivered once and in order, so a
component above a hop never sees a loss, a duplicate or a reordering
on any row (decsim/links/fabric.py PROTOCOLS); when its retry count
runs out, the link has failed and the run stops with an error.
"""

import dataclasses
from typing import Optional

import decsim.build.control as control_part
import decsim.build.decoders as decoders_part
import decsim.build.escalation as escalation_build
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
import decsim.records.windows as window_records
import decsim.seeding as seeding
import decsim.settings as machine_settings


@dataclasses.dataclass(frozen=True)
class Machine:
    """The parts of one run, built from its settings and connected.

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
    # None on a run whose switching slot is empty
    switching: Optional[escalation_build.Switching]
    observation: observation_module.Observation

    @classmethod
    def build(
        cls, settings: machine_settings.MachineSettings, seed: Optional[int] = 0
    ) -> "Machine":
        """Build every part from the run's settings, then assemble them.

        These are the steps a script that replaces one part repeats: the
        switching part, the plan and the decoder units are compiled
        first, since the parts are built for them; then each part is
        built on its own, and assemble connects them. The seed is the
        run's root; every stochastic component derives its own from it
        and its path.
        """
        engine = engine_module.Engine()
        switching = escalation_build.build_switching(
            settings.switching, settings.weak_decoder, engine
        )
        plan = plan_build.build_plan(settings, switching)
        burst_detector = escalation_build.build_burst_detector(
            settings, engine, plan
        )
        window_tier = settings.window_tier
        escalates = settings.switching is not None
        detection_events = readout_part.build_detection_events(
            settings.detection_events,
            settings.clock,
            plan.device,
            window_tier,
            escalates,
            burst_detector,
        )
        confidence_signal = None
        if switching is not None:
            confidence_signal = switching.confidence_signal
        pool = decoders_part.build_decoder_pool(
            settings, plan, detection_events, confidence_signal
        )
        weak_store_slot, strong_store_slot = store_slots(
            settings, window_tier, pool
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
            weak_store_slot,
            strong_store_slot,
            settings.clock,
            engine,
            detection_events,
            links,
        )
        windows = windows_part.Windows.build(
            settings, engine, plan, burst_detector, links, switching
        )
        decoders = decoders_part.Decoders.build(
            settings.decoder_manager, engine, pool
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
            switching,
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
        switching: Optional[escalation_build.Switching] = None,
        seed: Optional[int] = 0,
    ) -> "Machine":
        """Connect the parts, start them, seed them, and load the program.

        The connections are every wire that crosses from one part to
        another, in the order a readout travels; each part wired its own
        inside when it was built, and a switching part binds its policy
        last. Nothing is scheduled until the machine
        runs, so the order the parts were built in cannot move a tick
        (gem5 src/sim/sim_object.hh lines 194 and 280).
        """
        root_seed = _root_seed(seed)
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
        windows.connect(
            weak_store=readout.weak_syndrome_buffer,
            strong_store=readout.strong_syndrome_buffer,
            store_output=readout.primary_output,
            strong_output=readout.strong_output,
            strong_receiver=readout.strong_syndrome_round_receiver,
            primary_decoder=decoders.primary_decoder,
            strong_decoder=decoders.strong_decoder,
            decode_queue=decoders.decoder_manager,
            strong_decode_queue=decoders.strong_decoder_manager,
            input_fold=input_fold,
            frame=control.pauli_frame,
            conditional_release=control.conditional_release,
            factory=qpu.factory,
            detection_events=readout.detection_events,
        )
        control.connect(
            qpu=qpu.device,
            factory=qpu.factory,
            windows=windows.window_manager,
            decode_queue=decoders.decoder_manager,
        )
        if switching is not None:
            switching.connect(windows, decoders)
        # the managers hear their rows before the planner compiles a
        # model, and the sender asks the wired window side where it reads
        decoders.start()
        windows.start()
        readout.start()
        # the window side's one stochastic owner is the switching policy
        parts = (qpu, control, readout, decoders)
        seed_roots = _seed_roots(plan, links, parts, switching)
        seeding.bind_run_seed(root_seed, seed_roots)
        observation = _observe(
            settings,
            engine,
            links,
            qpu=qpu,
            control=control,
            readout=readout,
            windows=windows,
            decoders=decoders,
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
            switching=switching,
            observation=observation,
        )

    def start(self) -> None:
        """Queue the first event of every component that has one.

        Every component is built and wired first and nothing is
        scheduled while the graph is still being assembled; that is
        gem5's split between the constructor and startup, "the
        appropriate place to schedule initial event(s)"
        (gem5 src/sim/sim_object.hh lines 194 and 280).
        """
        self.qpu.factory.start()
        self.control.execution_runtime.start()

    def run(self) -> result_records.RunResult:
        """Start every component, run to quiescence, read the result."""
        self.start()
        self.engine.run()
        self.windows.check_settled()
        self.decoders.check_settled()
        self.readout.check_settled()
        return _capture_result(self)


def build_links(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> ports.Link:
    """The link fabric, on the run's links card."""
    return fabric.LinkFabric(settings.links, engine)


def store_slots(
    settings: machine_settings.MachineSettings,
    window_tier: window_records.DecoderTier,
    pool: decoders_part.DecoderPool,
) -> tuple:
    """The weak and the strong syndrome buffer slots; None for an unread one.

    The tier that decodes the plan's windows reads its own store through
    the chip's pool, and a switching run's strong tier reads the strong
    store through the host's; each pool says whether its units read the
    rounds in place.
    """
    chip_reads_in_place = not pool.chip.copies_input
    if window_tier is window_records.DecoderTier.STRONG:
        strong_store_slot = readout_part.StoreSlot(
            settings.strong_syndrome_buffer, chip_reads_in_place
        )
        return None, strong_store_slot
    weak_store_slot = readout_part.StoreSlot(
        settings.weak_syndrome_buffer, chip_reads_in_place
    )
    if pool.host is None:
        return weak_store_slot, None
    host_reads_in_place = not pool.host.copies_input
    strong_store_slot = readout_part.StoreSlot(
        settings.strong_syndrome_buffer, host_reads_in_place
    )
    return weak_store_slot, strong_store_slot


def _seed_roots(
    plan: plan_build.Plan,
    links: ports.Link,
    parts: tuple,
    switching: Optional[escalation_build.Switching],
) -> tuple:
    """The seed path of every stochastic owner; the names are results.

    The plan's own rows first, then each part's, then the switching
    policy, None on a run with no switching. The seeding sorts the roots
    by path, so the order they are listed in moves nothing.
    """
    escalation_policy = None
    if switching is not None:
        escalation_policy = switching.policy
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
    owners.append(("escalation_policy", escalation_policy))
    roots = []
    for name, owner in owners:
        path = (seed_records.RunSeedPathSegment("field", name),)
        roots.append((path, owner))
    return tuple(roots)


def _observe(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    links: ports.Link,
    *,
    qpu: qpu_part.Qpu,
    control: control_part.Control,
    readout: readout_part.Readout,
    windows: windows_part.Windows,
    decoders: decoders_part.Decoders,
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
        process_name=name,
        traffic_ledger=traffic_ledger,
    )


def _process_name(
    settings: machine_settings.MachineSettings, seed: Optional[int]
) -> str:
    """The machine the trace is of: its escalation, code distance and seed.

    The escalation is the yaml's word for the filled decode slots
    (MachineSettings.escalation_kind). The machine knows no sweep, so the
    point's other values name the trace's file (experiments/measure.py
    shot_label) and not this line.
    """
    kind = settings.escalation_kind
    distance = settings.qpu.distance
    return f"decsim {kind} d{distance} seed{seed}"


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
