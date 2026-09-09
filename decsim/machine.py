"""The root: one object that builds every component and wires them.

A Machine is gem5's shape (src/python/m5/SimObject.py: a SimObject's
Python class is its params, `allClasses` maps a name to a class; the
learning_gem5 simple.py script names each component once and assigns
its ports), with the wiring done by constructor because Python needs
no second bind step. A MachineSettings (decsim/settings.py) holds one
settings record per yaml section, each built by its own package's
`from_yaml`, and each package's own settings module maps that section's
`kind` to the class that fills the port, sinter's `BUILT_IN_DECODERS`
(sinter/_decoding/_decoding_all_built_in_decoders.py) made one dict per
pluggable part. A new component is one class that fills its port in
decsim/ports.py and one row in its package's table.

Every component is wired by constructor; nothing is bound to a
component after it is built. Where two components refer to each other
the per-job callback law breaks the cycle (SimPy's callback on the
event, simpy/core.py step()): the submitting side carries the return
path with the job, so the decoder manager is built first and the
window side takes it as its DecodeQueue; the strong redecode asks it
to await and accept a strong selection over the same port. The
controller's issuer carries the start callback with the issue, the
QPU's completion receiver is the runtime's body_done, and the runtime
tells the issuer what it knows at each release. The stores' callbacks
arrive by constructor: the held rounds are built first, each store
retries them when a slot frees, and the strong writer tells the window
manager what landed. The one stand-in is _LateWiring inside
_window_manager, for the courier's and the committer's callbacks to the
facade and the strong redecode built after them.

The packages import each other in one direction only, so the top can be
cut off and what is left still runs (Parnas 1972 lines 505-529;
Dijkstra's THE, dijkstra_the.txt 52-57). The levels, leaves first, are
what tools/check_uses_graph.py prints and check.sh enforces:

    0  config, records, tables, trace_source
    1  detector_error_model, engine, pauli_frame, ports, seeding,
       syndrome_buffer
    2  confidence, controller, decoders, escalation, links, qpu, windows
    3  frontends, observe
    4  settings
    5  build
    6  machine (this file)
    7  collect
    8  front
    9  __main__

Level 2 and below decode a window on a store with no window manager,
which is what the decoders' own tests run.

The eleven priced hops are the line where a call stops being local
(Waldo 1994, waldo1994.txt 302-304 and 852-855): a call across a hop
has a card, a payload a record names, and a send at one end; a call
inside a unit is never priced. Across that line decsim models latency
and memory access and no partial failure at all: no hop drops,
duplicates or reorders what it carries, and nothing retries. That is a
stated scope, not an omission, and a retry added to a hop as a tuning
knob would be a modeling change, not a parameter.
"""

import dataclasses
from typing import Any, Optional

import decsim.build.controller_side as controller_side
import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.listeners as listener_build
import decsim.build.plan as plan_build
import decsim.build.stores as store_build
import decsim.build.window_side as window_side
import decsim.controller.conditional_release as conditional_release_module
import decsim.controller.controller as controller_module
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.round_writes as round_writes
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.links.fabric as fabric
import decsim.links.link_profiles as link_profiles
import decsim.observe.link_traffic as link_traffic
import decsim.observe.observation as observation_module
import decsim.observe.wiring as wiring
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.records.results as result_records
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer_module
import decsim.tables as tables
import decsim.windows.window_manager as window_manager_module


@dataclasses.dataclass(frozen=True)
class Machine:
    """Every component of one run, built from its settings and wired.

    Build one per seed with Machine.build, then run it once; the
    components stay readable afterwards for the measurements and the
    views that read them. active_decoder is the unit of the tier that
    decodes the plan's windows, None when no decoder was named.
    """

    settings: machine_settings.MachineSettings
    engine: engine_module.Engine
    observation: observation_module.Observation
    links: fabric.LinkFabric
    conditional_release: conditional_release_module.ConditionalRelease
    round_store: round_store_module.RoundStore
    strong_round_store: Optional[round_store_module.RoundStore]
    strong_round_writer: Optional[strong_round_writer_module.StrongRoundWriter]
    pauli_frame: Optional[pauli_frame_module.PauliFrame]
    window_manager: window_manager_module.WindowManager
    assembler: round_assembly.RoundAssembler
    round_writer: round_writes.RoundWriter
    transmitter: round_transmission.RoundTransmitter
    decoder_manager: decoder_manager_module.DecoderManager
    active_decoder: Optional[Any]
    factory: Any
    syndrome_source: Any
    qpu: cycle_clock.QPUDevice
    controller: controller_module.Controller
    issuer: operation_issue.OperationIssuer
    instruction_output: instruction_output_module.InstructionOutput
    idle_rounds: idle_rounds_module.IdleRoundAccounting
    execution_runtime: execution_runtime_module.ExecutionRuntime
    operations: tuple

    @classmethod
    def build(
        cls, settings: machine_settings.MachineSettings, seed: Optional[int] = 0
    ) -> "Machine":
        """Build every component in pipeline order and wire it.

        The order is the one a readout travels, so a reader follows the
        wiring by reading down. The seed is the run's root; every
        stochastic component derives its own from it and its path.
        """
        root_seed = _root_seed(seed)
        observation = settings.observation
        engine = engine_module.Engine()
        escalation_policy = escalation_build.build_escalation_policy(
            settings.escalation
        )
        plan = plan_build.build_plan(settings, escalation_policy)
        pool = decoder_build.build_decoder_pool(
            settings, plan, escalation_policy
        )
        store_build.check_readout_cost_is_priced(settings)
        conditional_release = conditional_release_module.ConditionalRelease(
            engine
        )
        traffic_ledger = link_traffic.TrafficLedger(settings.links)
        fabric_row = tables.row(
            link_profiles.LINK_FABRICS, "links.kind", settings.links.kind
        )
        links = fabric_row.build(settings.links, engine)
        held_rounds = round_writes.HeldRounds(
            engine, settings.controller.packing_overflow
        )
        round_store = store_build.build_round_store(
            settings.round_store, held_rounds
        )
        strong_round_store = store_build.build_strong_round_store(
            settings.strong_round_store, escalation_policy, held_rounds
        )
        weak_output, strong_output = store_build.build_store_outputs(
            engine, links, round_store, strong_round_store
        )
        pauli_frame = store_build.build_pauli_frame(
            settings.pauli_frame, engine
        )
        # The decoder manager is built first, so the requester and the
        # strong redecode take the decode queue by constructor and every
        # job carries its return path.
        controller_side.check_strong_route(escalation_policy, pool.router)
        decoder_manager = controller_side.build_decoder_manager(
            engine, settings, escalation_policy, pool
        )
        window_manager = window_side.build_window_manager(
            engine,
            settings,
            escalation_policy,
            plan,
            links=links,
            conditional_release=conditional_release,
            fault_model_requirement_for=pool.router.fault_model_requirement_for,
            round_store=round_store,
            strong_round_store=strong_round_store,
            weak_output=weak_output,
            strong_output=strong_output,
            pauli_frame=pauli_frame,
            decode_queue=decoder_manager,
            on_workload_complete=lambda: factory.shutdown(),
        )
        strong_round_writer = store_build.build_strong_round_writer(
            engine, links, strong_round_store, window_manager
        )
        transmitter = round_transmission.RoundTransmitter(
            engine, links, window_manager, weak_output
        )
        publishes_from_strong_store = not window_manager.reads_windows_from(
            round_store
        )
        round_writer = round_writes.RoundWriter(
            engine,
            round_store,
            strong_round_writer,
            publishes_from_strong_store=publishes_from_strong_store,
            held_rounds=held_rounds,
            transmitter=transmitter,
        )
        detection_events = controller_side.build_detection_events(
            settings, plan.device
        )
        rounds_in_flight = round_assembly.RoundsInFlight(
            settings.controller.packing_rounds_in_flight,
            held_rounds,
            transmitter,
        )
        assembler = round_assembly.RoundAssembler(
            engine,
            settings.controller,
            detection_events=detection_events,
            on_packed=round_writer.admit,
            rounds_in_flight=rounds_in_flight,
        )
        factory = controller_side.build_factory(
            settings.magic_state_factory, engine, decoder_manager, plan
        )
        qpu = cycle_clock.QPUDevice(engine, plan.device, plan.round_ticks)
        pulse_ticks = settings.controller.decision_to_pulse_ticks()
        instruction_output = instruction_output_module.InstructionOutput(
            engine, links, qpu, pulse_ticks
        )
        streams = controller_side.build_feedback_streams(
            engine,
            plan,
            qpu,
            window_manager,
            retry_ready_operations=lambda: (
                execution_runtime.retry_ready_operations()
            ),
        )
        patch_by_identity = controller_side.resolved_patches_by_identity(plan)
        idle_rounds = idle_rounds_module.IdleRoundAccounting(
            plan.idle_policy, decoder_manager, patch_by_identity, streams, qpu
        )
        issuer = operation_issue.OperationIssuer(
            engine,
            streams,
            idle_rounds,
            window_manager,
            plan.run_plan.resolved_operations,
            instruction_output,
        )
        controller = controller_module.Controller(
            engine, links, settings.controller, assembler
        )
        execution_runtime = execution_runtime_module.ExecutionRuntime(
            engine,
            issuer=issuer,
            factory=factory,
            resource_claims_by_operation_id=plan.resource_claims,
        )
        # The QPU's receivers arrive after the controller is built.
        qpu.connect_readout_receiver(controller)
        qpu.connect_completion_receiver(execution_runtime.body_done)
        qpu.connect_idle_receiver(idle_rounds.emit_idle_round)
        input_transport = decoder_manager.input_transport()
        seed_roots = listener_build.build_seed_roots(
            code=plan.code,
            scheme=plan.scheme,
            device=plan.device,
            error_model_provider=plan.error_model_provider,
            decoder_router=pool.router,
            factory=factory,
            escalation_policy=escalation_policy,
            scheduler=pool.scheduler,
            decoder_memory_transfer=input_transport,
            boundary_policy=plan.boundary_policy,
            window_interaction=plan.window_interaction,
            idle_policy=plan.idle_policy,
            conditional_release=conditional_release,
            # the packing stage is four components now; its seed path
            # segment is a result (seeding hashes the segment names) and
            # stays, as memory_model's does
            syndrome_packing=None,
            controller=controller,
            qpu=qpu,
            execution_runtime=execution_runtime,
            pauli_frame=pauli_frame,
            # the retained-storage observer is gone; its seed path segment
            # is a result (seeding hashes the segment names) and stays
            memory_model=None,
        )
        seeding.bind_run_seed(root_seed, seed_roots)
        conditional_release.connect(
            instruction_output, execution_runtime.on_decision
        )
        process_name = controller_side.process_name(settings, seed)
        listeners = wiring.observe(
            observation,
            engine,
            process_name=process_name,
            operations=plan.operations,
            traffic_ledger=traffic_ledger,
            links=links,
            qpu=qpu,
            syndrome_source=plan.device,
            controller=controller,
            idle_rounds=idle_rounds,
            assembler=assembler,
            held_rounds=held_rounds,
            round_writer=round_writer,
            transmitter=transmitter,
            instruction_output=instruction_output,
            round_store=round_store,
            strong_round_store=strong_round_store,
            strong_round_writer=strong_round_writer,
            execution_runtime=execution_runtime,
            pauli_frame=pauli_frame,
            decoder_manager=decoder_manager,
            window_manager=window_manager,
            pool=pool,
        )
        listener_build.load_program(
            plan,
            conditional_release,
            window_manager,
            streams,
            idle_rounds,
            execution_runtime,
        )
        return cls(
            settings=settings,
            engine=engine,
            observation=listeners,
            links=links,
            conditional_release=conditional_release,
            round_store=round_store,
            strong_round_store=strong_round_store,
            strong_round_writer=strong_round_writer,
            pauli_frame=pauli_frame,
            window_manager=window_manager,
            assembler=assembler,
            round_writer=round_writer,
            transmitter=transmitter,
            decoder_manager=decoder_manager,
            active_decoder=pool.active,
            factory=factory,
            syndrome_source=plan.device,
            qpu=qpu,
            controller=controller,
            issuer=issuer,
            instruction_output=instruction_output,
            idle_rounds=idle_rounds,
            execution_runtime=execution_runtime,
            operations=plan.all_operations,
        )

    def run(self) -> result_records.RunResult:
        """Run the engine to quiescence, check settlement, read the result."""
        self.engine.run()
        strong_redecode = self.window_manager.strong_redecode
        if strong_redecode is not None and strong_redecode.has_pending():
            pending = strong_redecode.pending_work()
            raise RuntimeError(
                f"the run ended with pending strong escalations: {pending}"
            )
        self.decoder_manager.check_decode_work_settled()
        self.window_manager.check_settled()
        self.assembler.check_settled()
        self.round_writer.check_settled()
        self.transmitter.check_settled()
        if self.strong_round_writer is not None:
            self.strong_round_writer.check_settled()
        return _capture_result(self)


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
    execution_runtime = machine.execution_runtime
    if not engine.idle or not execution_runtime.workload_complete:
        raise RuntimeError("primary run ended before workload completed")
    truth_for = machine.syndrome_source.logical_observable_truth
    operation_by_id = {}
    for operation in machine.operations:
        operation_by_id[operation.id] = operation
    rows = []
    for operation_id in sorted(operation_by_id):
        operation = operation_by_id[operation_id]
        row = _operation_result(machine, operation, truth_for)
        rows.append(row)
    link_traffic = machine.observation.traffic.traffic_json_value()
    data_movement = _data_movement_value(machine.observation)
    return result_records.RunResult(
        terminal_status="complete",
        event_queue_empty=True,
        decode_work_settled=True,
        execution_workload_complete=True,
        execution_done_ticks=machine.observation.runtime_stamps.last_finish,
        fully_done_ticks=engine.now,
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
    actual = truth_for(operation_id)
    if actual is not None:
        actual = tuple(actual)
    failure = None
    if bits is not None and actual is not None:
        if len(bits) != len(actual):
            raise RuntimeError(
                f"operation {operation_id} predicted {len(bits)} logical "
                f"observables but the syndrome source sampled {len(actual)}"
            )
        failure = bits != actual
    binding = machine.issuer.stream_binding_for(operation_id)
    stream_offset = operation.stream_offset
    if binding is not None:
        stream_offset = binding.stream_offset
    return result_records.LogicalOperationResult(
        operation_id, status, bits, stream_offset, actual, failure
    )
