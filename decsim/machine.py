"""The root: one object that builds every component and wires them.

A Machine is gem5's shape (src/python/m5/SimObject.py: a SimObject's
Python class is its params, `allClasses` maps a name to a class; the
learning_gem5 simple.py script names each component once and assigns
its ports). A MachineSettings (decsim/settings.py) holds one
settings record per yaml section, each built by its own package's
`from_yaml`, and each package's own settings module maps that section's
`kind` to the class that fills the port, sinter's `BUILT_IN_DECODERS`
(sinter/_decoding/_decoding_all_built_in_decoders.py) made one dict per
pluggable part. A new component is one class that fills its port in
decsim/ports.py and one row in its package's table.

What the machine is made of and what is wired to what is one file of
two tables, decsim/assembly.py; this file reads them. The root compiles
the run's settings into the plan, the decoder pool and the escalation
policy, builds every seat from its settings, binds every wire by
assignment, which is gem5's script binding one port to another
(configs/learning_gem5/part1/simple.py:68), and only then lets a seat do
work. Where two components refer to each other the per-job callback law
breaks the cycle (SimPy's callback on the event, simpy/core.py step()):
the submitting side carries the return path with the job, so the window
side names the decoder manager as its DecodeQueue and the strong
redecode asks it to await and accept a strong selection over the same
port.

The packages import each other in one direction only, so the top can be
cut off and what is left still runs (Parnas 1972 lines 505-529;
Dijkstra's THE, dijkstra_the.txt 52-57). The levels, leaves first, are
what tools/check_uses_graph.py prints and check.sh enforces:

    0  config, records, tables, trace_source
    1  engine, ports, seeding
    2  detector_error_model, escalation, links, pauli_frame,
       syndrome_buffer, windows
    3  controller, decoders, qpu
    4  confidence, frontends, observe
    5  settings
    6  build
    7  assembly
    8  machine (this file)
    9  collect
    10 experiments
    11 __main__

Level 3 and below decode a window on a store with no window manager,
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

import decsim.assembly as assembly
import decsim.build.controller_side as controller_side
import decsim.build.decoders as decoder_build
import decsim.build.escalation as escalation_build
import decsim.build.listeners as listener_build
import decsim.build.parts as build_parts
import decsim.build.plan as plan_build
import decsim.build.stores as store_build
import decsim.controller.conditional_release as conditional_release_module
import decsim.controller.controller as controller_module
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.links.fabric as fabric
import decsim.observe.link_traffic as link_traffic
import decsim.observe.observation as observation_module
import decsim.observe.wiring as wiring
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.records.results as result_records
import decsim.seeding as seeding
import decsim.settings as machine_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import decsim.windows.window_manager as window_manager_module
from decsim.syndrome_buffer import (
    strong_syndrome_round_receiver as strong_syndrome_round_receiver_module,
)
from decsim.syndrome_buffer import (
    weak_syndrome_round_receiver as weak_syndrome_round_receiver_module,
)


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
    weak_syndrome_buffer: syndrome_buffer_module.SyndromeBuffer
    weak_syndrome_round_receiver: (
        weak_syndrome_round_receiver_module.WeakSyndromeRoundReceiver
    )
    strong_syndrome_buffer: Optional[syndrome_buffer_module.SyndromeBuffer]
    strong_syndrome_round_receiver: Optional[
        strong_syndrome_round_receiver_module.StrongSyndromeRoundReceiver
    ]
    pauli_frame: Optional[pauli_frame_module.PauliFrame]
    window_manager: window_manager_module.WindowManager
    assembler: round_assembly.RoundAssembler
    syndrome_round_sender: syndrome_round_sender.SyndromeRoundSender
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
        """Build every seat of the assembly file, wire it, and start it.

        The seed is the run's root; every stochastic component derives
        its own from it and its path.
        """
        root_seed = _root_seed(seed)
        engine = engine_module.Engine()
        escalation_policy = escalation_build.build_escalation_policy(
            settings.escalation, settings.weak_decoder
        )
        plan = plan_build.build_plan(settings, escalation_policy)
        detection_events = controller_side.build_detection_events(
            settings, plan.device
        )
        pool = decoder_build.build_decoder_pool(
            settings, plan, escalation_policy, detection_events
        )
        store_build.check_readout_cost_is_priced(settings)
        store_build.check_store_kinds(settings)
        controller_side.check_strong_route(escalation_policy, pool.router)
        parts = build_parts.Parts(
            settings=settings,
            engine=engine,
            plan=plan,
            escalation_policy=escalation_policy,
            pool=pool,
            detection_events=detection_events,
        )
        seats = assembly.build_seats(parts)
        wires = assembly.wires_for(parts)
        assembly.bind(wires, seats)
        assembly.start_wired_seats(seats)
        seed_roots = assembly.seed_roots(parts, seats)
        seeding.bind_run_seed(root_seed, seed_roots)
        listeners = _observe(settings, parts, seats, seed)
        strong_syndrome_buffer = seats.get("strong_syndrome_buffer")
        strong_syndrome_round_receiver = seats.get(
            "strong_syndrome_round_receiver"
        )
        pauli_frame = seats.get("pauli_frame")
        listener_build.load_program(
            plan,
            seats["conditional_release"],
            seats["window_manager"],
            seats["streams"],
            seats["idle_rounds"],
            seats["execution_runtime"],
        )
        return cls(
            settings=settings,
            engine=engine,
            observation=listeners,
            links=seats["links"],
            conditional_release=seats["conditional_release"],
            weak_syndrome_buffer=seats["weak_syndrome_buffer"],
            weak_syndrome_round_receiver=seats["weak_syndrome_round_receiver"],
            strong_syndrome_buffer=strong_syndrome_buffer,
            strong_syndrome_round_receiver=strong_syndrome_round_receiver,
            pauli_frame=pauli_frame,
            window_manager=seats["window_manager"],
            assembler=seats["assembler"],
            syndrome_round_sender=seats["syndrome_round_sender"],
            transmitter=seats["transmitter"],
            decoder_manager=seats["decoder_manager"],
            active_decoder=pool.active,
            factory=seats["factory"],
            syndrome_source=plan.device,
            qpu=seats["qpu"],
            controller=seats["controller"],
            issuer=seats["issuer"],
            instruction_output=seats["instruction_output"],
            idle_rounds=seats["idle_rounds"],
            execution_runtime=seats["execution_runtime"],
            operations=plan.all_operations,
        )

    def start(self) -> None:
        """Queue the first event of every component that has one.

        Every component is built and wired first and nothing is
        scheduled while the graph is still being assembled, so the order
        the root builds in cannot move a tick; each seat with a first
        event queues it here, in build order. That is gem5's split
        between the constructor and startup, "the appropriate place to
        schedule initial event(s)"
        (tmp/resources/gem5/src/sim/sim_object.hh lines 194 and 280).
        """
        self.factory.start()
        self.execution_runtime.start()

    def run(self) -> result_records.RunResult:
        """Start every component, run to quiescence, read the result."""
        self.start()
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
        self.syndrome_round_sender.check_settled()
        self.transmitter.check_settled()
        self.weak_syndrome_round_receiver.check_settled()
        if self.strong_syndrome_round_receiver is not None:
            self.strong_syndrome_round_receiver.check_settled()
        return _capture_result(self)


def _observe(
    settings: machine_settings.MachineSettings,
    parts: build_parts.Parts,
    seats: dict,
    seed: Optional[int],
) -> observation_module.Observation:
    """Connect the run's listeners, before the workload is loaded."""
    traffic_ledger = link_traffic.TrafficLedger(settings.links)
    process_name = controller_side.process_name(settings, seed)
    return wiring.observe(
        settings.observation,
        parts.engine,
        seats,
        process_name=process_name,
        operations=parts.plan.operations,
        traffic_ledger=traffic_ledger,
        syndrome_source=parts.plan.device,
        pool=parts.pool,
    )


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
