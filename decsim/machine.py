"""The root: one object that builds every component and wires them.

A Machine is gem5's shape (src/python/m5/SimObject.py: a SimObject's
Python class is its params, `allClasses` maps a name to a class; the
learning_gem5 simple.py script names each component once and assigns
its ports), with the wiring done by constructor because Python needs
no second bind step. MachineSettings holds one settings record per yaml
section, each built by its own package's `from_yaml`; the tables below
map each section's `kind` to the class that fills the port, sinter's
`BUILT_IN_DECODERS` (sinter/_decoding/_decoding_all_built_in_decoders.py)
made one dict per pluggable part. A new component is one class that
fills its port in decsim/ports.py and one row here.

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
"""

import copy
import dataclasses
import functools
from collections.abc import Mapping
from typing import Any, Optional

import stim

import decsim.confidence.complementary as complementary
import decsim.confidence.decoder as confidence_decoder
import decsim.config as config
import decsim.controller.controller as controller_module
import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.policies as policies
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.decoders.belief_matching.decoder as belief_matching
import decsim.decoders.belief_propagation_osd.decoder as belief_propagation_osd
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.decoders.decoder_memory as decoder_memory_module
import decsim.decoders.decoders as decoders
import decsim.decoders.minimum_weight_perfect_matching.decoder as minimum_weight_perfect_matching
import decsim.decoders.relay_belief_propagation.decoder as relay_belief_propagation
import decsim.decoders.schedulers as schedulers
import decsim.decoders.settings as decoder_settings
import decsim.decoders.staged_decoder as staged_decoder
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.union_find.decoder as union_find
import decsim.decoders.verify_windows as verify_windows
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.threshold_sources as threshold_sources
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.frontends.planner as planner
import decsim.frontends.settings as workload_settings
import decsim.links.fabric as fabric
import decsim.links.link_profiles as link_profiles
import decsim.links.settings as link_settings
import decsim.message as message
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
import decsim.observe.round_events as round_events_module
import decsim.observe.result_ledger as result_ledger_module
import decsim.observe.round_store_occupancy as round_store_occupancy_module
import decsim.observe.runtime_stamps as runtime_stamps_module
import decsim.observe.stage_records as stage_records_module
import decsim.observe.settings as observe_settings
import decsim.observe.trace_writer as trace_writer_module
import decsim.observe.window_ledger as window_ledger_module
import decsim.pauli_frame.conditional_release as conditional_release_module
import decsim.pauli_frame.pauli_frame as pauli_frame_module
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.round_policies as round_policies
import decsim.qpu.settings as qpu_settings
import decsim.qpu.stim_device as stim_device
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.seeding as seeding
import decsim.syndrome_buffer.round_store as round_store_module
import decsim.syndrome_buffer.settings as round_store_settings
import decsim.syndrome_buffer.strong_round_writer as strong_round_writer_module
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.decode_requests as decode_requests
import decsim.windows.operation_results as operation_results
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_commits as window_commits
import decsim.windows.window_interactions as window_interactions
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner_module
import decsim.windows.window_transfers as window_transfers_module
import decsim.windows.windowing_schemes as windowing_schemes

# ------------------------------------------------------------ the tables
#
# One dict per pluggable part: the `kind` a yaml section names, to the
# class or builder that fills the part's port. The machine looks a kind
# up once and refuses one that is not a row, naming the rows.

SYNDROME_SOURCES = {
    "stim_device": stim_device.StimDevice,
    "timing_only": syndrome_devices.TimingOnlyDevice,
    "syndrome_bits": syndrome_devices.SyndromeBitDevice,
    "recorded_stim": stim_device.RecordedStimDevice,
}
# A named row decodes every window for real and is charged its measured
# wall clock; a number instead of a name is a fixed core latency in
# microseconds on the MWPM path (_algorithm). Every row is one class on
# the Decoder port (decsim/decoders/decoder.py); sinter's
# BUILT_IN_DECODERS is the shape.
DECODERS = {
    "pymatching": minimum_weight_perfect_matching.PyMatchingDecoder,
    "unweighted_pymatching": minimum_weight_perfect_matching.UnweightedPyMatchingDecoder,
    "belief_matching": belief_matching.BeliefMatchingDecoder,
    "union_find": union_find.UnionFindDecoder,
    "tesseract": tesseract.TesseractDecoder,
    "relay_bp": relay_belief_propagation.RelayBeliefPropagationDecoder,
    "bposd": belief_propagation_osd.BeliefPropagationOsdDecoder,
}
# The soft output a switching run's weak decoder reports, and the wrapper
# that attaches it (escalation.gap_computation). No yaml key names the
# signal yet; the root takes the one row, and the switching policy
# expects its source.
CONFIDENCE_SIGNALS = {
    "complementary_gap": complementary.ComplementaryGap,
}
CONFIDENCE_SIGNAL_KIND = "complementary_gap"
CONFIDENCE_WRAPPERS = {
    "serial": confidence_decoder.SoftOutputDecoder,
    "parallel_pair": confidence_decoder.ParallelGapDecoder,
    "split_pair": confidence_decoder.SplitGapDecoder,
}
ROUND_STORES = {
    "round_store": round_store_module.RoundStore,
}
ESCALATIONS = {
    "weak_baseline": escalation_policies.Baseline,
    "strong_only": escalation_policies.StrongOnly,
    "switching": escalation_policies.Switching,
}
WINDOWING_SCHEMES = {
    "sliding": windowing_schemes.SlidingWindowScheme,
    "parallel": windowing_schemes.ParallelWindowScheme,
    "sandwich": windowing_schemes.TanSandwichScheme,
    "naive_online": windowing_schemes.NaiveOnlineScheme,
}
IDLE_POLICIES = {
    "separate_decode_jobs": policies.SeparateDecodeJobs,
    "ignore": policies.Ignore,
    "extend_stream": policies.ExtendStream,
}
# A workload row turns its settings and the run's code into the
# operations and, when the row fixes them, the rounds policy.
WORKLOADS = {
    "memory_circuit": workload_settings.memory_circuit_operations,
    "circuit_list": workload_settings.circuit_list_operations,
    "surgery_ir": workload_settings.surgery_ir_operations,
    "qlx": workload_settings.qlx_operations,
}
MAGIC_STATE_FACTORIES = {
    "infinite": magic_state_factories.InfiniteFactory,
    "distillation": magic_state_factories.DistillationFactory,
    "multi_level": magic_state_factories.MultiLevelDistillationFactory,
}
WINDOW_CHECKS = {
    "none": None,
    "tesseract": verify_windows.TesseractCheckedDecoder,
}
SECTIONS = (
    "clocks",
    "qpu",
    "controller",
    "idle_policy",
    "links",
    "round_store",
    "strong_round_store",
    "windows",
    "weak_decoder",
    "strong_decoder",
    "escalation",
    "pauli_frame",
    "workload",
    "observation",
)


@dataclasses.dataclass(frozen=True)
class MachineSettings:
    """One settings record per yaml section, plus the Python-only knobs.

    Every field has a default, so a Python caller names only what
    differs from a timing-only run of three-qubit surface code patches
    with no decoder at all. links is the fabric card; the reference card
    prices propagation only. decoder_manager holds the manager's
    Python-only knobs; magic_state_factory has no yaml key today.
    """

    clocks: config.ClockSettings = config.ClockSettings()
    qpu: qpu_settings.QpuSettings = qpu_settings.QpuSettings()
    controller: controller_settings.ControllerSettings = (
        controller_settings.ControllerSettings()
    )
    idle_policy: controller_settings.IdlePolicySettings = (
        controller_settings.IdlePolicySettings()
    )
    links: link_settings.FabricSettings = (
        link_profiles.logical_reference_profile()
    )
    round_store: round_store_settings.RoundStoreSettings = (
        round_store_settings.RoundStoreSettings()
    )
    strong_round_store: round_store_settings.RoundStoreSettings = (
        round_store_settings.RoundStoreSettings()
    )
    windows: window_settings.WindowSettings = window_settings.WindowSettings()
    weak_decoder: decoder_settings.DecoderSettings = (
        decoder_settings.DecoderSettings()
    )
    strong_decoder: decoder_settings.DecoderSettings = (
        decoder_settings.DecoderSettings()
    )
    decoder_manager: decoder_settings.DecoderManagerSettings = (
        decoder_settings.DecoderManagerSettings()
    )
    escalation: decoder_settings.EscalationSettings = (
        decoder_settings.EscalationSettings()
    )
    pauli_frame: Optional[pauli_frame_module.PauliFrameConfig] = None
    workload: workload_settings.WorkloadSettings = (
        workload_settings.WorkloadSettings()
    )
    magic_state_factory: qpu_settings.FactorySettings = (
        qpu_settings.FactorySettings()
    )
    observation: observe_settings.ObservationSettings = (
        observe_settings.ObservationSettings()
    )

    @classmethod
    def from_mapping(
        cls, sections: Mapping, *, name: str, base_directory
    ) -> "MachineSettings":
        """One yaml's sections, each handed to the package that owns it.

        name labels the links card in the traffic ledger; base_directory
        resolves the escalation section's relative table path.
        """
        unknown = set(sections) - set(SECTIONS)
        if unknown:
            listed = sorted(unknown)
            raise ValueError(
                f"the yaml has no section {listed}; the sections are "
                f"{list(SECTIONS)}"
            )
        clocks = config.ClockSettings.from_yaml(sections["clocks"])
        idle_policy = controller_settings.IdlePolicySettings()
        if "idle_policy" in sections:
            idle_policy = controller_settings.IdlePolicySettings(
                kind=sections["idle_policy"]
            )
        escalation_section = sections.get("escalation", {})
        observation_section = sections.get("observation", {})
        qpu = qpu_settings.QpuSettings.from_yaml(sections["qpu"])
        controller = controller_settings.ControllerSettings.from_yaml(
            sections["controller"], clocks
        )
        links = link_profiles.from_yaml(sections["links"], clocks, name)
        round_store = round_store_settings.RoundStoreSettings.from_yaml(
            sections["round_store"]
        )
        strong_round_store = round_store_settings.RoundStoreSettings.from_yaml(
            sections["strong_round_store"]
        )
        windows = window_settings.WindowSettings.from_yaml(sections["windows"])
        weak_decoder = _tier_settings(sections, "weak_decoder", clocks)
        strong_decoder = _tier_settings(sections, "strong_decoder", clocks)
        escalation = decoder_settings.EscalationSettings.from_yaml(
            escalation_section, base_directory
        )
        pauli_frame = pauli_frame_module.PauliFrameConfig.from_yaml(
            sections["pauli_frame"], clocks
        )
        workload = workload_settings.WorkloadSettings.from_yaml(
            sections["workload"]
        )
        observation = observe_settings.ObservationSettings.from_yaml(
            observation_section
        )
        return cls(
            clocks=clocks,
            qpu=qpu,
            controller=controller,
            idle_policy=idle_policy,
            links=links,
            round_store=round_store,
            strong_round_store=strong_round_store,
            windows=windows,
            weak_decoder=weak_decoder,
            strong_decoder=strong_decoder,
            escalation=escalation,
            pauli_frame=pauli_frame,
            workload=workload,
            observation=observation,
        )


@dataclasses.dataclass(frozen=True)
class LogicalOperationResult:
    """One operation's prediction and, when sampled, its truth.

    logical_failure is true when any predicted bit differs from truth.
    """

    operation_id: int
    result_status: str
    logical_observables: Optional[tuple]
    stream_offset: Optional[int]
    observable_truth: Optional[tuple] = None
    logical_failure: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class RunResult:
    """What one completed run computed; the gate hashes these fields."""

    terminal_status: str
    event_queue_empty: bool
    decode_work_settled: bool
    execution_workload_complete: bool
    execution_done_ticks: int
    fully_done_ticks: int
    operation_results: tuple
    link_traffic: dict
    # the copies, references and moves of the data path; None unless the
    # observation section asked for them
    data_movement: Optional[dict]


@dataclasses.dataclass(frozen=True)
class Machine:
    """Every component of one run, built from its settings and wired.

    Build one per seed with Machine.build, then run it once; the
    components stay readable afterwards for the measurements and the
    views that read them. active_decoder is the unit of the tier that
    decodes the plan's windows, None when no decoder was named.
    """

    settings: MachineSettings
    engine: engine_module.Engine
    observation: observation_module.Observation
    links: fabric.LinkFabric
    conditional_release: conditional_release_module.ConditionalRelease
    round_store: round_store_module.RoundStore
    round_store_occupancy: Optional[
        round_store_occupancy_module.RoundStoreOccupancy
    ]
    strong_round_store: Optional[round_store_module.RoundStore]
    strong_round_writer: Optional[strong_round_writer_module.StrongRoundWriter]
    pauli_frame: Optional[pauli_frame_module.PauliFrame]
    window_manager: window_manager_module.WindowManager
    assembler: round_assembly.RoundAssembler
    round_writer: round_writes.RoundWriter
    transmitter: round_transmission.RoundTransmitter
    round_events: round_events_module.RoundEventRecorder
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
        cls, settings: MachineSettings, seed: Optional[int] = 0
    ) -> "Machine":
        """Build every component in pipeline order and wire it.

        The order is the one a readout travels, so a reader follows the
        wiring by reading down. The seed is the run's root; every
        stochastic component derives its own from it and its path.
        """
        root_seed = _root_seed(seed)
        observation = settings.observation
        engine = engine_module.Engine()
        log = _connect_log(observation, engine)
        escalation_policy = _escalation_policy(settings.escalation)
        plan = _plan(settings, escalation_policy)
        pool = _decoder_pool(settings, plan)
        _check_readout_cost_is_priced(settings)
        conditional_release = conditional_release_module.ConditionalRelease(
            engine
        )
        traffic_ledger = link_traffic.TrafficLedger(settings.links)
        links = fabric.LinkFabric(settings.links, engine)
        links.transfer_delivered.connect(traffic_ledger.on_transfer)
        round_events = round_events_module.RoundEventRecorder(engine)
        held_rounds = round_writes.HeldRounds(
            engine, settings.controller.packing_overflow
        )
        round_store = _round_store(settings.round_store, held_rounds)
        round_store_occupancy = _round_store_occupancy(
            observation, engine, round_store
        )
        strong_round_store = _strong_round_store(
            settings.strong_round_store, escalation_policy, held_rounds
        )
        if strong_round_store is not None:
            strong_round_store.round_stored.connect(round_events.round_stored)
        pauli_frame = _pauli_frame(settings.pauli_frame, engine)
        # The decoder manager is built first, so the requester and the
        # strong redecode take the decode queue by constructor and every
        # job carries its return path.
        _check_strong_route(settings, pool.router)
        # Every listener is built before the component it hears, so a
        # source fired while the program loads is already heard.
        window_ledger = window_ledger_module.WindowLedger()
        result_ledger = result_ledger_module.ResultLedger()
        runtime_stamps = runtime_stamps_module.RuntimeStamps()
        queue_depth = queue_depth_module.QueueDepthLog()
        controller_counters = controller_counters_module.ControllerCounters()
        command_events = command_events_module.CommandEvents()
        stages = _connect_stage_records(pool)
        trace_writer = _trace_writer(observation, engine, settings)
        data_movement = _data_movement(observation)
        decode_records = _decode_records(observation)
        decoder_manager = _decoder_manager(
            engine, settings, escalation_policy, pool, links
        )
        _connect_decode_records(decoder_manager, decode_records)
        decoder_manager.queue.depth_changed.connect(queue_depth.depth_changed)
        window_manager = _window_manager(
            engine,
            settings,
            escalation_policy,
            plan,
            window_ledger=window_ledger,
            result_ledger=result_ledger,
            links=links,
            conditional_release=conditional_release,
            fault_model_requirement_for=pool.router.fault_model_requirement_for,
            syndrome_buffer=round_store,
            syndrome_buffer_1=strong_round_store,
            pauli_frame=pauli_frame,
            decode_queue=decoder_manager,
            on_workload_complete=lambda: factory.shutdown(),
        )
        strong_round_writer = _strong_round_writer(
            engine, links, strong_round_store, window_manager
        )
        transmitter = round_transmission.RoundTransmitter(
            engine, links, round_store, window_manager
        )
        publishes_from_strong_store = (
            window_manager.retention.primary_store is not round_store
        )
        round_writer = round_writes.RoundWriter(
            engine,
            round_store,
            strong_round_writer,
            publishes_from_strong_store=publishes_from_strong_store,
            held_rounds=held_rounds,
            transmitter=transmitter,
        )
        form_round = getattr(plan.device, "form_round", None)
        rounds_in_flight = round_assembly.RoundsInFlight(
            settings.controller.packing_rounds_in_flight,
            held_rounds,
            transmitter,
        )
        assembler = round_assembly.RoundAssembler(
            engine,
            settings.controller,
            form_round=form_round,
            on_packed=round_writer.admit,
            rounds_in_flight=rounds_in_flight,
        )
        factory = _factory(
            settings.magic_state_factory, engine, decoder_manager, plan
        )
        qpu = cycle_clock.QPUDevice(engine, plan.device, plan.round_ticks)
        qpu.command_event.connect(command_events.command_event)
        pulse_ticks = settings.controller.decision_to_pulse_ticks()
        instruction_output = instruction_output_module.InstructionOutput(
            engine, links, qpu, pulse_ticks
        )
        streams = _feedback_streams(
            engine,
            plan,
            qpu,
            window_manager,
            retry_ready_operations=lambda: (
                execution_runtime.retry_ready_operations()
            ),
        )
        patch_by_identity = _patch_by_identity(plan)
        idle_rounds = idle_rounds_module.IdleRoundAccounting(
            plan.idle_policy, decoder_manager, patch_by_identity, streams, qpu
        )
        idle_rounds.idle_round_emitted.connect(
            controller_counters.idle_round_emitted
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
        _connect_round_events(
            round_events,
            controller,
            assembler,
            held_rounds,
            round_writer,
            transmitter,
            instruction_output,
        )
        execution_runtime = execution_runtime_module.ExecutionRuntime(
            engine,
            issuer=issuer,
            factory=factory,
            resource_claims_by_operation_id=plan.resource_claims,
        )
        _connect_runtime_stamps(execution_runtime, runtime_stamps)
        # The QPU's receivers arrive through its constructor once the
        # qpu lane builds the controller first.
        qpu.connect_readout_receiver(controller)
        qpu.connect_completion_receiver(execution_runtime.body_done)
        qpu.connect_idle_receiver(idle_rounds.emit_idle_round)
        seed_roots = _seed_roots(
            code=plan.code,
            scheme=plan.scheme,
            device=plan.device,
            error_model_provider=plan.error_model_provider,
            decoder_router=pool.router,
            factory=factory,
            escalation_policy=escalation_policy,
            scheduler=pool.scheduler,
            decoder_memory_transfer=decoder_manager.service.staging.transport,
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
        _load_program(
            plan,
            conditional_release,
            window_manager,
            streams,
            idle_rounds,
            execution_runtime,
        )
        listeners = _observation(
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
            round_events=round_events,
            pauli_frame=pauli_frame,
            operations=plan.operations,
            trace_writer=trace_writer,
            data_movement=data_movement,
        )
        return cls(
            settings=settings,
            engine=engine,
            observation=listeners,
            links=links,
            conditional_release=conditional_release,
            round_store=round_store,
            round_store_occupancy=round_store_occupancy,
            strong_round_store=strong_round_store,
            strong_round_writer=strong_round_writer,
            pauli_frame=pauli_frame,
            window_manager=window_manager,
            assembler=assembler,
            round_writer=round_writer,
            transmitter=transmitter,
            round_events=round_events,
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

    def run(self) -> RunResult:
        """Run the engine to quiescence, check settlement, read the result."""
        self.engine.run()
        strong_redecode = self.window_manager.strong_redecode
        if strong_redecode is not None and strong_redecode.has_pending():
            pending = strong_redecode.pending_work()
            raise RuntimeError(
                f"the run ended with pending strong escalations: {pending}"
            )
        self.decoder_manager.check_decode_work_settled()
        self.assembler.check_settled()
        self.round_writer.check_settled()
        self.transmitter.check_settled()
        if self.strong_round_writer is not None:
            self.strong_round_writer.check_settled()
        return _capture_result(self)


def build_decoder_unit(settings: MachineSettings, tier: str):
    """The decoder unit of one tier, weak or strong, as the root builds it.

    A named row decodes for real inside a StagedDecoder whose fetch and
    release stages are cycles of the tier's clock; a number is a fixed
    core latency on the MWPM path. The tier that decodes the plan's
    windows carries the confidence signal's wrapper when the escalation
    switches (CONFIDENCE_WRAPPERS by gap_computation) and the Tesseract
    referee when the observation asks for it; the strong tier of a
    switching run is unwrapped (its result is final). A Python-built
    decoder is returned as it is. None when the tier names no decoder.
    """
    tier_settings = getattr(settings, f"{tier}_decoder")
    if tier_settings.decoder is not None:
        return tier_settings.decoder
    if tier_settings.kind is None:
        return None
    algorithm = _algorithm(tier_settings.kind, tier)
    is_active = tier == settings.escalation.decodes_on
    is_switching = settings.escalation.kind == "switching"
    if is_active and is_switching:
        # a named row is measured on the host clock; a number is priced
        is_measured = isinstance(tier_settings.kind, str)
        algorithm = _gap_wrapped(algorithm, settings.escalation, is_measured)
    check = _row(
        WINDOW_CHECKS,
        "observation.check_windows_with",
        settings.observation.check_windows_with,
    )
    if is_active and check is not None:
        algorithm = check(algorithm)
    return _staged_unit(tier_settings, algorithm)


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Everything the wiring reads that the planner and the workload fix."""

    code: Any
    layout: Any
    scheme: Any
    boundary_policy: Any
    window_interaction: Any
    idle_policy: Any
    operations: tuple
    decode_operations: tuple
    dynamic_streams: tuple
    protected_regions: tuple
    all_operations: tuple
    planned_operations: tuple
    view_by_id: dict
    run_plan: planner.RunPlan
    resource_claims: dict
    device: Any
    error_model_provider: Any

    @property
    def round_ticks(self) -> int:
        return self.run_plan.round_ticks


@dataclasses.dataclass(frozen=True)
class _DecoderPool:
    """The router over the tiers' units and the manager's pool knobs."""

    router: Any
    active: Any
    unit_pools: dict
    decoder_memory: Optional[decoder_memory_module.DecoderMemoryConfig]
    scheduler: Any


def _row(table: dict, section: str, kind):
    """The table row a section's kind names; a kind off the table is refused."""
    if kind not in table:
        rows = sorted(table)
        raise ValueError(
            f"{section} {kind!r} is not a row of its table; the rows are {rows}"
        )
    return table[kind]


def _tier_settings(
    sections: Mapping, tier: str, clocks: config.ClockSettings
) -> decoder_settings.DecoderSettings:
    """A tier's section, or no decoder when the yaml leaves it out."""
    if tier not in sections:
        return decoder_settings.DecoderSettings()
    return decoder_settings.DecoderSettings.from_yaml(sections[tier], clocks)


def _root_seed(value) -> Optional[int]:
    """The run's root seed: None for entropy, else an unsigned 64-bit int."""
    if value is None:
        return None
    root_seed = int(value)
    if not 0 <= root_seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return root_seed


# ------------------------------------------------- the escalation policy


def _escalation_policy(settings: decoder_settings.EscalationSettings):
    """The policy of the escalation kind, or the Python-built one."""
    if settings.policy is not None:
        return settings.policy
    row = _row(ESCALATIONS, "escalation.kind", settings.kind)
    if row is escalation_policies.Switching:
        threshold = _threshold_source(settings)
        signal = _confidence_signal()
        return escalation_policies.Switching(threshold, signal.source)
    return row()


def _threshold_source(settings: decoder_settings.EscalationSettings):
    """The sweep point's online source, or the fixed threshold.

    The table source is resolved to a fixed threshold per sweep point by
    the front (ExperimentConfig.point_settings), so a table run reaches
    the root with its threshold in nats or not at all.
    """
    if settings.online_threshold is not None:
        return settings.online_threshold
    if settings.gap_threshold_nats is None:
        raise ValueError(
            "escalation.threshold_source table resolves the threshold per "
            "sweep point in the front (ExperimentConfig.point_settings); "
            "build the machine through it, or give gap_threshold_db"
        )
    return threshold_sources.FixedThreshold(settings.gap_threshold_nats)


# ------------------------------------------------ the workload and plan


def _plan(settings: MachineSettings, escalation_policy) -> _Plan:
    """The code, the workload's operations and the window plan."""
    code, layout = settings.qpu.build_code(settings.windows)
    operations, decode_operations, dynamic_streams, rounds_policy = _operations(
        settings.workload, code
    )
    every_operation = operations + decode_operations + dynamic_streams
    all_operations = _unique_operations(every_operation)
    views = []
    for operation in all_operations:
        view = message.OperationPlanningView.from_operation(operation)
        views.append(view)
    views = tuple(views)
    view_by_id = {}
    for view in views:
        view_by_id[view.id] = view
    external_blocker_ids = []
    for operation in decode_operations + dynamic_streams:
        external_blocker_ids.append(operation.id)
    planner.check_operation_graph(
        list(operations),
        validate_blockers=True,
        external_blocker_ids=external_blocker_ids,
    )
    scheme = _scheme(settings.windows, settings.escalation)
    boundary_policy = _boundary_policy(settings.windows, settings.escalation)
    window_interaction = settings.windows.window_interaction
    if window_interaction is None:
        window_interaction = window_interactions.DefaultWindowInteraction()
    is_sliding = type(scheme) is windowing_schemes.SlidingWindowScheme
    if dynamic_streams and not is_sliding:
        raise ValueError("dynamic streams require SlidingWindowScheme")
    has_static_decode_plan = settings.workload.decode_operations is not None
    has_frontend = settings.workload.kind in ("surgery_ir", "qlx")
    commit_round_count = code.commit_rounds()
    buffer_round_count = code.buffer_rounds()
    run_shape = message.RunShape(
        scheme=scheme,
        boundary_policy=boundary_policy,
        operations=views,
        commit_round_count=commit_round_count,
        buffer_round_count=buffer_round_count,
        is_double_window=settings.escalation.double_window,
        is_bulk_strong=settings.decoder_manager.bulk_strong,
        has_dynamic_streams=bool(dynamic_streams),
        has_static_decode_plan=has_static_decode_plan,
        has_frontend=has_frontend,
    )
    escalation_policy.check_plan(run_shape)
    planned_operations = _decode_plan_operations(
        operations,
        decode_operations,
        dynamic_streams,
        static_decode_selected=has_static_decode_plan,
    )
    planned_ids = []
    for operation in planned_operations:
        planned_ids.append(operation.id)
    run_plan = planner.plan_execution(
        operations=views,
        planned_operation_ids=tuple(planned_ids),
        code=code,
        layout=layout,
        scheme=scheme,
        rounds_policy=rounds_policy,
        fallback_round_microseconds=settings.qpu.round_period_microseconds,
        retain_strong_context=escalation_policy.requires_strong_context,
        double_window=settings.escalation.double_window,
        has_open_ended_dynamic_streams=bool(dynamic_streams),
    )
    resource_claims = {}
    for operation in operations:
        view = view_by_id[operation.id]
        claims = layout.resources_for(view)
        resource_claims[operation.id] = tuple(claims)
    device = _syndrome_source(settings.qpu)
    _install_device_circuits(device, all_operations)
    error_model_provider = settings.qpu.error_model_provider
    if error_model_provider is None:
        error_model_provider = device
    elif hasattr(error_model_provider, "operation_circuit_scope"):
        _install_device_circuits(error_model_provider, all_operations)
    idle_policy = _idle_policy(settings.idle_policy)
    return _Plan(
        code=code,
        layout=layout,
        scheme=scheme,
        boundary_policy=boundary_policy,
        window_interaction=window_interaction,
        idle_policy=idle_policy,
        operations=operations,
        decode_operations=decode_operations,
        dynamic_streams=dynamic_streams,
        protected_regions=tuple(settings.workload.protected_regions),
        all_operations=all_operations,
        planned_operations=tuple(planned_operations),
        view_by_id=view_by_id,
        run_plan=run_plan,
        resource_claims=resource_claims,
        device=device,
        error_model_provider=error_model_provider,
    )


def _operations(settings: workload_settings.WorkloadSettings, code) -> tuple:
    """The workload's private operation copies and its rounds policy.

    The run never mutates the caller's operations; an operation without
    its own feedback boundary mode takes the workload's.
    """
    row = _row(WORKLOADS, "workload.kind", settings.kind)
    source_operations, fixed_rounds_policy = row(settings, code)
    rounds_policy = settings.rounds_policy
    if rounds_policy is None:
        rounds_policy = fixed_rounds_policy
    if rounds_policy is None:
        rounds_policy = round_policies.GateRounds()
    decode_operations = settings.decode_operations or ()
    copies = {}
    operations = _copies(source_operations, copies, settings)
    decode_copies = _copies(decode_operations, copies, settings)
    stream_copies = _copies(settings.dynamic_streams, copies, settings)
    planner.check_workload_identity(operations, decode_copies, stream_copies)
    return operations, decode_copies, stream_copies, rounds_policy


def _copies(
    operations, copies: dict, settings: workload_settings.WorkloadSettings
) -> tuple:
    """Private copies of the operations, one per object identity."""
    copied = []
    for operation in operations:
        identity = id(operation)
        if identity not in copies:
            copies[identity] = _private_copy(operation, settings)
        copied.append(copies[identity])
    return tuple(copied)


def _private_copy(operation, settings: workload_settings.WorkloadSettings):
    private = copy.copy(operation)
    if private.feedback_boundary_mode is None:
        private.feedback_boundary_mode = settings.feedback_boundary_mode
    return private


def _unique_operations(operations) -> tuple:
    unique = {}
    for operation in operations:
        unique.setdefault(operation.id, operation)
    unique_operations = unique.values()
    return tuple(unique_operations)


def _decode_plan_operations(
    operations, decode_operations, dynamic_streams, *, static_decode_selected
) -> tuple:
    """The operations whose windows the plan decodes."""
    if static_decode_selected:
        return decode_operations
    dynamic_ids = set()
    for stream in dynamic_streams:
        dynamic_ids.add(stream.id)
    planned = []
    for operation in operations:
        if not operation.emits_detector_data:
            continue
        if operation.stream_id in dynamic_ids:
            continue
        planned.append(operation)
    return tuple(planned)


def _scheme(
    windows: window_settings.WindowSettings,
    escalation: decoder_settings.EscalationSettings,
):
    """The windowing scheme of the kind, or the Python-built one.

    Switching needs the lookahead terminal policy on sliding windows:
    the literature-exact flush has no trailing tail context.
    """
    if windows.scheme is not None:
        return windows.scheme
    row = _row(WINDOWING_SCHEMES, "windows.kind", windows.kind)
    if escalation.kind != "switching":
        return row()
    if windows.kind != "sliding":
        raise ValueError("escalation switching requires windows.kind sliding")
    lookahead = windowing_schemes.SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD
    return windowing_schemes.SlidingWindowScheme(terminal_policy=lookahead)


def _boundary_policy(
    windows: window_settings.WindowSettings,
    escalation: decoder_settings.EscalationSettings,
):
    """Eager shipping, or Held while a serial escalation is pending.

    Serial switching holds boundaries until results are final
    (descendants wait out an escalation); double_window keeps the weak
    chain committing eagerly and absorbs escalations in strong windows.
    """
    if windows.boundary_policy is not None:
        return windows.boundary_policy
    is_serial_switching = (
        escalation.kind == "switching" and not escalation.double_window
    )
    if is_serial_switching:
        return policies.Held()
    return policies.Eager()


def _idle_policy(settings: controller_settings.IdlePolicySettings):
    if settings.policy is not None:
        return settings.policy
    row = _row(IDLE_POLICIES, "idle_policy", settings.kind)
    return row()


def _syndrome_source(settings: qpu_settings.QpuSettings):
    """The device of the qpu kind, or the Python-built one."""
    if settings.device is not None:
        return settings.device
    row = _row(SYNDROME_SOURCES, "qpu.kind", settings.kind)
    return row(**settings.arguments)


def _install_device_circuits(device, operations) -> None:
    """Give a per-operation device its own copy of every circuit."""
    scope = getattr(device, "operation_circuit_scope", None)
    if scope == "none":
        for operation in operations:
            operation.circuit = None
        return
    if scope != "per_operation":
        raise ValueError(
            "device operation_circuit_scope must be none or per_operation"
        )
    for operation in operations:
        if operation.circuit is None:
            continue
        circuit_text = str(operation.circuit)
        operation.circuit = stim.Circuit(circuit_text)


# ------------------------------------------------------ the decoder pool


def _decoder_pool(settings: MachineSettings, plan: _Plan) -> _DecoderPool:
    """The router over the tiers and the manager's pools.

    Switching puts two units behind one router: the strong pool serves
    escalated jobs, and split_pair adds a gap pool for the sibling
    solves. Any other escalation routes every job to the tier that
    decodes the plan's windows.
    """
    manager = settings.decoder_manager
    scheduler = manager.scheduler
    if scheduler is None:
        scheduler = schedulers.FifoScheduler()
    weak = build_decoder_unit(settings, "weak")
    strong = build_decoder_unit(settings, "strong")
    active = weak
    if settings.escalation.decodes_on == "strong":
        active = strong
    has_no_decoder = active is None and manager.router is None
    if has_no_decoder and plan.planned_operations:
        tier = settings.escalation.decodes_on
        raise ValueError(
            f"the plan decodes windows on the {tier} tier, which names "
            "no decoder: give it a kind or a Python-built decoder"
        )
    router = manager.router
    unit_pools = manager.unit_pools
    if settings.escalation.kind == "switching" and router is None:
        router, unit_pools = _switching_pools(settings, weak, strong)
    if router is None:
        router = decoders.CodeRouter(default=active)
    if unit_pools is None:
        active_settings = _active_tier_settings(settings)
        unit_pools = {"default": active_settings.units}
    decoder_memory = _decoder_memory(settings)
    return _DecoderPool(
        router=router,
        active=active,
        unit_pools=dict(unit_pools),
        decoder_memory=decoder_memory,
        scheduler=scheduler,
    )


def _switching_pools(settings: MachineSettings, weak, strong) -> tuple:
    """The switching router and its pools: default, strong, maybe gap."""
    if strong is None:
        raise ValueError(
            "escalation switching escalates to the strong_decoder, which "
            "this configuration does not define"
        )
    unit_pools = {
        "default": settings.weak_decoder.units,
        "strong": settings.strong_decoder.units,
    }
    gap = None
    if settings.escalation.gap_computation == "split_pair":
        gap = _gap_unit(settings)
        unit_pools["gap"] = settings.escalation.gap_units
    router = decoders.SwitchingRouter(weak, strong, gap=gap)
    return router, unit_pools


def _active_tier_settings(
    settings: MachineSettings,
) -> decoder_settings.DecoderSettings:
    tier = settings.escalation.decodes_on
    return getattr(settings, f"{tier}_decoder")


def _decoder_memory(
    settings: MachineSettings,
) -> Optional[decoder_memory_module.DecoderMemoryConfig]:
    """The pools' input memory: the given one, or the active tier's SRAM."""
    given = settings.decoder_manager.decoder_memory
    if given is not None:
        return given
    active_settings = _active_tier_settings(settings)
    unit_memory_rounds = active_settings.unit_memory_rounds
    if unit_memory_rounds is None:
        return None
    return decoder_memory_module.DecoderMemoryConfig(
        {"default": unit_memory_rounds}
    )


def _algorithm(kind, tier: str):
    """A tier's algorithm: a table row, or a fixed latency on MWPM."""
    if isinstance(kind, str):
        row = _row(DECODERS, f"{tier}_decoder.kind", kind)
        return row(latency_model=None)
    latency_model = decoders.PresetLatencyDecoder(kind)
    return minimum_weight_perfect_matching.PyMatchingDecoder(latency_model)


def _gap_wrapped(
    algorithm, escalation: decoder_settings.EscalationSettings, is_measured
):
    """The weak algorithm carrying the signal, in the computation's wrapper."""
    signal = _confidence_signal()
    wrapper = _row(
        CONFIDENCE_WRAPPERS,
        "escalation.gap_computation",
        escalation.gap_computation,
    )
    is_split = wrapper is confidence_decoder.SplitGapDecoder
    if is_split and not is_measured:
        raise ValueError(
            "split_pair times two real forced-class solves on separate "
            "units, so the weak tier needs a named wall-clock "
            "algorithm; a priced card already models the whole unit "
            "(use parallel_pair to change its cost sheet instead)"
        )
    return wrapper(algorithm, signal)


def _confidence_signal():
    """The signal row a switching run's weak decoder reports and decides on."""
    row = _row(
        CONFIDENCE_SIGNALS, "escalation.confidence", CONFIDENCE_SIGNAL_KIND
    )
    return row()


def _gap_unit(settings: MachineSettings) -> staged_decoder.StagedDecoder:
    """split_pair's sibling pool: one forced solve per window, timed for real.

    Staged over the weak tier's engine timing, since it reads the same
    rounds.
    """
    signal = _confidence_signal()
    half = confidence_decoder.GapHalfDecoder(signal)
    return _staged_unit(settings.weak_decoder, half)


def _staged_unit(
    tier_settings: decoder_settings.DecoderSettings, algorithm
) -> staged_decoder.StagedDecoder:
    """The algorithm between its fetch and release stages.

    Fetch cycles per round before it, release cycles per job after it,
    on the tier's engine clock.
    """
    fetch = staged_decoder.DecoderStage(
        "fetch", cycles_per_round=tier_settings.fetch_cycles_per_round
    )
    release = staged_decoder.DecoderStage(
        "release", cycles_per_job=tier_settings.release_cycles_per_job
    )
    timing = staged_decoder.UnitTiming(
        before=(fetch,),
        after=(release,),
        frequency_mhz=tier_settings.engine_megahertz,
    )
    return staged_decoder.StagedDecoder(algorithm, timing)


# --------------------------------------------------- the stores and frame


def _round_store(
    settings: round_store_settings.RoundStoreSettings,
    held_rounds: round_writes.HeldRounds,
):
    """Buffer 0; a freed slot retries the rounds held for room."""
    row = _row(ROUND_STORES, "round_store.kind", settings.kind)
    return row(settings, on_slot_freed=held_rounds.retry)


def _round_store_occupancy(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    round_store,
):
    """The L5 listener on Buffer 0, only when the observation asks."""
    if not observation.round_store_occupancy:
        return None
    occupancy = round_store_occupancy_module.RoundStoreOccupancy(engine)
    round_store.round_stored.connect(occupancy.round_stored)
    round_store.round_released.connect(occupancy.round_released)
    return occupancy


def _strong_round_store(
    settings: round_store_settings.RoundStoreSettings,
    escalation_policy,
    held_rounds: round_writes.HeldRounds,
):
    """The room-side store, only when a tier reads from the room side."""
    row = _row(ROUND_STORES, "strong_round_store.kind", settings.kind)
    uses_strong_store = (
        escalation_policy.requires_strong_context
        or escalation_policy.primary_tier is message.DecoderTier.STRONG
    )
    if not uses_strong_store:
        return None
    return row(settings, on_slot_freed=held_rounds.retry)


def _strong_round_writer(
    engine: engine_module.Engine,
    links: fabric.LinkFabric,
    strong_round_store,
    window_manager,
):
    """The crossing into the room-side store; the window manager hears it."""
    if strong_round_store is None:
        return None
    return strong_round_writer_module.StrongRoundWriter(
        engine,
        links,
        strong_round_store,
        on_round_stored=window_manager.accept_room_round,
    )


def _pauli_frame(
    settings: Optional[pauli_frame_module.PauliFrameConfig],
    engine: engine_module.Engine,
) -> Optional[pauli_frame_module.PauliFrame]:
    if settings is None:
        return None
    return settings.resolve(engine)


def _check_readout_cost_is_priced(settings: MachineSettings) -> None:
    """A readout cost on the controller needs a card that leaves it out."""
    readout_ticks = settings.controller.readout_to_bits_ticks()
    links = settings.links
    if readout_ticks > 0 and not (
        links.is_controller_processing_outside_qpu_to_controller
    ):
        raise ValueError(
            "a separate controller readout cost requires a link profile "
            "whose QC latency excludes that cost"
        )


# ------------------------------------------------ the pipeline components


def _window_manager(
    engine: engine_module.Engine,
    settings: MachineSettings,
    escalation_policy,
    plan: _Plan,
    *,
    window_ledger: window_ledger_module.WindowLedger,
    result_ledger: result_ledger_module.ResultLedger,
    links,
    conditional_release,
    fault_model_requirement_for,
    syndrome_buffer,
    syndrome_buffer_1,
    pauli_frame,
    decode_queue,
    on_workload_complete,
) -> window_manager_module.WindowManager:
    """The windows facade over its six components, wired by constructor."""
    run_plan = plan.run_plan
    models = window_planner_module.WindowModels(
        plan.error_model_provider, fault_model_requirement_for
    )
    planner = window_planner_module.WindowPlanner(
        plan.scheme,
        run_plan.resolved_operations,
        run_plan.execution,
        models,
        plan.planned_operations,
    )
    window_ledger.load_planned(planner.windows_by_key)
    planner.window_planned.connect(window_ledger.window_planned)
    tracker = round_tracker_module.RoundTracker(plan.scheme, planner)
    retention = round_retention_module.RoundRetention(
        syndrome_buffer,
        syndrome_buffer_1,
        planner,
        tracker,
        is_strong_context_retained=escalation_policy.requires_strong_context,
        primary_tier=escalation_policy.primary_tier,
    )
    transfers = window_transfers_module.WindowTransfers(engine, links)
    builder = decode_requests.DecodeRequestBuilder(
        engine, planner, tracker, plan.window_interaction, transfers
    )
    ledger = committed_rounds.LogicalLedger()
    results = operation_results.OperationResults(
        planner,
        tracker,
        retention,
        ledger,
        conditional_release,
        on_workload_complete,
    )
    publisher = window_commits.CorrectionPublisher(transfers, pauli_frame)
    # the courier's landing callback reaches the facade and the
    # committer's two hooks reach the strong redecode, both built after
    # them; this stands in at wiring time, and a run that never
    # escalates gives the committer no redecode at all
    late = _LateWiring()
    committer_redecode = None
    if escalation_policy.requires_strong_context:
        committer_redecode = late
    courier = window_boundaries.BoundaryCourier(
        planner,
        transfers,
        plan.window_interaction,
        plan.boundary_policy,
        late.accept_boundary,
    )
    committer = window_commits.WindowCommitter(
        engine,
        planner,
        tracker,
        courier,
        publisher,
        committer_redecode,
        results,
    )
    committer.window_committed.connect(window_ledger.window_committed)
    results.operation_result_delivered.connect(
        result_ledger.operation_result_delivered
    )
    requester = decode_requests.DecodeRequester(
        tracker, retention, builder, decode_queue, escalation_policy, committer
    )
    strong_redecode = _strong_redecode(
        escalation_policy,
        settings.escalation,
        engine,
        planner,
        tracker,
        retention,
        builder,
        transfers,
        requester,
        ledger,
        plan.window_interaction,
        decode_queue,
        committer.accept_strong_result,
        window_ledger,
    )
    late.strong_redecode = strong_redecode
    window_manager = window_manager_module.WindowManager(
        engine,
        planner=planner,
        tracker=tracker,
        retention=retention,
        requester=requester,
        courier=courier,
        results=results,
        strong_redecode=strong_redecode,
        window_interaction=plan.window_interaction,
        feedback_boundary_mode=settings.workload.feedback_boundary_mode,
    )
    late.window_manager = window_manager
    return window_manager


def _strong_redecode(
    escalation_policy,
    escalation: decoder_settings.EscalationSettings,
    engine,
    planner,
    tracker,
    retention,
    builder,
    transfers,
    requester,
    ledger,
    interaction,
    decode_queue,
    on_strong_decoded,
    window_ledger: window_ledger_module.WindowLedger,
):
    """The window side of the strong tier, or None when never escalating.

    The strong window's shape is the forward window of Toshio Sec. III C
    under double_window, the two-sided context of Sec. III A otherwise.
    """
    if not escalation_policy.requires_strong_context:
        return None
    if escalation.double_window:
        shape = strong_window_shapes.ForwardWindow(
            engine,
            planner,
            tracker,
            retention,
            builder,
            requester,
            ledger,
            interaction,
        )
        shape.window_absorbed.connect(window_ledger.window_absorbed)
    else:
        shape = strong_window_shapes.ContextWindow(
            engine, planner, tracker, retention, builder
        )
    return strong_redecode_module.StrongRedecode(
        engine, shape, transfers, decode_queue, on_strong_decoded
    )


class _LateWiring:
    """The facade and the strong redecode, for the components built first.

    The courier tells the facade when a boundary landed; the committer
    tells the strong redecode which weak result to escalate and when a
    weak window committed.
    """

    def __init__(self) -> None:
        self.window_manager = None
        self.strong_redecode = None

    def accept_boundary(self, key: tuple, is_unblocked: bool) -> None:
        self.window_manager.accept_boundary(key, is_unblocked)

    def escalate(self, job: message.DecodeJob) -> None:
        self.strong_redecode.escalate(job)

    def submit_if_far_boundary_committed(self, key: tuple) -> None:
        self.strong_redecode.submit_if_far_boundary_committed(key)


def _decoder_manager(
    engine: engine_module.Engine,
    settings: MachineSettings,
    escalation_policy,
    pool: _DecoderPool,
    links,
) -> decoder_manager_module.DecoderManager:
    return decoder_manager_module.DecoderManager(
        engine,
        link=links,
        router=pool.router,
        scheduler=pool.scheduler,
        unit_pools=pool.unit_pools,
        bulk_strong=settings.decoder_manager.bulk_strong,
        decoder_memory=pool.decoder_memory,
        escalation_policy=escalation_policy,
    )


def _trace_writer(
    observation: observe_settings.ObservationSettings,
    engine: engine_module.Engine,
    settings: MachineSettings,
) -> Optional[trace_writer_module.TraceWriter]:
    """The Chrome trace writer, only when the section names a path."""
    if not observation.writes_trace:
        return None
    name = f"decsim {settings.workload.kind} {settings.workload.code_task}"
    return trace_writer_module.TraceWriter(engine, name)


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
    pool: "_DecoderPool",
) -> None:
    """Hand the trace and the counters every source of the data path.

    The hops and residences of docs/rewrite/notes/data_path.md sections
    3 and 4, each from the component where it happens; a run with
    neither listener connects nothing and fires into empty lists.
    """
    if data_movement is not None:
        qpu.round_emitted.connect(data_movement.round_emitted)
        links.transfer_delivered.connect(data_movement.transfer_delivered)
        _connect_store_counts(data_movement, round_store)
        if strong_round_store is not None:
            _connect_store_counts(data_movement, strong_round_store)
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
    qpu.round_emitted.connect(trace_writer.round_emitted)
    qpu.command_event.connect(trace_writer.command_event)
    links.transfer_delivered.connect(trace_writer.transfer_delivered)
    for source in _copy_sources(
        controller,
        assembler,
        round_writer,
        strong_round_writer,
        decoder_manager,
        window_manager,
    ):
        source.connect(trace_writer.copy_made)
    _connect_store_trace(trace_writer, round_store, "Buffer 0")
    if strong_round_store is not None:
        _connect_store_trace(trace_writer, strong_round_store, "Buffer 1")
    _connect_decoder_trace(trace_writer, decoder_manager, pool)
    _connect_window_trace(trace_writer, window_manager, decoder_manager)
    if pauli_frame is not None:
        pauli_frame.correction_accepted.connect(
            trace_writer.correction_accepted
        )
        pauli_frame.correction_committed.connect(
            trace_writer.correction_committed
        )


def _connect_store_counts(
    data_movement: data_movement_module.DataMovement, store
) -> None:
    """One store's references: registered, transferred and released."""
    store.hold_registered.connect(data_movement.hold_registered)
    store.hold_transferred.connect(data_movement.hold_transferred)
    store.hold_released.connect(data_movement.hold_released)


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
        controller.copy_made,
        assembler.copy_made,
        round_writer.copy_made,
    ]
    if strong_round_writer is not None:
        sources.append(strong_round_writer.copy_made)
    sources.append(decoder_manager.service.staging.copy_made)
    sources.append(window_manager.requester.builder.copy_made)
    return sources


def _connect_store_trace(
    trace_writer: trace_writer_module.TraceWriter, store, store_name: str
) -> None:
    """One store's residences, its occupancy and its holds."""
    capacity = store.settings.rounds
    stored = functools.partial(trace_writer.round_stored, store_name, capacity)
    store.round_stored.connect(stored)
    published = functools.partial(trace_writer.round_published, store_name)
    store.round_published.connect(published)
    released = functools.partial(trace_writer.round_released, store_name)
    store.round_released.connect(released)
    registered = functools.partial(trace_writer.hold_registered, store_name)
    store.hold_registered.connect(registered)
    transferred = functools.partial(trace_writer.hold_transferred, store_name)
    store.hold_transferred.connect(transferred)
    hold_released = functools.partial(trace_writer.hold_released, store_name)
    store.hold_released.connect(hold_released)


def _connect_decoder_trace(
    trace_writer: trace_writer_module.TraceWriter,
    decoder_manager,
    pool: "_DecoderPool",
) -> None:
    """The ready queue, the units' services, their memories and stages."""
    queue = decoder_manager.queue
    queue.job_enqueued.connect(trace_writer.job_enqueued)
    queue.depth_changed.connect(trace_writer.depth_changed)
    service = decoder_manager.service
    service.job_dispatched.connect(trace_writer.job_dispatched)
    service.input_landed.connect(trace_writer.input_landed)
    service.job_started.connect(trace_writer.job_started)
    service.job_finished.connect(trace_writer.job_finished)
    for unit in decoder_manager.pool.units():
        taken = functools.partial(trace_writer.memory_taken, unit.name)
        unit.memory.taken.connect(taken)
    for decoder in _staged_decoders(pool):
        decoder.stage_recorded.connect(trace_writer.stage_recorded)


def _staged_decoders(pool: "_DecoderPool") -> list:
    """Every routed decoder that reports its internal stages.

    The router's seed children are the tiers and the per-code rows; a
    row that wraps another (the confidence and check wrappers) reports
    its own children the same way, so the walk finds a staged decoder
    wherever it sits.
    """
    found = []
    seen = set()
    pending = [pool.router]
    while pending:
        decoder = pending.pop()
        if decoder is None or id(decoder) in seen:
            continue
        identity = id(decoder)
        seen.add(identity)
        if hasattr(decoder, "stage_recorded"):
            found.append(decoder)
        children = getattr(decoder, "run_seed_children", None)
        if children is None:
            continue
        for child in children():
            pending.append(child.child)
    return found


def _connect_window_trace(
    trace_writer: trace_writer_module.TraceWriter,
    window_manager,
    decoder_manager,
) -> None:
    """The windows a stream lays, their verdicts, commits and absorptions."""
    window_manager.planner.window_planned.connect(trace_writer.window_planned)
    window_manager.requester.committer.window_committed.connect(
        trace_writer.window_committed
    )
    decoder_manager.outcomes.verdict_given.connect(trace_writer.verdict_given)
    strong_redecode = window_manager.strong_redecode
    shape = getattr(strong_redecode, "shape", None)
    absorbed = getattr(shape, "window_absorbed", None)
    if absorbed is not None:
        absorbed.connect(trace_writer.window_absorbed)


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
    outcomes.request_ended.connect(decode_records.request_ended)
    outcomes.service_ended.connect(decode_records.service_ended)


def _connect_runtime_stamps(
    execution_runtime: execution_runtime_module.ExecutionRuntime,
    stamps: runtime_stamps_module.RuntimeStamps,
) -> None:
    """The stamps hear every tick of an operation's life."""
    execution_runtime.operation_issued.connect(stamps.operation_issued)
    execution_runtime.operation_started.connect(stamps.operation_started)
    execution_runtime.body_finished.connect(stamps.body_finished)
    execution_runtime.decode_released.connect(stamps.decode_released)
    execution_runtime.result_returned.connect(stamps.result_returned)


def _check_strong_route(settings: MachineSettings, router) -> None:
    """A switching run routes a strong job away from the weak decoder.

    Run once on probe jobs: one decoder for both hints is the user's
    mistake.
    """
    if settings.escalation.kind != "switching":
        return
    weak_probe = message.DecodeJob(op_id=-1, window_id=0, n_rounds=0)
    strong_probe = message.DecodeJob(
        op_id=-1, window_id=0, n_rounds=0, hint="strong"
    )
    strong_decoder = router.route(strong_probe)
    weak_decoder = router.route(weak_probe)
    if strong_decoder is weak_decoder:
        raise ValueError(
            "the strong tier routes to the same decoder as the weak tier; "
            "give strong_decoder its own kind, or a router that sends "
            "hint 'strong' to a distinct decoder"
        )


def _factory(
    settings: qpu_settings.FactorySettings,
    engine: engine_module.Engine,
    decoder_manager,
    plan: _Plan,
):
    """The factory of the kind.

    A distillation row decodes its corrections on the run's decoder
    manager; the multi-level row paces its levels on the run's round.
    """
    row = _row(MAGIC_STATE_FACTORIES, "magic_state_factory.kind", settings.kind)
    if row is magic_state_factories.InfiniteFactory:
        return row(engine)
    if row is magic_state_factories.MultiLevelDistillationFactory:
        return row(
            engine,
            decode_service=decoder_manager,
            round_ticks=plan.round_ticks,
            **settings.arguments,
        )
    return row(engine, decode_service=decoder_manager, **settings.arguments)


def _feedback_streams(
    engine: engine_module.Engine,
    plan: _Plan,
    qpu,
    window_manager,
    *,
    retry_ready_operations,
):
    """The stream bookkeeping, only when the workload has feedback."""
    uses_streams = bool(plan.protected_regions)
    for operation in plan.all_operations:
        if operation.stream_id is not None:
            uses_streams = True
    if not uses_streams:
        return feedback_streams.NoFeedbackStreams()
    run_plan = plan.run_plan
    return feedback_streams.FeedbackStreams(
        engine,
        qpu=qpu,
        window_manager=window_manager,
        regions=plan.protected_regions,
        resolved_operations=run_plan.resolved_operations,
        resolved_patches=run_plan.resolved_patches,
        retry_ready_operations=retry_ready_operations,
    )


def _patch_by_identity(plan: _Plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity


# ------------------------------------------------------- the listeners


def _connect_round_events(
    round_events: round_events_module.RoundEventRecorder,
    controller,
    assembler,
    held_rounds,
    round_writer,
    transmitter,
    instruction_output,
) -> None:
    """The flight recorder hears every round event and controller output."""
    for component in (
        controller,
        assembler,
        held_rounds,
        round_writer,
        transmitter,
    ):
        component.round_event.connect(round_events.record)
    instruction_output.output_event.connect(round_events.output)


def _connect_stage_records(
    pool: "_DecoderPool",
) -> stage_records_module.StageLedger:
    """The run's stage history, heard from every routed decoder."""
    stages = stage_records_module.StageLedger()
    for decoder in _staged_decoders(pool):
        decoder.stage_recorded.connect(stages.stage_recorded)
    return stages


def _frame_corrections(
    pauli_frame,
) -> flight_recorder_module.FrameCorrections:
    """The frame's accepted and landed corrections, for the recorder."""
    corrections = flight_recorder_module.FrameCorrections()
    if pauli_frame is None:
        return corrections
    pauli_frame.correction_accepted.connect(corrections.correction_accepted)
    pauli_frame.correction_committed.connect(corrections.correction_committed)
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
    pool.unit_busy.connect(utilization.unit_busy)
    pool.unit_freed.connect(utilization.unit_freed)
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
        unit.memory.deposited.connect(deposited)
        taken = functools.partial(occupancy.taken, unit.name)
        unit.memory.taken.connect(taken)
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


def _observation(
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
    round_events,
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
        trace_writer=trace_writer,
        data_movement=data_movement,
        decode_records=decode_records,
        runtime_stamps=runtime_stamps,
        queue_depth=queue_depth,
        controller_counters=controller_counters,
        command_events=command_events,
        stages=stages,
        decode_backlog=decode_backlog,
        decoder_utilization=decoder_utilization,
        decoder_memory_occupancy=decoder_memory_occupancy,
    )


def _seed_roots(**parts) -> tuple:
    """The seed path of every stochastic owner; the segments are results."""
    roots = []
    for name, value in parts.items():
        path = (message.RunSeedPathSegment("field", name),)
        roots.append((path, value))
    return tuple(roots)


def _load_program(
    plan: _Plan,
    conditional_release,
    window_manager,
    streams,
    idle_rounds,
    execution_runtime,
) -> None:
    """Register the workload with every component that reads it.

    The streams first, then every operation with the windows, then the
    idle accounting, then the runtime starts the roots.
    """
    for operation in plan.operations:
        if operation.blocked_by is not None:
            conditional_release.register_blocked_operation(
                operation.id, operation.blocked_by
            )
    for operation in plan.planned_operations:
        window_manager.register_operation(operation)
    run_plan = plan.run_plan
    window_manager.install_planned_holds(run_plan.buffering)
    for stream in plan.dynamic_streams:
        window_manager.register_stream(stream)
    program = message.ExecutionProgram(
        plan.operations,
        plan.decode_operations,
        plan.dynamic_streams,
        plan.protected_regions,
    )
    streams.load(program)
    for operation in program.operations:
        window_manager.register_operation(operation)
    idle_rounds.load(program)
    execution_runtime.load_program(program)


# ------------------------------------------------------------ the result


def _capture_result(machine: Machine) -> RunResult:
    """Project the finished components into the immutable run result."""
    engine = machine.engine
    execution_runtime = machine.execution_runtime
    if not engine.idle or not execution_runtime.workload_complete:
        raise RuntimeError("primary run ended before workload completed")
    truth_for = getattr(
        machine.syndrome_source, "logical_observable_truth", None
    )
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
    return RunResult(
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
) -> LogicalOperationResult:
    """One operation's predicted observables beside the sampled truth."""
    operation_id = operation.id
    logical = machine.observation.results.observables_for(operation_id)
    bits = None
    status = "no_logical_output"
    if logical is not None:
        bits = tuple(logical)
        status = "logical_observables"
    actual = None
    if truth_for is not None:
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
    return LogicalOperationResult(
        operation_id, status, bits, stream_offset, actual, failure
    )
