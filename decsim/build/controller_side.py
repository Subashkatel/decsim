"""Build one controller-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py).
"""

from typing import Optional, Union

import decsim.build.parts as build_parts
import decsim.config as config
import decsim.controller.conditional_release as conditional_release_module
import decsim.controller.controller as controller_module
import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.syndrome_round_sender as syndrome_round_sender
import decsim.decoders.decode_queue as decode_queue
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.decoders.strong_requests as strong_requests_module
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.settings as event_settings
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.pauli_frame.decision_dispatch as decision_dispatch_module
import decsim.ports as ports
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.tables as tables


def build_conditional_release(
    parts: build_parts.Parts,
) -> conditional_release_module.ConditionalRelease:
    """The gate a conditional operation's release waits at."""
    del parts
    return conditional_release_module.ConditionalRelease()


def build_strong_requests(
    parts: build_parts.Parts,
) -> strong_requests_module.StrongRequests:
    """The ledger of strong requests, which both sides' managers take."""
    del parts
    return strong_requests_module.StrongRequests()


def build_decoder_manager(
    parts: build_parts.Parts,
) -> decoder_manager_module.DecoderManager:
    """The chip's decoder manager: every pool but the strong one.

    The strong pool is the host's manager's, so this one never holds a
    strong job; the ledger is the seat both managers are wired to.
    """
    unit_pools = {}
    for name, units in parts.pool.unit_pools.items():
        if name != decode_queue.STRONG_POOL:
            unit_pools[name] = units
    return _decoder_manager(parts, unit_pools)


def build_strong_decoder_manager(
    parts: build_parts.Parts,
) -> decoder_manager_module.DecoderManager:
    """The host's decoder manager: the strong pool alone.

    The same class as the chip's, over the strong units, with its own
    ready queue, staging and outcomes (LATTE 2509.03954 lines 705-720,
    the host's scheduler owns the decode queue and the thread pool).
    The escalation side and the requester's strong sibling submit here.
    """
    units = parts.pool.unit_pools[decode_queue.STRONG_POOL]
    unit_pools = {decode_queue.STRONG_POOL: units}
    return _decoder_manager(parts, unit_pools)


def build_detection_events(
    settings: machine_settings.MachineSettings,
    device,
    escalation_policy: ports.EscalationPolicy,
    burst_detector: Optional[ports.BurstDetector] = None,
) -> ports.DetectionEventPlacement:
    """Where this machine forms its detection events, as one component.

    Every path a round of this run takes to a decoder crosses exactly
    one seat of detection_events.formed_at: a path with none would
    decode raw outcomes, and a path with two would form events of
    events, a wrong answer either way. A source that does not answer
    the DetectionEventFormer port forms nothing. A burst detector
    counts the rounds the first seat on the primary tier's path forms.
    The decoder pool is compiled from this seat, so the root builds it
    before the rest.
    """
    formed_at = settings.detection_events.formed_at
    paths = _paths_of_the_run(escalation_policy)
    for path in paths:
        _check_one_seat_on(path, formed_at)
    source = None
    if isinstance(device, ports.DetectionEventFormer):
        source = device
    observed_seat = None
    if burst_detector is not None:
        _check_forms_events(source)
        observed_seat = _first_seat_on(paths[0], formed_at)
    return formation.SeatedFormation(
        source, settings.detection_events, observed_seat, burst_detector
    )


def build_factory(parts: build_parts.Parts) -> ports.MagicStateFactory:
    """The factory of the kind, from the one collaborators record.

    Every row is handed the run's engine and round: the multi-level row
    paces its levels on the round, and a row ignores what its card does
    not use. The decoder manager a card's correction decodes go to is
    bound later, as the row's decode_queue port.
    """
    settings = parts.settings.magic_state_factory
    row = tables.row(
        qpu_settings.MAGIC_STATE_FACTORIES,
        "magic_state_factory.kind",
        settings.kind,
    )
    collaborators = magic_state_factories.FactoryCollaborators(
        engine=parts.engine,
        round_ticks=parts.plan.round_ticks,
        settings=settings.row_settings,
    )
    return row(collaborators)


def build_transmitter(
    parts: build_parts.Parts,
) -> round_transmission.RoundTransmitter:
    """The controller's end of the route a written round travels."""
    return round_transmission.RoundTransmitter(parts.engine)


def build_syndrome_round_sender(
    parts: build_parts.Parts,
) -> syndrome_round_sender.SyndromeRoundSender:
    """The writer that puts a finished round into every store it reaches."""
    return syndrome_round_sender.SyndromeRoundSender(parts.engine)


def build_rounds_in_flight(
    parts: build_parts.Parts,
) -> round_assembly.RoundsInFlight:
    """The packing stage's bound on the rounds it has in flight."""
    return round_assembly.RoundsInFlight(
        parts.settings.controller.packing_rounds_in_flight
    )


def build_assembler(parts: build_parts.Parts) -> round_assembly.RoundAssembler:
    """The packing stage: fragments in, one packed round out."""
    return round_assembly.RoundAssembler(
        parts.engine, parts.settings.controller
    )


def build_qpu(parts: build_parts.Parts) -> cycle_clock.QPUDevice:
    """The device on its cycle clock, one QEC round per cycle."""
    cycle_clock_domain = config.Clock(parts.plan.round_ticks)
    return cycle_clock.QPUDevice(
        parts.engine, cycle_clock_domain, parts.plan.code
    )


def build_instruction_output(
    parts: build_parts.Parts,
) -> instruction_output_module.InstructionOutput:
    """The controller's output path to the QPU."""
    settings = parts.settings.controller
    return instruction_output_module.InstructionOutput(
        parts.engine, settings.clock, settings.decision_to_pulse_cycles
    )


def build_feedback_streams(
    parts: build_parts.Parts,
) -> Union[
    feedback_streams.NoFeedbackStreams, feedback_streams.FeedbackStreams
]:
    """The stream bookkeeping, only when the workload has feedback."""
    if not has_feedback(parts.plan):
        return feedback_streams.NoFeedbackStreams()
    run_plan = parts.plan.run_plan
    return feedback_streams.FeedbackStreams(
        parts.engine,
        regions=parts.plan.protected_regions,
        resolved_operations=run_plan.resolved_operations,
        resolved_patches=run_plan.resolved_patches,
    )


def build_idle_rounds(
    parts: build_parts.Parts,
) -> idle_rounds_module.IdleRoundAccounting:
    """The idle accounting, over the resolved patches it charges."""
    patch_by_identity = resolved_patches_by_identity(parts.plan)
    return idle_rounds_module.IdleRoundAccounting(
        parts.plan.idle_policy, patch_by_identity
    )


def build_issuer(parts: build_parts.Parts) -> operation_issue.OperationIssuer:
    """The issuer that turns one admitted operation into a command."""
    resolved_operations = parts.plan.run_plan.resolved_operations
    return operation_issue.OperationIssuer(parts.engine, resolved_operations)


def build_controller(parts: build_parts.Parts) -> controller_module.Controller:
    """The controller's readout intake."""
    return controller_module.Controller(parts.engine, parts.settings.controller)


def build_execution_runtime(
    parts: build_parts.Parts,
) -> execution_runtime_module.ExecutionRuntime:
    """The runtime that drives every operation's lifecycle."""
    return execution_runtime_module.ExecutionRuntime(
        parts.engine, parts.plan.resource_claims
    )


def build_decision_dispatch(
    parts: build_parts.Parts,
) -> decision_dispatch_module.DecisionDispatch:
    """The frame's end of the path a decision takes to the controller."""
    return decision_dispatch_module.DecisionDispatch(parts.engine)


def has_feedback(plan) -> bool:
    """Whether any operation shares a stream or declares a region."""
    if plan.protected_regions:
        return True
    for operation in plan.all_operations:
        if operation.stream_id is not None:
            return True
    return False


def process_name(
    settings: machine_settings.MachineSettings, seed: Optional[int]
) -> str:
    """The machine the trace is of: its escalation, code distance and seed.

    The machine knows no sweep, so the point's other values name the
    trace's file (experiments/measure.py shot_label) and not this line.
    """
    kind = settings.escalation.kind
    distance = settings.qpu.distance
    return f"decsim {kind} d{distance} seed{seed}"


def resolved_patches_by_identity(plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity


def _decoder_manager(
    parts: build_parts.Parts, unit_pools: dict
) -> decoder_manager_module.DecoderManager:
    """One manager over the named pools, on the run's one manager card."""
    pool = parts.pool
    settings = parts.settings.decoder_manager
    scheduler = settings.scheduler()
    return decoder_manager_module.DecoderManager(
        parts.engine,
        scheduler=scheduler,
        unit_pools=unit_pools,
        bulk_strong=settings.bulk_strong,
        decoder_memory=pool.decoder_memory,
        clock=settings.clock,
        dispatch_cycles=settings.dispatch_cycles,
        copies_input_by_pool=pool.copies_input_by_pool,
        blocks_unit_by_pool=pool.blocks_unit_by_pool,
        formation_by_pool=pool.formation_by_pool,
    )


def _check_forms_events(source) -> None:
    """A burst detector counts detection events, so the source forms some."""
    if source is not None:
        return
    raise ValueError(
        "burst_detector counts detection events, and this qpu.kind "
        "forms none; name a source that forms them, or write "
        "burst_detector: {kind: none}"
    )


def _paths_of_the_run(escalation_policy: ports.EscalationPolicy) -> tuple:
    """The paths this run's rounds take to a decoder, the primary first."""
    strong = window_records.DecoderTier.STRONG
    if escalation_policy.primary_tier is strong:
        return (event_settings.STRONG_PATH,)
    if escalation_policy.requires_strong_context:
        return (event_settings.WEAK_PATH, event_settings.ESCALATION_PATH)
    return (event_settings.WEAK_PATH,)


def _check_one_seat_on(path: tuple, formed_at: tuple) -> None:
    """The path crosses exactly one seat that forms, or the run is refused."""
    seats = []
    for seat in path:
        if seat in formed_at:
            seats.append(seat)
    if len(seats) == 1:
        return
    listed = ", ".join(path)
    raise ValueError(
        f"detection_events.formed_at {list(formed_at)} forms the rounds "
        f"on the path {listed} at {seats}: every path to a decoder "
        "crosses exactly one seat, since a path with none decodes raw "
        "outcomes and a path with two forms events of events"
    )


def _first_seat_on(path: tuple, formed_at: tuple) -> str:
    """The seat of formed_at the path crosses; the path crosses one."""
    for seat in path:
        if seat in formed_at:
            return seat
    raise AssertionError("every path crosses one seat")
