"""Build one controller-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py).
"""

from typing import Optional

import decsim.config as config
import decsim.controller.conditional_release as conditional_release_module
import decsim.controller.controller as controller_module
import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.idle_rounds as idle_rounds_module
import decsim.controller.instruction_output as instruction_output_module
import decsim.controller.operation_issue as operation_issue
import decsim.controller.round_assembly as round_assembly
import decsim.controller.round_transmission as round_transmission
import decsim.controller.round_writes as round_writes
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.frontends.execution_runtime as execution_runtime_module
import decsim.pauli_frame.decision_dispatch as decision_dispatch_module
import decsim.ports as ports
import decsim.qpu.cycle_clock as cycle_clock
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import decsim.tables as tables


def build_conditional_release(parts):
    """The gate a conditional operation's release waits at."""
    return conditional_release_module.ConditionalRelease(parts.engine)


def build_decoder_manager(parts):
    """The decoder side's manager: the pool's router, scheduler and units."""
    pool = parts.pool
    settings = parts.settings.decoder_manager
    return decoder_manager_module.DecoderManager(
        parts.engine,
        router=pool.router,
        scheduler=pool.scheduler,
        unit_pools=pool.unit_pools,
        bulk_strong=settings.bulk_strong,
        decoder_memory=pool.decoder_memory,
        escalation_policy=parts.escalation_policy,
        clock=settings.clock,
        dispatch_cycles=settings.dispatch_cycles,
        copies_input_by_pool=pool.copies_input_by_pool,
        blocks_unit_by_pool=pool.blocks_unit_by_pool,
        formation_by_pool=pool.formation_by_pool,
    )


def build_detection_events(
    settings: machine_settings.MachineSettings, device
) -> ports.DetectionEventPlacement:
    """Where this machine forms its detection events, as one component.

    controller.detection_events_formed_at names the row; a value that is
    not one is refused here, before the first round is packed. Both rows
    are built the same way, with the run's former and the controller's
    own formation cost in cycles. A source that does not answer the
    DetectionEventFormer port forms nothing either way. The decoder pool
    is compiled from this seat, so the root builds it before the rest.
    """
    former = None
    if isinstance(device, ports.DetectionEventFormer):
        former = device
    row = tables.row(
        controller_settings.DETECTION_EVENT_FORMATION,
        "controller.detection_events_formed_at",
        settings.controller.detection_events_formed_at,
    )
    detection_event_formation_cycles = (
        settings.controller.detection_event_cycles_per_round
    )
    return row(former, detection_event_formation_cycles)


def build_factory(parts):
    """The factory of the kind, from the one collaborators record.

    A distillation row decodes its corrections on the run's decoder
    manager; the multi-level row paces its levels on the run's round; a
    row that needs neither reads the engine alone. The card refuses an
    ambiguous decode service as it is read, so the decode service is the
    one neighbour a seat still takes at construction and this row sits
    after the decoder manager's.
    """
    settings = parts.settings.magic_state_factory
    row = tables.row(
        qpu_settings.MAGIC_STATE_FACTORIES,
        "magic_state_factory.kind",
        settings.kind,
    )
    collaborators = magic_state_factories.FactoryCollaborators(
        engine=parts.engine,
        decode_service=parts.seats["decoder_manager"],
        round_ticks=parts.plan.round_ticks,
        arguments=settings.arguments,
    )
    return row(collaborators)


def build_transmitter(parts):
    """The controller's end of the route a written round travels."""
    return round_transmission.RoundTransmitter(parts.engine)


def build_round_writer(parts):
    """The writer that puts a finished round into every store it reaches."""
    return round_writes.RoundWriter(parts.engine)


def build_rounds_in_flight(parts):
    """The packing stage's bound on the rounds it has in flight."""
    return round_assembly.RoundsInFlight(
        parts.settings.controller.packing_rounds_in_flight
    )


def build_assembler(parts):
    """The packing stage: fragments in, one packed round out."""
    return round_assembly.RoundAssembler(
        parts.engine, parts.settings.controller
    )


def build_qpu(parts):
    """The device on its cycle clock, one QEC round per cycle."""
    cycle_clock_domain = config.Clock(parts.plan.round_ticks)
    return cycle_clock.QPUDevice(
        parts.engine, parts.plan.device, cycle_clock_domain
    )


def build_instruction_output(parts):
    """The controller's output path to the QPU."""
    settings = parts.settings.controller
    return instruction_output_module.InstructionOutput(
        parts.engine, settings.clock, settings.decision_to_pulse_cycles
    )


def build_feedback_streams(parts):
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


def build_idle_rounds(parts):
    """The idle accounting, over the resolved patches it charges."""
    patch_by_identity = resolved_patches_by_identity(parts.plan)
    return idle_rounds_module.IdleRoundAccounting(
        parts.plan.idle_policy, patch_by_identity
    )


def build_issuer(parts):
    """The issuer that turns one admitted operation into a command."""
    resolved_operations = parts.plan.run_plan.resolved_operations
    return operation_issue.OperationIssuer(parts.engine, resolved_operations)


def build_controller(parts):
    """The controller's readout intake."""
    return controller_module.Controller(parts.engine, parts.settings.controller)


def build_execution_runtime(parts):
    """The runtime that drives every operation's lifecycle."""
    return execution_runtime_module.ExecutionRuntime(
        parts.engine, parts.plan.resource_claims
    )


def build_decision_dispatch(parts):
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
    """The point the trace is of, so two files are told apart at a glance."""
    kind = settings.escalation.kind
    distance = settings.qpu.distance
    probability = settings.workload.physical_error_probability
    return f"decsim {kind} d{distance} p{probability} seed{seed}"


def check_strong_route(escalation_policy, router) -> None:
    """A run that may escalate routes a strong job away from the weak one.

    Run once on probe jobs: one decoder for both job kinds is the
    user's mistake.
    """
    if not escalation_policy.requires_strong_context:
        return
    weak_probe = decoding_records.DecodeJob(
        operation_id=-1, window_id=0, round_count=0
    )
    strong_probe = decoding_records.DecodeJob(
        operation_id=-1,
        window_id=0,
        round_count=0,
        kind=decoding_records.DecodeJobKind.STRONG_REDECODE,
    )
    strong_decoder = router.route(strong_probe)
    weak_decoder = router.route(weak_probe)
    if strong_decoder is weak_decoder:
        raise ValueError(
            "the strong tier routes to the same decoder as the weak tier; "
            "give strong_decoder its own kind, or a router that sends a "
            "strong re-decode to a distinct decoder"
        )


def resolved_patches_by_identity(plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity
