"""Build the decoder manager, the factory and the controller's streams."""

from typing import Optional

import decsim.build.decoders as decoder_build
import decsim.build.plan as plan_build
import decsim.controller.feedback_streams as feedback_streams
import decsim.controller.round_assembly as round_assembly
import decsim.controller.settings as controller_settings
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import decsim.tables as tables


def build_decoder_manager(
    engine: engine_module.Engine,
    settings: machine_settings.MachineSettings,
    escalation_policy,
    pool: decoder_build.DecoderPool,
) -> decoder_manager_module.DecoderManager:
    """The decoder side's manager: the pool's router, scheduler and units."""
    dispatch_ticks = settings.decoder_manager.dispatch_ticks()
    return decoder_manager_module.DecoderManager(
        engine,
        router=pool.router,
        scheduler=pool.scheduler,
        unit_pools=pool.unit_pools,
        bulk_strong=settings.decoder_manager.bulk_strong,
        decoder_memory=pool.decoder_memory,
        escalation_policy=escalation_policy,
        dispatch_ticks=dispatch_ticks,
        copies_input_by_pool=pool.copies_input_by_pool,
        blocks_unit_by_pool=pool.blocks_unit_by_pool,
    )


def build_detection_events(
    settings: machine_settings.MachineSettings, device
) -> round_assembly.DetectionEventFormation:
    """The device's formation table and where this machine forms events.

    controller.detection_events_formed_at names the row; a value that is
    not one is refused here, before the first round is packed. A source
    with no formation table forms nothing either way.
    """
    form_round = getattr(device, "form_round", None)
    at_the_controller = tables.row(
        controller_settings.DETECTION_EVENT_FORMATION,
        "controller.detection_events_formed_at",
        settings.controller.detection_events_formed_at,
    )
    return round_assembly.DetectionEventFormation(form_round, at_the_controller)


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


def build_factory(
    settings: qpu_settings.FactorySettings,
    engine: engine_module.Engine,
    decoder_manager,
    plan: plan_build.Plan,
):
    """The factory of the kind, from the one collaborators record.

    A distillation row decodes its corrections on the run's decoder
    manager; the multi-level row paces its levels on the run's round; a
    row that needs neither reads the engine alone.
    """
    row = tables.row(
        qpu_settings.MAGIC_STATE_FACTORIES,
        "magic_state_factory.kind",
        settings.kind,
    )
    collaborators = magic_state_factories.FactoryCollaborators(
        engine=engine,
        decode_service=decoder_manager,
        round_ticks=plan.round_ticks,
        arguments=settings.arguments,
    )
    return row(collaborators)


def build_feedback_streams(
    engine: engine_module.Engine,
    plan: plan_build.Plan,
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


def resolved_patches_by_identity(plan: plan_build.Plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity
