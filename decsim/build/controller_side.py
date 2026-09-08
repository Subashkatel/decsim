"""Build the decoder manager, the factory and the controller's streams."""

from typing import Optional

import decsim.build.decoders as decoder_build
import decsim.build.plan as plan_build
import decsim.controller.feedback_streams as feedback_streams
import decsim.decoders.decoder_manager as decoder_manager_module
import decsim.engine as engine_module
import decsim.qpu.magic_state_factories as magic_state_factories
import decsim.qpu.settings as qpu_settings
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings


def build_decoder_manager(
    engine: engine_module.Engine,
    settings: machine_settings.MachineSettings,
    escalation_policy,
    pool: decoder_build.DecoderPool,
) -> decoder_manager_module.DecoderManager:
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
    )


def process_name(
    settings: machine_settings.MachineSettings, seed: Optional[int]
) -> str:
    """The point the trace is of, so two files are told apart at a glance."""
    kind = settings.escalation.kind
    distance = settings.qpu.distance
    probability = settings.workload.physical_error_probability
    return f"decsim {kind} d{distance} p{probability} seed{seed}"


def check_strong_route(
    settings: machine_settings.MachineSettings, router
) -> None:
    """A switching run routes a strong job away from the weak decoder.

    Run once on probe jobs: one decoder for both job kinds is the
    user's mistake.
    """
    if settings.escalation.kind != "switching":
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
    """The factory of the kind.

    A distillation row decodes its corrections on the run's decoder
    manager; the multi-level row paces its levels on the run's round.
    """
    row = machine_settings.row(
        qpu_settings.MAGIC_STATE_FACTORIES,
        "magic_state_factory.kind",
        settings.kind,
    )
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


def _patch_by_identity(plan: plan_build.Plan) -> dict:
    """The resolved patches by identity, for the idle accounting."""
    patch_by_identity = {}
    for patch in plan.run_plan.resolved_patches:
        patch_by_identity[patch.patch_identity] = patch
    return patch_by_identity
