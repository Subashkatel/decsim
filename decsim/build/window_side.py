"""Build the window manager and everything behind its facade.

Each component is built from its own settings and then bound to its
neighbours by assignment, in the order a window meets them, which is
gem5's script naming a component once and assigning its ports
(tmp/resources/gem5/configs/learning_gem5/part1/simple.py:68). A seat
the run does not need is not built, and no port of it is bound.
"""

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.confidence.gap_join as gap_join_module
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.links.window_transfers as window_transfers_module
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.tables as tables
import decsim.windows.built_window_models as built_window_models
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.decode_requests as decode_requests
import decsim.windows.operation_results as operation_results
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_commits as window_commits
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner_module


def build_window_manager(
    engine: engine_module.Engine,
    settings: machine_settings.MachineSettings,
    escalation_policy,
    plan: plan_build.Plan,
    *,
    links,
    conditional_release,
    router,
    input_fold,
    round_store,
    strong_round_store,
    weak_output,
    strong_output,
    pauli_frame,
    decode_queue,
    factory,
) -> window_manager_module.WindowManager:
    """The windows facade over its components, each bound to the next."""
    run_plan = plan.run_plan
    built_models = settings.workload.built_models
    if built_models is None:
        built_models = built_window_models.BuiltWindowModels()
    models = window_planner_module.WindowModels(built_models)
    if plan.error_model_provider is not None:
        models.provider = plan.error_model_provider
    models.router = router
    planner = window_planner_module.WindowPlanner(
        run_plan.resolved_operations,
        run_plan.execution,
        plan.planned_operations,
    )
    planner.scheme = plan.scheme
    planner.models = models
    planner.start()
    tracker = round_tracker_module.RoundTracker()
    tracker.scheme = plan.scheme
    tracker.planner = planner
    retention = round_retention_module.RoundRetention(
        is_strong_context_retained=escalation_policy.requires_strong_context,
        primary_tier=escalation_policy.primary_tier,
    )
    retention.weak_store = round_store
    if strong_round_store is not None:
        retention.strong_store = strong_round_store
    retention.planner = planner
    retention.tracker = tracker
    transfers = window_transfers_module.WindowTransfers(engine)
    transfers.link = links
    decoder_output = decoder_output_module.DecoderOutput()
    decoder_output.transfers = transfers
    if pauli_frame is not None:
        decoder_output.frame = pauli_frame
    primary_output = weak_output
    if escalation_policy.primary_tier is window_records.DecoderTier.STRONG:
        primary_output = strong_output
    copies_the_fold = _copies_the_boundary_fold(settings, escalation_policy)
    gate = decode_requests.WindowInputGate(copies_the_fold)
    gate.planner = planner
    gate.interaction = plan.window_interaction
    gate.input_fold = input_fold
    builder = decode_requests.DecodeRequestBuilder(engine)
    builder.planner = planner
    builder.tracker = tracker
    builder.interaction = plan.window_interaction
    builder.gate = gate
    ledger = committed_rounds.LogicalLedger()
    results = operation_results.OperationResults()
    results.planner = planner
    results.tracker = tracker
    results.retention = retention
    results.ledger = ledger
    results.conditional_release = conditional_release
    results.factory = factory
    courier = window_boundaries.BoundaryCourier()
    courier.planner = planner
    courier.transfers = transfers
    courier.interaction = plan.window_interaction
    courier.boundary_policy = plan.boundary_policy
    committer = window_commits.WindowCommitter(engine)
    committer.courier = courier
    committer.decoder_output = decoder_output
    committer.results = results
    verdict = window_commits.WindowVerdict(
        engine,
        clock=settings.escalation.clock,
        threshold_cycles=settings.escalation.threshold_cycles,
        switch_cycles=settings.escalation.switch_cycles,
    )
    verdict.planner = planner
    verdict.tracker = tracker
    verdict.escalation_policy = escalation_policy
    verdict.decode_queue = decode_queue
    verdict.committer = committer
    gap_join = _window_gap_join(settings, engine, verdict, decode_queue)
    read_cycles = settings.round_store.read_cycles
    if escalation_policy.primary_tier is window_records.DecoderTier.STRONG:
        read_cycles = 0
    requester = decode_requests.DecodeRequester(
        read_clock=settings.round_store.clock,
        read_cycles=read_cycles,
        clock=settings.windows.clock,
        decision_cycles=settings.windows.decision_cycles,
    )
    requester.tracker = tracker
    requester.retention = retention
    requester.builder = builder
    requester.decode_queue = decode_queue
    requester.escalation_policy = escalation_policy
    requester.verdict = verdict
    requester.store_output = primary_output
    if gap_join is not None:
        requester.gap_join = gap_join
    strong_redecode = _strong_redecode(
        escalation_policy,
        settings.escalation,
        engine,
        planner,
        tracker,
        retention,
        builder,
        decoder_output,
        strong_output,
        requester,
        ledger,
        courier,
        plan.window_interaction,
        decode_queue,
        verdict,
    )
    if strong_redecode is not None:
        committer.strong_redecode = strong_redecode
        verdict.strong_redecode = strong_redecode
    window_manager = window_manager_module.WindowManager(
        engine,
        feedback_boundary_mode=settings.workload.feedback_boundary_mode,
    )
    window_manager.planner = planner
    window_manager.tracker = tracker
    window_manager.retention = retention
    window_manager.requester = requester
    window_manager.courier = courier
    window_manager.results = results
    if strong_redecode is not None:
        window_manager.strong_redecode = strong_redecode
    window_manager.window_interaction = plan.window_interaction
    courier.windows = window_manager
    window_manager.start()
    return window_manager


def _copies_the_boundary_fold(
    settings: machine_settings.MachineSettings, escalation_policy
) -> bool:
    """Whether the tier that decodes the plan's windows folds into a copy.

    <tier>.boundary_fold names the row; a value that is not one is
    refused here, where the tier is named.
    """
    tier = escalation_policy.primary_tier.value
    tier_settings = settings.decoder_settings_for(tier)
    return tables.row(
        decoder_settings.DECODER_BOUNDARY_FOLDS,
        f"{tier}_decoder.boundary_fold",
        tier_settings.boundary_fold,
    )


def _window_gap_join(
    settings: machine_settings.MachineSettings, engine, verdict, decode_queue
):
    """The confidence join of an escalating run: every solve's on_decoded.

    None when the row decides on no confidence, and none for a
    Python-built policy, which brings its own decoder and reports its
    own soft output from one decode.
    """
    if settings.escalation.policy is not None:
        return None
    row = escalation_build.escalation_row(settings.escalation)
    if not row.decides_on_a_confidence:
        return None
    gap_join = gap_join_module.WindowGapJoin(engine)
    gap_join.signal = escalation_build.confidence_signal(settings.escalation)
    gap_join.verdict = verdict
    gap_join.decode_queue = decode_queue
    return gap_join


def _strong_redecode(
    escalation_policy,
    escalation: escalation_settings.EscalationSettings,
    engine,
    planner,
    tracker,
    retention,
    builder,
    decoder_output,
    strong_output,
    requester,
    ledger,
    courier,
    interaction,
    decode_queue,
    verdict,
):
    """The window side of the strong tier, or None when never escalating.

    escalation.strong_window names the row: the forward window of Toshio
    Sec. III C, or decsim's own two-sided context.
    """
    if not escalation_policy.requires_strong_context:
        return None
    row = escalation_build.strong_window_row(escalation)
    regions = strong_regions.StrongRegions(interaction)
    regions.planner = planner
    regions.tracker = tracker
    regions.retention = retention
    shape = row(engine)
    shape.regions = regions
    shape.planner = planner
    shape.retention = retention
    shape.builder = builder
    shape.requester = requester
    shape.ledger = ledger
    shape.courier = courier
    strong_redecode = strong_redecode_module.StrongRedecode(engine)
    strong_redecode.shape = shape
    strong_redecode.decoder_output = decoder_output
    strong_redecode.strong_output = strong_output
    strong_redecode.decode_queue = decode_queue
    strong_redecode.verdict = verdict
    return strong_redecode
