"""Build the window manager and everything behind its facade.

The facade's nine components are wired by constructor here, in the order
a window meets them; the courier's and the committer's callbacks to
components built after them arrive through one stand-in, _LateWiring,
which is the only late bind in decsim.
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
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.links.window_transfers as window_transfers_module
import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.settings as machine_settings
import decsim.syndrome_buffer.round_output as round_output
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
    fault_model_requirement_for,
    round_store,
    strong_round_store,
    pauli_frame,
    decode_queue,
    on_workload_complete,
) -> window_manager_module.WindowManager:
    """The windows facade over its six components, wired by constructor."""
    run_plan = plan.run_plan
    built_models = settings.workload.built_models
    if built_models is None:
        built_models = built_window_models.BuiltWindowModels()
    models = window_planner_module.WindowModels(
        plan.error_model_provider, fault_model_requirement_for, built_models
    )
    planner = window_planner_module.WindowPlanner(
        plan.scheme,
        run_plan.resolved_operations,
        run_plan.execution,
        models,
        plan.planned_operations,
    )
    tracker = round_tracker_module.RoundTracker(plan.scheme, planner)
    retention = round_retention_module.RoundRetention(
        round_store,
        strong_round_store,
        planner,
        tracker,
        is_strong_context_retained=escalation_policy.requires_strong_context,
        primary_tier=escalation_policy.primary_tier,
    )
    transfers = window_transfers_module.WindowTransfers(engine, links)
    decoder_output = decoder_output_module.DecoderOutput(transfers, pauli_frame)
    weak_output = round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "Buffer 0",
    )
    strong_output = round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.STRONG_BUFFER_TO_STRONG_DECODER,
        "Buffer 1",
    )
    primary_output = weak_output
    if escalation_policy.primary_tier is window_records.DecoderTier.STRONG:
        primary_output = strong_output
    copies_the_fold = _copies_the_boundary_fold(settings, escalation_policy)
    gate = decode_requests.WindowInputGate(
        planner, plan.window_interaction, copies_the_fold
    )
    builder = decode_requests.DecodeRequestBuilder(
        engine, planner, tracker, plan.window_interaction, gate
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
        decoder_output,
        plan.window_interaction,
        plan.boundary_policy,
        late.accept_boundary,
    )
    committer = window_commits.WindowCommitter(
        engine, courier, decoder_output, results, committer_redecode
    )
    verdict = window_commits.WindowVerdict(
        planner,
        tracker,
        escalation_policy,
        committer_redecode,
        decode_queue,
        committer,
    )
    gap_join = _window_gap_join(settings, engine, verdict, decode_queue)
    requester = decode_requests.DecodeRequester(
        tracker,
        retention,
        builder,
        decode_queue,
        escalation_policy,
        verdict,
        primary_output,
        gap_join,
    )
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
        plan.window_interaction,
        decode_queue,
        verdict.accept_strong_result,
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


def _copies_the_boundary_fold(
    settings: machine_settings.MachineSettings, escalation_policy
) -> bool:
    """Whether the tier that decodes the plan's windows folds into a copy.

    <tier>.boundary_fold names the row; a value that is not one is
    refused here, where the tier is named.
    """
    tier = escalation_policy.primary_tier.value
    tier_settings = getattr(settings, f"{tier}_decoder")
    return tables.row(
        decoder_settings.DECODER_BOUNDARY_FOLDS,
        f"{tier}_decoder.boundary_fold",
        tier_settings.boundary_fold,
    )


def _window_gap_join(
    settings: machine_settings.MachineSettings, engine, verdict, decode_queue
):
    """The confidence join of a switching run: every solve's on_decoded.

    None when the escalation is not the root's switching policy: a
    Python-built policy brings its own decoder, which reports its own
    soft output from one decode.
    """
    if settings.escalation.kind != "switching":
        return None
    if settings.escalation.policy is not None:
        return None
    signal = escalation_build.confidence_signal(settings.escalation)
    return gap_join_module.WindowGapJoin(engine, signal, verdict, decode_queue)


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
    interaction,
    decode_queue,
    on_strong_decoded,
):
    """The window side of the strong tier, or None when never escalating.

    escalation.strong_window names the row: the forward window of Toshio
    Sec. III C, or decsim's own two-sided context.
    """
    if not escalation_policy.requires_strong_context:
        return None
    row = escalation_build.strong_window_row(escalation)
    regions = strong_regions.StrongRegions(
        planner, tracker, retention, interaction
    )
    collaborators = strong_window_shapes.StrongWindowCollaborators(
        engine=engine,
        regions=regions,
        planner=planner,
        retention=retention,
        builder=builder,
        requester=requester,
        ledger=ledger,
    )
    shape = row(collaborators)
    return strong_redecode_module.StrongRedecode(
        engine,
        shape,
        decoder_output,
        strong_output,
        decode_queue,
        on_strong_decoded,
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

    def escalate(self, job: decoding_records.DecodeJob) -> None:
        self.strong_redecode.escalate(job)

    def submit_if_commit_releases(self, key: tuple) -> None:
        self.strong_redecode.submit_if_commit_releases(key)

    def cancel_held_sibling(self, key: tuple) -> None:
        self.strong_redecode.cancel_held_sibling(key)
