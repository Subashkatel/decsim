"""Build one window-side seat each, from the run's settings.

Every function here reads the settings its seat needs and returns the
seat; which seats a run has and what each is wired to is the assembly
file's (decsim/assembly.py), which is gem5's script naming a component
once and assigning its ports
(tmp/resources/gem5/configs/learning_gem5/part1/simple.py:68).
"""

import decsim.build.escalation as escalation_build
import decsim.confidence.gap_join as gap_join_module
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.settings as decoder_settings
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


def build_models(parts):
    """The window models, warm-started from the workload's built ones."""
    built_models = parts.settings.workload.built_models
    if built_models is None:
        built_models = built_window_models.BuiltWindowModels()
    return window_planner_module.WindowModels(built_models)


def build_planner(parts):
    """Which windows exist: the plan's, and a stream's as it grows."""
    run_plan = parts.plan.run_plan
    return window_planner_module.WindowPlanner(
        run_plan.resolved_operations,
        run_plan.execution,
        parts.plan.planned_operations,
    )


def build_tracker(parts):
    """Which rounds of a window have arrived."""
    del parts
    return round_tracker_module.RoundTracker()


def build_retention(parts):
    """Which store holds a window's rounds, and for how long."""
    policy = parts.escalation_policy
    return round_retention_module.RoundRetention(
        is_strong_context_retained=policy.requires_strong_context,
        primary_tier=policy.primary_tier,
    )


def build_window_transfers(parts):
    """The window side's transfers, which execute its sends."""
    return window_transfers_module.WindowTransfers(parts.engine)


def build_decoder_output(parts):
    """Where a finished decode's correction goes."""
    del parts
    return decoder_output_module.DecoderOutput()


def build_gate(parts):
    """The window input gate, which knows whether its tier folds a copy."""
    copies_the_fold = _copies_the_boundary_fold(
        parts.settings, parts.escalation_policy
    )
    return decode_requests.WindowInputGate(copies_the_fold)


def build_request_builder(parts):
    """The builder that stamps every decode job with its ordinal."""
    return decode_requests.DecodeRequestBuilder(parts.engine)


def build_ledger(parts):
    """Which owner committed which rounds."""
    del parts
    return committed_rounds.LogicalLedger()


def build_results(parts):
    """One operation's results, gathered as its windows commit."""
    del parts
    return operation_results.OperationResults()


def build_courier(parts):
    """The courier that carries a boundary from window to window."""
    del parts
    return window_boundaries.BoundaryCourier()


def build_committer(parts):
    """The commit of one decoded window."""
    return window_commits.WindowCommitter(parts.engine)


def build_verdict(parts):
    """Whether a decoded window is accepted or escalated."""
    escalation = parts.settings.escalation
    return window_commits.WindowVerdict(
        parts.engine,
        clock=escalation.clock,
        threshold_cycles=escalation.threshold_cycles,
        switch_cycles=escalation.switch_cycles,
    )


def build_confidence_signal(parts):
    """The row the escalation's confidence is read from."""
    return escalation_build.confidence_signal(parts.settings.escalation)


def build_gap_join(parts):
    """The join of every solve of a window into one confidence."""
    return gap_join_module.WindowGapJoin(parts.engine)


def build_requester(parts):
    """The requester, whose read cost is the store its tier reads from."""
    settings = parts.settings
    read_cycles = settings.weak_syndrome_buffer.read_cycles
    if _reads_from_the_room_side(parts.escalation_policy):
        read_cycles = 0
    return decode_requests.DecodeRequester(
        read_clock=settings.weak_syndrome_buffer.clock,
        read_cycles=read_cycles,
        clock=settings.windows.clock,
        decision_cycles=settings.windows.decision_cycles,
    )


def build_regions(parts):
    """The strong region of an escalated window."""
    del parts
    return strong_regions.StrongRegions()


def build_strong_window_shape(parts):
    """The strong window of the escalation's row: forward, or two-sided.

    escalation.strong_window names the row: the forward window of Toshio
    Sec. III C, or decsim's own two-sided context.
    """
    row = escalation_build.strong_window_row(parts.settings.escalation)
    return row(parts.engine)


def build_strong_redecode(parts):
    """The window side of the strong tier."""
    return strong_redecode_module.StrongRedecode(parts.engine)


def build_window_manager(parts):
    """The windows facade the controller side talks to."""
    mode = parts.settings.workload.feedback_boundary_mode
    return window_manager_module.WindowManager(
        parts.engine, feedback_boundary_mode=mode
    )


def decides_on_a_confidence(
    settings: machine_settings.MachineSettings,
) -> bool:
    """Whether the run joins the confidence of every solve of a window.

    False when the row decides on no confidence, and false for a
    Python-built policy, which brings its own decoder and reports its
    own soft output from one decode.
    """
    if settings.escalation.policy is not None:
        return False
    row = escalation_build.escalation_row(settings.escalation)
    return row.decides_on_a_confidence


def _reads_from_the_room_side(escalation_policy) -> bool:
    """Whether the plan's decoding tier reads the strong syndrome buffer."""
    strong = window_records.DecoderTier.STRONG
    return escalation_policy.primary_tier is strong


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
