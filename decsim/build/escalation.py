"""Build the escalation policy the yaml names, and what it decides on.

The policy instance is the authority over its own tier; the table row is
only how the yaml names it (sinter's _mux_sampler.py:33-40, which
resolves the caller's object before its own table).
"""

import copy
from typing import TYPE_CHECKING, Optional

import decsim.burst_detectors.settings as burst_detector_settings
import decsim.confidence.signals as confidence_signals
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.ports as ports
import decsim.settings as machine_settings
import decsim.tables as tables

if TYPE_CHECKING:
    import decsim.build.plan as plan_build


def primary_tier(settings: escalation_settings.EscalationSettings) -> str:
    """The tier that decodes the plan's windows, for a caller with no policy.

    A policy the caller built is the one fact and answers for itself; the
    kind is only how the yaml names a policy, so the table is the lookup
    of last resort. sinter resolves the caller's own decoders before its
    built-in table (sinter/_collection/_mux_sampler.py:33-40) and gem5
    reads a built object's own params rather than its class table
    (src/python/m5/SimObject.py:204-205). The experiments layer asks
    this rather than building the policy, because a switching policy's
    threshold is resolved per sweep point and a config may reach here
    without one.
    """
    row = escalation_row(settings)
    return row.primary_tier.value


def strong_window_row(settings: escalation_settings.EscalationSettings):
    """The strong window shape class the escalation section names."""
    return tables.row(
        escalation_settings.STRONG_WINDOW_SHAPES,
        "escalation.strong_window",
        settings.strong_window,
    )


def absorbs_weak_windows(
    settings: escalation_settings.EscalationSettings,
) -> bool:
    """Whether the strong window replaces the weak windows it covers."""
    row = strong_window_row(settings)
    return row.absorbs_weak_windows


def escalation_row(settings: escalation_settings.EscalationSettings):
    """The policy this run escalates with: the built one, or the kind's row.

    Both answer the facts the port declares (primary_tier,
    requires_strong_context, decides_on_a_confidence), so a build site
    that has no policy object yet reads them here.
    """
    if settings.policy is not None:
        return settings.policy
    return tables.row(
        escalation_settings.ESCALATIONS, "escalation.kind", settings.kind
    )


def builds_a_confidence_signal(
    settings: escalation_settings.EscalationSettings,
) -> bool:
    """Whether the run joins the confidence of every solve of a window.

    False when the row decides on no confidence, and false for a
    Python-built policy, which brings its own decoder and reports its
    own soft output from one decode.
    """
    if settings.policy is not None:
        return False
    row = escalation_row(settings)
    return row.decides_on_a_confidence


def build_escalation_policy(
    settings: escalation_settings.EscalationSettings,
    weak_decoder: decoder_settings.DecoderSettings,
):
    """The policy of the escalation kind, or a copy of the Python-built one.

    Each machine binds its own peers onto the policy's ports, and one
    settings record builds a machine per shot, so the built policy is a
    prototype: each machine gets a shallow copy, whose ports start
    unbound and whose collaborators (its threshold source) are the
    prototype's own, shared by every shot as a row's are.
    """
    if settings.policy is not None:
        return copy.copy(settings.policy)
    row = escalation_row(settings)
    collaborators = _collaborators(row, settings, weak_decoder)
    return row(collaborators)


def confidence_row(escalation: escalation_settings.EscalationSettings):
    """The signal row escalation.confidence names, as a class.

    What a row needs from a decode and what it says when refused are
    the class's, so a build site that only checks the pairing reads
    them here without building the row.
    """
    return tables.row(
        confidence_signals.CONFIDENCE_SIGNALS,
        "escalation.confidence",
        escalation.confidence,
    )


def confidence_signal(
    escalation: escalation_settings.EscalationSettings,
    weak_decoder: decoder_settings.DecoderSettings,
):
    """The signal row a switching run's weak decoder reports and decides on.

    Every row builds itself from the escalation section and the weak
    decoder's settings: the card that prices its own computation, the
    decode's weight step, the threshold it grows to, the unit's cycle
    count; a row reads what it needs and ignores the rest.
    """
    row = confidence_row(escalation)
    return row.from_settings(escalation, weak_decoder)


def build_burst_detector(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    plan: "plan_build.Plan",
    escalation_policy: ports.EscalationPolicy,
) -> Optional[ports.BurstDetector]:
    """The detector burst_detector.kind names; None for the row none.

    The detector feeds escalation, so a policy that never escalates is
    refused beside it; it scores detection events, so every operation
    it scores brings the circuit they are formed from, and it is
    calibrated from that circuit, its round count and the round period.
    """
    section = settings.burst_detector
    row = tables.row(
        burst_detector_settings.BURST_DETECTORS,
        "burst_detector.kind",
        section.kind,
    )
    if row is None:
        return None
    _refuse_a_policy_that_cannot_escalate(section.kind, escalation_policy)
    circuits = _counted_circuits(plan)
    round_period = settings.qpu.round_period_microseconds
    return row(section.row_settings, engine, circuits, round_period)


def _collaborators(
    row,
    settings: escalation_settings.EscalationSettings,
    weak_decoder: decoder_settings.DecoderSettings,
) -> escalation_policies.EscalationCollaborators:
    """The one record every escalation row is built from.

    A row that decides on no confidence reads none of the three fields,
    and the escalation section carries none of the keys they come from,
    so the record is empty for it.
    """
    if not row.decides_on_a_confidence:
        return escalation_policies.NO_CONFIDENCE
    threshold = _threshold_source(settings)
    signal = confidence_signal(settings, weak_decoder)
    return escalation_policies.EscalationCollaborators(
        threshold=threshold,
        expected_source=signal.source,
        run_both_at_once=settings.run_both_at_once,
    )


def _threshold_source(settings: escalation_settings.EscalationSettings):
    """The row escalation.threshold_source names, for this sweep point.

    A row the experiments layer builds once per point (it learns across
    the point's shots) arrives already built; every other row is built
    here from the point's threshold in nats, its one constructor
    argument. The table source is resolved to a number per sweep point
    by the experiments layer (ExperimentConfig.point_task), so a
    table run reaches the root with its threshold in nats or not at all.
    """
    row = tables.row(
        escalation_settings.THRESHOLD_SOURCES,
        "escalation.threshold_source",
        settings.threshold_source,
    )
    if row.built_per_sweep_point:
        return _sweep_point_source(settings)
    if settings.gap_threshold_nats is None:
        raise ValueError(
            "escalation.threshold_source table resolves the threshold "
            "per sweep point in the experiments layer "
            "(ExperimentConfig.point_task); build the machine "
            "through it, or give gap_threshold_db"
        )
    return row(settings.gap_threshold_nats)


def _sweep_point_source(settings: escalation_settings.EscalationSettings):
    """The source the experiments layer built, shared by every shot."""
    if settings.online_threshold is None:
        raise ValueError(
            "escalation.threshold_source online is built once per sweep "
            "point by the experiments layer "
            "(ExperimentConfig.point_task), which seeds it and "
            "shares it across the point's shots; build the machine "
            "through it"
        )
    return settings.online_threshold


def _refuse_a_policy_that_cannot_escalate(kind: str, escalation_policy):
    if escalation_policy.requires_strong_context:
        return
    raise ValueError(
        f"burst_detector.kind {kind} sends a burst's windows to the "
        "strong decoder, which only escalation.kind switching does; "
        "write burst_detector: {kind: none}, or escalate with switching"
    )


def _counted_circuits(plan) -> dict:
    """Each counted operation's circuit and round count, by operation id."""
    round_count_by_operation = {}
    for resolved in plan.run_plan.resolved_operations:
        operation_id = resolved.operation_id
        round_count_by_operation[operation_id] = resolved.round_count
    circuits = {}
    for operation in plan.planned_operations:
        _refuse_an_uncounted_operation(operation)
        round_count = round_count_by_operation[operation.id]
        circuits[operation.id] = (operation.circuit, round_count)
    return circuits


def _refuse_an_uncounted_operation(operation) -> None:
    """The detector counts one standalone operation's events per circuit."""
    if operation.circuit is None:
        raise ValueError(
            f"operation {operation.id} has no circuit, so no detection "
            "events are formed for the burst detector to count"
        )
    if operation.stream_id is not None:
        raise ValueError(
            f"operation {operation.id} is a segment of stream "
            f"{operation.stream_id!r}; the burst detector counts "
            "standalone operations, one circuit each"
        )
