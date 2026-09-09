"""Build the escalation policy the yaml names, and what it decides on.

The policy instance is the authority over its own tier; the table row is
only how the yaml names it (design audit note 15 section 3, sinter's
_mux_sampler.py:33-40, which resolves the caller's object before its own
table).
"""

import decsim.confidence.signals as confidence_signals
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.tables as tables


def primary_tier(settings: escalation_settings.EscalationSettings) -> str:
    """The tier that decodes the plan's windows, for a caller with no policy.

    A policy the caller built is the one fact and answers for itself; the
    kind is only how the yaml names a policy, so the table is the lookup
    of last resort. sinter resolves the caller's own decoders before its
    built-in table (sinter/_collection/_mux_sampler.py:33-40) and gem5
    reads a built object's own params rather than its class table
    (src/python/m5/SimObject.py:204-205). The front asks this rather than
    building the policy, because a switching policy's threshold is
    resolved per sweep point and a config may reach here without one.
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


def build_escalation_policy(settings: escalation_settings.EscalationSettings):
    """The policy of the escalation kind, or the Python-built one."""
    if settings.policy is not None:
        return settings.policy
    row = escalation_row(settings)
    collaborators = _collaborators(row, settings)
    return row(collaborators)


def confidence_signal(escalation: escalation_settings.EscalationSettings):
    """The signal row a switching run's weak decoder reports and decides on.

    Every row takes the same one argument, the card that prices its own
    computation on the weak unit; None leaves the row on its own cost
    model.
    """
    row = tables.row(
        confidence_signals.CONFIDENCE_SIGNALS,
        "escalation.confidence",
        escalation.confidence,
    )
    return row(walk_microseconds=escalation.confidence_walk_microseconds)


def _collaborators(
    row, settings: escalation_settings.EscalationSettings
) -> escalation_policies.EscalationCollaborators:
    """The one record every escalation row is built from.

    A row that decides on no confidence reads none of the three fields,
    and the escalation section carries none of the keys they come from,
    so the record is empty for it.
    """
    if not row.decides_on_a_confidence:
        return escalation_policies.NO_CONFIDENCE
    threshold = _threshold_source(settings)
    signal = confidence_signal(settings)
    return escalation_policies.EscalationCollaborators(
        threshold=threshold,
        expected_source=signal.source,
        run_both_at_once=settings.run_both_at_once,
    )


def _threshold_source(settings: escalation_settings.EscalationSettings):
    """The row escalation.threshold_source names, for this sweep point.

    A row the front builds once per point (it learns across the point's
    shots) arrives already built; every other row is built here from the
    point's threshold in nats, its one constructor argument. The table
    source is resolved to a number per sweep point by the front
    (ExperimentConfig.point_settings), so a table run reaches the root
    with its threshold in nats or not at all.
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
            "escalation.threshold_source table resolves the threshold per "
            "sweep point in the front (ExperimentConfig.point_settings); "
            "build the machine through it, or give gap_threshold_db"
        )
    return row(settings.gap_threshold_nats)


def _sweep_point_source(settings: escalation_settings.EscalationSettings):
    """The source the front built for this point, which every shot shares."""
    if settings.online_threshold is None:
        raise ValueError(
            "escalation.threshold_source online is built once per sweep "
            "point by the front (ExperimentConfig.point_settings), which "
            "seeds it and shares it across the point's shots; build the "
            "machine through it"
        )
    return settings.online_threshold
