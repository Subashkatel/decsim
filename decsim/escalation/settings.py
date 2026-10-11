"""The switching settings: when a window is decoded again, and on what.

SwitchingSettings fills the machine's switching slot on a run that
decodes weak first and escalates a window to the strong decoder (Toshio
et al. 2510.25222).
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Protocol

import decsim.config as config
import decsim.engine as engine_module
import decsim.escalation.deadlines as deadlines
import decsim.escalation.policies as escalation_policies
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.escalation.verdicts as verdicts
import decsim.ports as ports
import decsim.records.routes as route_records
import decsim.windows.revision_rules as revision_rules

# the verdict and rules of an edge whose settings name none: today's
GAP_THRESHOLD = verdicts.GapThreshold.Settings()
LATEST_REVISION = revision_rules.LatestRevision.Settings()
CANCEL_DOWNSTREAM = revision_rules.CancelDownstream.Settings()


class ThresholdSettings(Protocol):
    """A threshold row's settings record: decibels in, nats out.

    The facts at_task and for_task read are MachineSettings.task_facts,
    keyed by threshold_sources.TASK_FACTS.
    """

    threshold_decibels: Optional[float]

    @property
    def threshold_nats(self) -> Optional[float]:
        """The threshold as the weight a gap is compared in; None unread."""

    def at_task(self, facts: Mapping) -> "ThresholdSettings":
        """The record as the task's facts give it: a table's row read."""

    def for_task(self, facts: Mapping) -> Optional[ports.ThresholdSource]:
        """The source a task's shots share; None for most rows."""

    def build(self) -> ports.ThresholdSource:
        """A fresh source of this record."""


class VerdictSettings(Protocol):
    """A verdict row's settings record (verdicts.py).

    reads_threshold says whether the verdict decides on the edge's
    threshold.
    """

    name: str
    reads_threshold: bool

    def build(self) -> ports.EscalationVerdict:
        """A fresh verdict."""


class ReplaceRuleSettings(Protocol):
    """A replace rule's settings record (windows/revision_rules.py).

    replaces_provisionally says whether a later revision replaces the
    prediction before the window is final.
    """

    name: str
    replaces_provisionally: bool

    def build(self) -> ports.ReplaceRule:
        """A fresh rule."""


class CancelRuleSettings(Protocol):
    """A cancel rule's settings record (windows/revision_rules.py).

    cancels_on_keep says whether a keep cancels the target's decode.
    """

    name: str
    cancels_on_keep: bool

    def build(self) -> ports.CancelRule:
        """A fresh rule."""


class DeadlineSettings(Protocol):
    """An edge's deadline record (deadlines.py).

    slack_microseconds is the budget from the source job's arrival.
    """

    name: str
    slack_microseconds: float
    trigger: deadlines.TriggerSettings
    late_source_result: str

    def build(self, engine: engine_module.Engine) -> deadlines.Deadline:
        """A fresh deadline on the run's engine."""


class StrongWindowSettings(Protocol):
    """A strong window row's settings record (strong_window_shapes.py).

    restart_reread_buffer_regions is the double window's re-read width.
    """

    name: str
    absorbs_weak_windows: bool
    boundary_policy: ports.BoundaryPolicySettings
    restart_reread_buffer_regions: int

    def build(
        self, engine: engine_module.Engine
    ) -> strong_window_shapes.StrongWindowShape:
        """A fresh shape on the run's engine."""


@dataclasses.dataclass(frozen=True)
class SwitchingSettings:
    """The machine's switching slot: weak first, escalate on low confidence.

    Toshio et al. 2510.25222 Sec. III A. confidence is the signal the
    weak tier reports (decsim/confidence/), and the build refuses a weak
    decoder that cannot produce its evidence. threshold is a threshold
    row (threshold_sources.py); a weak result whose gap is at or above
    it is kept (the paper uses 20 dB). run_both_at_once is Step 1, the
    strong decoder started with the weak one and cancelled on
    confidence; False is the same section's on-demand variant (lines
    631-640). strong_window is the shape the strong tier re-decodes
    (strong_window_shapes.py): redo_window, the default, or
    double_window, Sec. III C. clock, threshold_cycles and switch_cycles
    price the verdict's threshold and switch logic; clock None is the
    machine's clock. The complementary gap's two solves are two jobs of
    the weak pool, so its unit_count decides whether they overlap.

    verdict is the row that keeps or escalates each weak result
    (verdicts.py), the gap against the threshold by default.
    replace_rule says which revision the frame takes, and cancel_rule
    what a keep does to the strong decode already asked for
    (windows/revision_rules.py).

    deadline, None by default, gives each source job a deadline and a
    trigger that starts the target decode before the verdict, and makes
    a late source result unavailable (deadlines.py, after DART-Q,
    Bhave et al. 2605.09142). It starts the target early, so it
    contradicts run_both_at_once, which starts it with every source
    job, and an absorbing strong window, which starts it late.

    strong_answer_route names one of records/routes.py
    ANSWER_ROUTE_NAMES, the way a strong answer reaches the frame.
    direct, the default: the strong host joins the weak crossing
    commit, which rides up with the selection, and sends the answer on
    strong_decoder_to_frame (Toshio 2510.25222 Fig. 1).
    through_weak_chip: the answer goes down on
    strong_decoder_to_weak_decoder, the weak chip joins it in one cycle
    of clock, and sends it on weak_decoder_to_frame (Yang 2605.04892
    Table I, the Pauli frame update). A run with no switching sends a
    strong answer direct.
    """

    confidence: ports.ConfidenceSettings
    threshold: ThresholdSettings
    clock: Optional[config.Clock] = None
    threshold_cycles: int = 0
    switch_cycles: int = 0
    run_both_at_once: bool = False
    strong_window: StrongWindowSettings = (
        strong_window_shapes.RedoWindow.Settings()
    )
    strong_answer_route: str = "direct"
    verdict: VerdictSettings = config.added_field(GAP_THRESHOLD)
    replace_rule: ReplaceRuleSettings = config.added_field(LATEST_REVISION)
    cancel_rule: CancelRuleSettings = config.added_field(CANCEL_DOWNSTREAM)
    deadline: Optional[DeadlineSettings] = config.added_field(None)

    def __post_init__(self) -> None:
        config.check_cycles("threshold_cycles", self.threshold_cycles)
        config.check_cycles("switch_cycles", self.switch_cycles)
        routes = route_records.ANSWER_ROUTE_NAMES
        if self.strong_answer_route not in routes:
            raise ValueError(
                f"switching.strong_answer_route must be one of "
                f"{tuple(routes)}, got {self.strong_answer_route!r}"
            )
        if self.deadline is not None:
            self._refuse_deadline_contradictions()

    def build_policy(
        self, threshold: ports.ThresholdSource
    ) -> ports.EscalationPolicy:
        """The policy that gives each weak result its verdict."""
        verdict = self.verdict
        built_verdict = verdict.build()
        return escalation_policies.Switching(
            threshold,
            self.run_both_at_once,
            built_verdict,
            verdict.reads_threshold,
        )

    def _refuse_deadline_contradictions(self) -> None:
        """A deadline starts the target early; these start it otherwise."""
        if self.run_both_at_once:
            raise ValueError(
                "switching.deadline starts the target decode when its "
                "trigger fires; run_both_at_once starts it with every "
                "source job (the two policies contradict; pick one)"
            )
        shape = self.strong_window
        if shape.absorbs_weak_windows:
            raise ValueError(
                f"switching.deadline starts the target decode before the "
                f"verdict; strong_window {shape.name} defers it until "
                f"the far weak boundary exists (the two policies "
                f"contradict; pick one)"
            )
