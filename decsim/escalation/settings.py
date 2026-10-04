"""The switching settings: when a window is decoded again, and on what.

SwitchingSettings is the machine's switching slot, filled on a run that
decodes weak first and escalates a window to the strong decoder. Its
strong window is the geometry the strong tier re-decodes
(strong_window_shapes.py, Toshio et al. arXiv 2510.25222).
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional, Protocol

import decsim.config as config
import decsim.engine as engine_module
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.ports as ports


class ConfidenceSettings(Protocol):
    """A confidence row's settings record, which builds the signal.

    ComplementaryGap.Settings, ClusterGap.Settings and
    ExtraClusterGap.Settings (decsim/confidence/) are the three.
    """

    name: str

    def build(
        self,
        weak_algorithm: ports.DecoderSettings,
        threshold_nats: Optional[float],
    ) -> ports.ConfidenceSignal:
        """The signal, from the weak decoder's record and the threshold."""


class ThresholdSettings(Protocol):
    """A threshold row's settings record (threshold_sources.py).

    It takes its threshold in decibels and hands it out in nats.
    """

    threshold_decibels: Optional[float]

    @property
    def threshold_nats(self) -> Optional[float]:
        """The threshold as the weight a gap is compared in; None unread."""

    def at_point(self, facts: Mapping) -> "ThresholdSettings":
        """The record as the point's facts give it: a table's row read."""

    def for_point(self, facts: Mapping) -> Optional[ports.ThresholdSource]:
        """The source a sweep point's shots share; None for most rows."""

    def build(self) -> ports.ThresholdSource:
        """A fresh source of this record."""


class StrongWindowSettings(Protocol):
    """A strong window row's settings record (strong_window_shapes.py).

    absorbs_weak_windows and boundary_policy are its row's declarations;
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

    Weak first, escalate serially on a small complementary gap (Toshio
    2510.25222 Sec. III A without the parallel head start).
    threshold is the Settings record of a threshold row
    (threshold_sources.py), which keeps the weak result when its gap is
    at or above the threshold (the paper uses 20 dB): fixed
    holds it as given, table holds the one an offline calibration csv
    gives the sweep point (threshold_column, calibrated for this run's
    window geometry), online starts there and adapts it across a point's
    shots, serial switching only; run_both_at_once is Sec. III A's Step
    1, the strong decoder started with the weak one and cancelled on
    confidence (false, the default, is the same section's on-demand
    variant, lines 631-640); strong_window is the Settings record of the
    shape of the window the strong tier re-decodes
    (decsim/escalation/strong_window_shapes.py): redo_window, the
    default, re-decodes the escalated window's commit region with its
    past face pinned on the earlier neighbour's committed correction and
    one buffer ahead (Bombin et al. 2303.04846 lines 775-788 and
    1456-1458); double_window is the paper's Sec. III C scheme as
    it is stated, an r_com + 2 r_buf extent read with no context, both
    faces pinned (lines 1248-1259), whose record holds how many of the
    strong region's buffer regions the restarted weak window re-reads.
    confidence is the Settings record of the signal the weak tier
    reports and the threshold decides on (decsim/confidence/), and the
    build refuses a weak decoder whose decode cannot produce that
    signal's evidence; walk_microseconds, the card that prices the
    signal's own computation on the weak unit, is a field of that
    record, None leaving each row on its own cost model.
    The complementary gap's two forced-class solves are two ordinary
    jobs of the weak pool, so weak_decoder.units alone decides whether
    they overlap. clock, threshold_cycles and switch_cycles price the
    verdict's threshold and switch logic; clock None is the machine's
    clock. An online threshold's calibrator is no setting: it is the
    point's state, built by the point's task (collect.Task) and handed
    to Machine.build.
    """

    confidence: ConfidenceSettings
    threshold: ThresholdSettings
    clock: Optional[config.Clock] = None
    threshold_cycles: int = 0
    switch_cycles: int = 0
    run_both_at_once: bool = False
    strong_window: StrongWindowSettings = (
        strong_window_shapes.RedoWindow.Settings()
    )

    def __post_init__(self) -> None:
        config.check_cycles("threshold_cycles", self.threshold_cycles)
        config.check_cycles("switch_cycles", self.switch_cycles)
