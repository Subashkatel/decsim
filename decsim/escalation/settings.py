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
import decsim.escalation.policies as escalation_policies
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.ports as ports


class ConfidenceSettings(Protocol):
    """A confidence row's settings record, which builds the signal."""

    name: str

    def build(
        self,
        weak_algorithm: ports.DecoderSettings,
        threshold_nats: Optional[float],
    ) -> ports.ConfidenceSignal:
        """The signal, from the weak decoder's record and the threshold."""


class ThresholdSettings(Protocol):
    """A threshold row's settings record: decibels in, nats out."""

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

    def build_policy(
        self, threshold: ports.ThresholdSource
    ) -> ports.EscalationPolicy:
        """The policy that gives each weak result its verdict."""
        return escalation_policies.Switching(threshold, self.run_both_at_once)
