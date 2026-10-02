"""The switching part: the confidence, the policy and the strong window side.

A run whose switching slot is filled decodes weak first and escalates a
window on low confidence. This part builds what only that run has (the
signal the weak decoder reports, the policy that decides on it with its
threshold, the strong regions, the strong window's shape, the pending
strong windows and the strong re-decode) and binds the policy onto the
window side's and the decoder managers' optional ports. A run with no
switching builds none of it and leaves those ports unbound, as a gem5
cache with no prefetcher holds a NULL one
(src/mem/cache/Cache.py:108, Param.BasePrefetcher(NULL)).
"""

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

import decsim.burst_detectors.settings as burst_detector_settings
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.ports as ports
import decsim.settings as machine_settings
import decsim.tables as tables

if TYPE_CHECKING:
    import decsim.build.decoders as decoders_part
    import decsim.build.plan as plan_build
    import decsim.build.windows as windows_part


@dataclasses.dataclass(frozen=True)
class Switching:
    """What a switching run adds: the decision and the strong window side.

    confidence_signal is what the weak decoder reports and policy decides
    on; regions says which rounds an escalated window covers, shape lays
    the strong window over them, pending_strong_windows holds one until
    the conditions its row named are met, and strong_redecode submits it
    and takes its result back. The windows part wires the four strong
    components among its own.
    """

    confidence_signal: ports.ConfidenceSignal
    policy: escalation_policies.Switching
    regions: strong_regions.StrongRegions
    # the row escalation.strong_window names
    shape: Any
    pending_strong_windows: pending_strong_windows.PendingStrongWindows
    strong_redecode: strong_redecode_module.StrongRedecode

    @classmethod
    def build(
        cls,
        settings: escalation_settings.SwitchingSettings,
        weak_decoder: decoder_settings.DecoderPoolSettings,
        engine: engine_module.Engine,
    ) -> "Switching":
        """The signal from the weak decoder, the policy, the strong side.

        The confidence record builds its row from the weak decoder's own
        settings and the point's threshold: the decode's weight step,
        the threshold it grows to, the unit's cycle count; a row reads
        what it needs and ignores the rest. The policy expects the
        signal's source. The shape is the row escalation.strong_window
        names: the redo window, or the double window of Toshio Sec. III C.
        """
        threshold_nats = settings.threshold.threshold_nats
        signal = settings.confidence.build(
            weak_decoder.algorithm, threshold_nats
        )
        threshold = _threshold_source(settings)
        policy = escalation_policies.Switching(
            threshold=threshold,
            expected_source=signal.source,
            run_both_at_once=settings.run_both_at_once,
        )
        regions = strong_regions.StrongRegions()
        shape_row = strong_window_row(settings)
        shape = shape_row(engine)
        pending = pending_strong_windows.PendingStrongWindows()
        strong_redecode = strong_redecode_module.StrongRedecode(engine)
        return cls(
            confidence_signal=signal,
            policy=policy,
            regions=regions,
            shape=shape,
            pending_strong_windows=pending,
            strong_redecode=strong_redecode,
        )

    def connect(
        self,
        windows: "windows_part.Windows",
        decoders: "decoders_part.Decoders",
    ) -> None:
        """Bind the policy on the ports a run with no switching leaves empty.

        The verdict asks it to keep or escalate, the requester which tiers
        decode a ready window, both managers teach it a strong result, and
        a burst detector, when the run has one, overrides its threshold.
        """
        windows.verdict.escalation_policy = self.policy
        windows.requester.escalation_policy = self.policy
        decoders.decoder_manager.escalation_policy = self.policy
        decoders.strong_decoder_manager.escalation_policy = self.policy
        if windows.burst_detector is not None:
            self.policy.burst_detector = windows.burst_detector


def build_switching(
    settings: Optional[escalation_settings.SwitchingSettings],
    weak_decoder: Optional[decoder_settings.DecoderPoolSettings],
    engine: engine_module.Engine,
) -> Optional[Switching]:
    """The switching part of a run whose slot is filled; None otherwise."""
    if settings is None:
        return None
    return Switching.build(settings, weak_decoder, engine)


def strong_window_row(settings: escalation_settings.SwitchingSettings):
    """The strong window shape class the switching slot names."""
    return tables.row(
        escalation_settings.STRONG_WINDOW_SHAPES,
        "escalation.strong_window",
        settings.strong_window,
    )


def absorbs_weak_windows(
    settings: Optional[escalation_settings.SwitchingSettings],
) -> bool:
    """Whether the strong window replaces the weak windows it covers."""
    if settings is None:
        return False
    row = strong_window_row(settings)
    return row.absorbs_weak_windows


def build_burst_detector(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    plan: "plan_build.Plan",
) -> Optional[ports.BurstDetector]:
    """The detector burst_detector.kind names; None for the row none.

    The detector feeds escalation, so a run with no switching is refused
    beside it; it scores detection events, so every operation it scores
    brings the circuit they are formed from, and it is calibrated from
    that circuit, its round count and the round period.
    """
    section = settings.burst_detector
    row = tables.row(
        burst_detector_settings.BURST_DETECTORS,
        "burst_detector.kind",
        section.kind,
    )
    if row is None:
        return None
    _refuse_a_run_that_cannot_escalate(section.kind, settings.switching)
    circuits = _counted_circuits(plan)
    round_period = settings.qpu.round_period_microseconds
    return row(section.row_settings, engine, circuits, round_period)


def _threshold_source(settings: escalation_settings.SwitchingSettings):
    """The point's threshold source, built from its row's record.

    A row the experiments layer builds once per point (it learns across
    the point's shots) arrives already built as online_threshold; every
    other row's record builds its source here. The table record holds
    its number once the experiments layer has looked the point up
    (ExperimentConfig.point_task), and refuses to build before.
    """
    if settings.online_threshold is not None:
        return settings.online_threshold
    return settings.threshold.build()


def _refuse_a_run_that_cannot_escalate(
    kind: str, switching: Optional[escalation_settings.SwitchingSettings]
) -> None:
    if switching is not None:
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
