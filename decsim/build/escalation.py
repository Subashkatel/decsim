"""The switching part: the confidence, the policy and the strong window side.

A run whose switching slot is filled decodes weak first and escalates a
window on low confidence. This part builds what only that run has (the
signal the weak decoder reports, the policy that decides on it with its
threshold, the strong regions, the strong window's shape, the pending
strong windows and the strong re-decode), wires the strong side to the
window side, and binds the re-decode, the signal and the policy onto
the window side's and the decoder managers' optional ports. A run with
no switching builds none of it and leaves those ports unbound, as a
gem5 cache with no prefetcher holds a NULL one
(src/mem/cache/Cache.py:108, Param.BasePrefetcher(NULL)).
"""

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.policies as escalation_policies
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.ports as ports

if TYPE_CHECKING:
    import decsim.build.decoders as decoders_part
    import decsim.build.plan as plan_build
    import decsim.build.readout as readout_part
    import decsim.build.windows as windows_part


@dataclasses.dataclass(frozen=True)
class Switching:
    """What a switching run adds: the decision and the strong window side.

    confidence_signal is what the weak decoder reports and policy decides
    on; regions says which rounds an escalated window covers, shape lays
    the strong window over them, pending_strong_windows holds one until
    the conditions its row named are met, and strong_redecode submits it
    and takes its result back. connect wires the four strong components
    to the window side's.
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
        online_threshold: Optional[ports.ThresholdSource] = None,
    ) -> "Switching":
        """The signal from the weak decoder, the policy, the strong side.

        The confidence record builds its row from the weak decoder's own
        settings and the point's threshold: the decode's weight step,
        the threshold it grows to, the unit's cycle count; a row reads
        what it needs and ignores the rest. The policy expects the
        signal's source, and decides on online_threshold, the point's
        calibrator, when the threshold learns across shots. The shape is
        the strong window record's row: the redo window, or the double
        window of Toshio Sec. III C.
        """
        threshold_nats = settings.threshold.threshold_nats
        signal = settings.confidence.build(
            weak_decoder.algorithm, threshold_nats
        )
        threshold = _threshold_source(settings, online_threshold)
        policy = escalation_policies.Switching(
            threshold=threshold,
            expected_source=signal.source,
            run_both_at_once=settings.run_both_at_once,
        )
        regions = strong_regions.StrongRegions()
        shape = settings.strong_window.build(engine)
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
        plan: "plan_build.Plan",
        readout: "readout_part.Readout",
        windows: "windows_part.Windows",
        decoders: "decoders_part.Decoders",
    ) -> None:
        """Wire the strong side, then bind it on the ports left empty.

        The strong side reads the window side's plan, rounds and stores,
        takes an escalated region from the strong store and submits to
        the host's manager. The committer, the verdict and the window
        manager hand the re-decode its windows, and the join reads the
        signal. The verdict asks the policy to keep or escalate, the
        requester which tiers decode a ready window, and both managers
        teach it a strong result.
        """
        self._wire_the_strong_side(plan, windows)
        strong_redecode = self.strong_redecode
        strong_redecode.strong_receiver = readout.strong_syndrome_round_receiver
        strong_redecode.strong_output = readout.strong_output
        strong_redecode.decode_queue = decoders.strong_decoder_manager
        windows.committer.strong_redecode = strong_redecode
        windows.verdict.strong_redecode = strong_redecode
        windows.window_manager.strong_redecode = strong_redecode
        windows.gap_join.signal = self.confidence_signal
        windows.verdict.escalation_policy = self.policy
        windows.requester.escalation_policy = self.policy
        decoders.decoder_manager.escalation_policy = self.policy
        decoders.strong_decoder_manager.escalation_policy = self.policy

    def check_settled(self) -> None:
        """No strong escalation is pending."""
        strong_redecode = self.strong_redecode
        if not strong_redecode.has_pending():
            return
        pending = strong_redecode.pending_work()
        raise RuntimeError(
            f"the run ended with pending strong escalations: {pending}"
        )

    def _wire_the_strong_side(
        self, plan: "plan_build.Plan", windows: "windows_part.Windows"
    ) -> None:
        regions = self.regions
        regions.interaction = plan.window_interaction
        regions.planner = windows.planner
        regions.tracker = windows.tracker
        regions.retention = windows.retention
        shape = self.shape
        shape.regions = regions
        shape.planner = windows.planner
        shape.retention = windows.retention
        shape.builder = windows.builder
        shape.requester = windows.requester
        shape.ledger = windows.ledger
        shape.courier = windows.courier
        strong_redecode = self.strong_redecode
        strong_redecode.shape = shape
        strong_redecode.pending = self.pending_strong_windows
        strong_redecode.retention = windows.retention
        strong_redecode.decoder_output = windows.decoder_output
        strong_redecode.verdict = windows.verdict


def build_switching(
    settings: Optional[escalation_settings.SwitchingSettings],
    weak_decoder: Optional[decoder_settings.DecoderPoolSettings],
    engine: engine_module.Engine,
    online_threshold: Optional[ports.ThresholdSource] = None,
) -> Optional[Switching]:
    """The switching part of a run whose slot is filled; None otherwise."""
    if settings is None:
        return None
    return Switching.build(settings, weak_decoder, engine, online_threshold)


def _threshold_source(
    settings: escalation_settings.SwitchingSettings,
    online_threshold: Optional[ports.ThresholdSource],
):
    """The point's threshold source, built from its row's record.

    A row built once per point (it learns across the point's shots)
    arrives already built as online_threshold; every other row's record
    builds its source here.
    """
    if online_threshold is not None:
        return online_threshold
    return settings.threshold.build()
