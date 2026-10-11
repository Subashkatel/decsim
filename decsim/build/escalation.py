"""The switching part: the confidence, the policy and the strong window side.

A switching run decodes weak first and escalates a window on low
confidence. This part binds the re-decode, the signal and the policy
onto the window side's and the decoder managers' optional ports; a run
with no switching leaves them unbound, as a gem5 cache with no
prefetcher holds a NULL one (src/mem/cache/Cache.py:108).
"""

import dataclasses
from typing import TYPE_CHECKING, Optional

import decsim.build.routes as routes_build
import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.ports as ports
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.window_commits as window_commits

if TYPE_CHECKING:
    import decsim.build.decoders as decoders_part
    import decsim.build.plan as plan_build
    import decsim.build.readout as readout_part
    import decsim.build.windows as windows_part


@dataclasses.dataclass(frozen=True)
class Switching:
    """One run's switching side, from the escalation decision onward.

    regions says which rounds an escalated window covers, shape lays the
    strong window over them, pending_strong_windows holds one until its
    conditions are met, and strong_redecode submits it and takes its
    result back.
    """

    confidence_signal: ports.ConfidenceSignal
    policy: ports.EscalationPolicy
    regions: strong_regions.StrongRegions
    # the strong window settings' shape
    shape: strong_window_shapes.StrongWindowPorts
    pending_strong_windows: pending_strong_windows.PendingStrongWindows
    strong_redecode: strong_redecode_module.StrongRedecode

    @classmethod
    def build(
        cls,
        settings: escalation_settings.SwitchingSettings,
        weak_decoder: decoder_settings.DecoderPoolSettings,
        engine: engine_module.Engine,
        online_threshold: Optional[ports.ThresholdSource] = None,
        machine_clock: Optional[config.Clock] = None,
    ) -> "Switching":
        """The signal from the weak decoder, the policy, the strong side.

        The confidence is built from the weak decoder's own settings and
        the task's threshold. The policy decides on online_threshold,
        the task's calibrator, when the threshold learns across shots.
        The verdict is priced on the switching clock, the machine's when
        the settings name none.
        """
        threshold_nats = settings.threshold.threshold_nats
        signal = settings.confidence.build(
            weak_decoder.algorithm, threshold_nats
        )
        threshold = _threshold_source(settings, online_threshold)
        policy = settings.build_policy(threshold)
        regions = strong_regions.StrongRegions()
        shape = settings.strong_window.build(engine)
        pending = pending_strong_windows.PendingStrongWindows()
        costs = _verdict_costs(settings, machine_clock)
        hops = routes_build.legacy_edge(settings.strong_answer_route)
        strong_redecode = strong_redecode_module.StrongRedecode(
            engine, hops, costs
        )
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
        """Wire the strong side, then bind it on the ports left empty."""
        self._wire_the_strong_side(plan, windows)
        strong_redecode = self.strong_redecode
        strong_redecode.carried_from = readout.weak_syndrome_buffer
        strong_redecode.strong_receiver = readout.strong_syndrome_round_receiver
        strong_redecode.strong_output = readout.strong_output
        strong_redecode.decode_queue = decoders.strong_decoder_manager
        strong_redecode.policy = self.policy
        windows.gap_join.signal = self.confidence_signal
        decoders.decoder_manager.escalation_policy = self.policy
        decoders.strong_decoder_manager.escalation_policy = self.policy

    def edges(self) -> escalate_edges.EscalateEdges:
        """The run's one escalate edge, from the weak pool to the strong."""
        return escalate_edges.EscalateEdges((self.strong_redecode,))

    def statistics_parts(self) -> tuple:
        """The strong redecode, at the switching settings' path."""
        return (("switching", self.strong_redecode),)

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
    machine_clock: Optional[config.Clock] = None,
) -> Optional[Switching]:
    """The switching part of a run whose slot is filled; None otherwise."""
    if settings is None:
        return None
    return Switching.build(
        settings, weak_decoder, engine, online_threshold, machine_clock
    )


def escalate_edges_of(
    switching: Optional[Switching],
) -> escalate_edges.EscalateEdges:
    """The run's escalate edges; none on a run with no switching."""
    if switching is None:
        return escalate_edges.EscalateEdges()
    return switching.edges()


def _verdict_costs(
    settings: escalation_settings.SwitchingSettings,
    machine_clock: Optional[config.Clock],
) -> window_commits.VerdictCosts:
    """The threshold and switch cycles, on the switching clock."""
    clocked = config.with_machine_clock(settings, machine_clock)
    return window_commits.VerdictCosts(
        clock=clocked.clock,
        threshold_cycles=clocked.threshold_cycles,
        switch_cycles=clocked.switch_cycles,
    )


def _threshold_source(
    settings: escalation_settings.SwitchingSettings,
    online_threshold: Optional[ports.ThresholdSource],
):
    """online_threshold, built once per task, else one from the settings."""
    if online_threshold is not None:
        return online_threshold
    return settings.threshold.build()
