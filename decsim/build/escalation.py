"""The switching parts: one per escalate edge, its verdict and its redo side.

A switching run decodes on the entry pool first and escalates a window
along an edge on low confidence. Each edge's part binds its re-decode,
its source's signal and its policy onto the window side's and the
decoder managers' optional ports; a run with no edge leaves them
unbound, as a gem5 cache with no prefetcher holds a NULL one
(src/mem/cache/Cache.py:108).
"""

import dataclasses
from typing import TYPE_CHECKING, Optional

import decsim.build.layout as layout_module
import decsim.confidence.gap_join as gap_join_module
import decsim.confidence.no_signal as no_signal
import decsim.config as config
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.settings as escalation_settings
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.ports as ports
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.revision_rules as revision_rules
import decsim.windows.window_commits as window_commits

if TYPE_CHECKING:
    import decsim.build.decoders as decoders_part
    import decsim.build.plan as plan_build
    import decsim.build.readout as readout_part
    import decsim.build.windows as windows_part


@dataclasses.dataclass(frozen=True)
class Switching:
    """One escalate edge's switching side, from the escalation decision on.

    edge is the edge's plan and target_store the store its target pool
    reads. regions says which rounds an escalated window covers, shape
    lays the strong window over them, pending_strong_windows holds one
    until its conditions are met, and strong_redecode submits it and
    takes its result back. source_join reads the source pool's results
    for the edge's signal on an edge past the entry; the entry edge's
    is the window side's (build/windows.py).
    """

    edge: layout_module.EdgePlan
    target_store: str
    confidence_signal: ports.ConfidenceSignal
    policy: ports.EscalationPolicy
    regions: strong_regions.StrongRegions
    # the strong window settings' shape
    shape: strong_window_shapes.StrongWindowPorts
    pending_strong_windows: pending_strong_windows.PendingStrongWindows
    strong_redecode: strong_redecode_module.StrongRedecode
    source_join: Optional[gap_join_module.WindowGapJoin] = None

    @classmethod
    def build(
        cls,
        edge: layout_module.EdgePlan,
        layout: layout_module.Layout,
        engine: engine_module.Engine,
        online_threshold: Optional[ports.ThresholdSource] = None,
        machine_clock: Optional[config.Clock] = None,
    ) -> "Switching":
        """The signal from the source pool, the policy, the redo side.

        The confidence is built from the source pool's own settings and
        the edge's threshold. The policy decides on online_threshold,
        the task's calibrator, when the threshold learns across shots.
        The verdict is priced on the switching clock, the machine's when
        the settings name none.
        """
        settings = edge.settings
        hops = edge.hops
        source = layout.pool(hops.source)
        target = layout.pool(hops.target)
        source_settings = source.settings
        threshold_nats = settings.threshold.threshold_nats
        signal = settings.confidence.build(
            source_settings.algorithm, threshold_nats
        )
        _refuse_a_threshold_with_no_gap(settings, signal)
        threshold = _threshold_source(settings, online_threshold)
        policy = settings.build_policy(threshold)
        regions = strong_regions.StrongRegions()
        shape = settings.strong_window.build(engine)
        pending = pending_strong_windows.PendingStrongWindows()
        costs = _verdict_costs(settings, machine_clock)
        rules = _edge_rules(settings, costs)
        strong_redecode = strong_redecode_module.StrongRedecode(
            engine, hops, rules
        )
        return cls(
            edge=edge,
            target_store=target.store,
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
        """Wire the redo side, then bind it on the ports left empty."""
        self._wire_the_strong_side(plan, windows)
        strong_redecode = self.strong_redecode
        carried = readout.store_parts(self.edge.carried_from)
        target_store = readout.store_parts(self.target_store)
        target_manager = decoders.manager_of(strong_redecode.target)
        strong_redecode.carried_from = carried.store
        strong_redecode.strong_receiver = target_store.receiver
        strong_redecode.strong_output = target_store.output
        strong_redecode.decode_queue = target_manager
        strong_redecode.policy = self.policy
        windows.retention.add_edge(
            strong_redecode.target, carried.store, target_store.store
        )
        target_manager.escalation_policy = self.policy
        self._bind_the_source_join(windows, decoders)

    def statistics_parts(self) -> tuple:
        """The strong redecode, at the edge's path."""
        return ((self.edge.name, self.strong_redecode),)

    def seed_roots(self) -> tuple:
        """The policy, at the edge's seed root."""
        return ((self.edge.policy_seed, self.policy),)

    def check_settled(self) -> None:
        """No strong escalation is pending, and no source solve unjoined."""
        strong_redecode = self.strong_redecode
        if strong_redecode.has_pending():
            pending = strong_redecode.pending_work()
            raise RuntimeError(
                f"the run ended with pending strong escalations: {pending}"
            )
        if self.source_join is None:
            return
        unresolved = self.source_join.unresolved_windows()
        if unresolved:
            raise RuntimeError(
                f"the run ended with windows holding an unjoined solve on "
                f"edge {self.edge.name!r}: {unresolved}"
            )

    def _bind_the_source_join(
        self,
        windows: "windows_part.Windows",
        decoders: "decoders_part.Decoders",
    ) -> None:
        """The source join reads the edge's signal off the source's results."""
        join = self.source_join
        if join is None:
            return
        source = self.strong_redecode.source
        join.signal = self.confidence_signal
        join.verdict = windows.verdict
        join.decode_queue = decoders.manager_of(source)

    def _wire_the_strong_side(
        self, plan: "plan_build.Plan", windows: "windows_part.Windows"
    ) -> None:
        regions = self.regions
        regions.interaction = plan.window_interaction
        regions.planner = windows.planner
        regions.tracker = windows.tracker
        regions.retention = windows.retention
        shape = self.shape
        strong_redecode = self.strong_redecode
        shape.edge = strong_redecode
        shape.regions = regions
        shape.planner = windows.planner
        shape.retention = windows.retention
        shape.builder = windows.builder
        shape.requester = windows.requester
        shape.ledger = windows.ledger
        shape.courier = windows.courier
        strong_redecode.shape = shape
        strong_redecode.pending = self.pending_strong_windows
        strong_redecode.retention = windows.retention
        strong_redecode.decoder_output = windows.decoder_output
        strong_redecode.verdict = windows.verdict


def build_switching(
    layout: layout_module.Layout,
    engine: engine_module.Engine,
    online_threshold: Optional[ports.ThresholdSource] = None,
    machine_clock: Optional[config.Clock] = None,
) -> tuple:
    """One switching part per escalate edge, in graph order.

    The task's calibrator, online_threshold, decides on the entry edge,
    the one whose threshold the task learns. An edge past the entry
    takes neither a region that absorbs its source's windows nor a
    confidence of two solves (stage 5).
    """
    parts = []
    threshold = online_threshold
    for edge in layout.edges:
        part = Switching.build(edge, layout, engine, threshold, machine_clock)
        if parts:
            _refuse_past_the_entry(part)
            join = gap_join_module.WindowGapJoin(engine)
            part = dataclasses.replace(part, source_join=join)
        parts.append(part)
        threshold = None
    return tuple(parts)


def join_the_chain(switching_parts: tuple) -> None:
    """Each edge past the entry joins the results of the pool it leaves.

    The edge into that pool hands its target's results to the join, so
    the signal of the edge out reads them before its verdict, as the
    entry edge's signal reads the entry pool's solves. It also starts
    the edge out beside each target job it queues, and passes a keep's
    cancel on to it.
    """
    redecode_by_target = {}
    for part in switching_parts:
        redecode = part.strong_redecode
        redecode_by_target[redecode.target] = redecode
    for part in switching_parts:
        if part.source_join is None:
            continue
        source = part.strong_redecode.source
        entering = redecode_by_target[source]
        entering.target_join = part.source_join
        entering.next_edge = part.strong_redecode


def signal_by_pool(switching_parts: tuple) -> dict:
    """Each source pool's confidence signal; none with no edge."""
    signals = {}
    for part in switching_parts:
        source = part.strong_redecode.source
        signals[source] = part.confidence_signal
    return signals


def escalate_edges_of(switching_parts: tuple) -> escalate_edges.EscalateEdges:
    """The run's escalate edges, in graph order."""
    edges = []
    for part in switching_parts:
        edges.append(part.strong_redecode)
    return escalate_edges.EscalateEdges(tuple(edges))


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


def _edge_rules(
    settings: escalation_settings.SwitchingSettings,
    costs: window_commits.VerdictCosts,
) -> revision_rules.EdgeRules:
    """The edge's rules built from its settings, its verdict at costs."""
    replace_rule = settings.replace_rule
    cancel_rule = settings.cancel_rule
    replace = replace_rule.build()
    cancel = cancel_rule.build()
    return revision_rules.EdgeRules(replace, cancel, costs)


def _threshold_source(
    settings: escalation_settings.SwitchingSettings,
    online_threshold: Optional[ports.ThresholdSource],
):
    """online_threshold, built once per task, else one from the settings."""
    if online_threshold is not None:
        return online_threshold
    return settings.threshold.build()


def _refuse_a_threshold_with_no_gap(
    settings: escalation_settings.SwitchingSettings,
    signal: ports.ConfidenceSignal,
) -> None:
    """A verdict that reads the threshold needs a gap to read.

    Under confidence none it would escalate every window, a run that
    looks like a threshold study and is not one.
    """
    verdict = settings.verdict
    if not verdict.reads_threshold:
        return
    if signal.source != no_signal.NO_SIGNAL_SOURCE:
        return
    raise ValueError(
        f"switching.verdict {verdict.name} reads a confidence gap, and "
        "confidence none gives none; name a status, always or never "
        "verdict, or a confidence row"
    )


def _refuse_past_the_entry(part: Switching) -> None:
    """Refuse what only the entry edge's windows can do yet."""
    edge = part.edge
    settings = edge.settings
    shape = settings.strong_window
    signal = part.confidence_signal
    if shape.absorbs_weak_windows:
        _refuse(edge, f"strong_window {shape.name}")
    if signal.forced_logical_classes:
        _refuse(edge, "a confidence of two solves")


def _refuse(edge: layout_module.EdgePlan, what: str) -> None:
    raise NotImplementedError(
        f"escalate edge {edge.name!r}: {what} on an edge that does not "
        f"leave the entry pool is not implemented yet (stage 5)"
    )
