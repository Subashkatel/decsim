"""The windows part: which windows exist, when each is decoded, where it goes.

The window manager is the part's face to the rest of the machine: the
stores tell it which rounds landed, it closes each window whose rounds
are in, the requester asks the decoder manager for a decode, the
verdict keeps the result or escalates it, and the committer writes the
correction and carries the boundary to the next window. A run that may
escalate adds the strong re-decode, and a run that decides on a
confidence adds the signal and the join of a window's solves.
"""

import dataclasses
from typing import Any, Optional

import decsim.build.escalation as escalation_build
import decsim.build.plan as plan_build
import decsim.confidence.gap_join as gap_join_module
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.pending_strong_windows as pending_strong_windows
import decsim.escalation.strong_redecode as strong_redecode_module
import decsim.escalation.strong_regions as strong_regions
import decsim.links.window_transfers as window_transfers_module
import decsim.ports as ports
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


@dataclasses.dataclass(frozen=True)
class Windows:
    """Every component that turns landed rounds into committed corrections.

    confidence_signal and gap_join are None on a run that decides on no
    confidence. The four strong fields are None on a run that never
    escalates: regions says which rounds an escalated window covers,
    shape lays the strong window over them, pending_strong_windows holds
    one until the conditions its row named are met, and strong_redecode
    submits it and takes its result back.
    """

    # the policy and the detector are built before the parts, since the
    # plan and the decoder units are compiled for them
    escalation_policy: ports.EscalationPolicy
    burst_detector: Optional[ports.BurstDetector]
    models: window_planner_module.WindowModels
    planner: window_planner_module.WindowPlanner
    tracker: round_tracker_module.RoundTracker
    retention: round_retention_module.RoundRetention
    window_transfers: window_transfers_module.WindowTransfers
    decoder_output: decoder_output_module.DecoderOutput
    gate: decode_requests.WindowInputGate
    builder: decode_requests.DecodeRequestBuilder
    ledger: committed_rounds.LogicalLedger
    results: operation_results.OperationResults
    courier: window_boundaries.BoundaryCourier
    committer: window_commits.WindowCommitter
    verdict: window_commits.WindowVerdict
    requester: decode_requests.DecodeRequester
    confidence_signal: Optional[ports.ConfidenceSignal]
    gap_join: Optional[gap_join_module.WindowGapJoin]
    regions: Optional[strong_regions.StrongRegions]
    # the row escalation.strong_window names
    shape: Optional[Any]
    pending_strong_windows: Optional[
        pending_strong_windows.PendingStrongWindows
    ]
    strong_redecode: Optional[strong_redecode_module.StrongRedecode]
    window_manager: window_manager_module.WindowManager

    @classmethod
    def build(
        cls,
        settings: machine_settings.MachineSettings,
        engine: engine_module.Engine,
        plan: plan_build.Plan,
        escalation_policy: ports.EscalationPolicy,
        burst_detector: Optional[ports.BurstDetector],
        links: ports.Link,
    ) -> "Windows":
        """Every component of the window side, wired to one another.

        One line per component, in the order a window meets them, then
        the wires inside the part.
        """
        run_plan = plan.run_plan
        built_models = settings.workload.built_models
        if built_models is None:
            built_models = built_window_models.BuiltWindowModels()
        models = window_planner_module.WindowModels(built_models)
        planner = window_planner_module.WindowPlanner(
            run_plan.resolved_operations,
            run_plan.execution,
            plan.planned_operations,
        )
        tracker = round_tracker_module.RoundTracker()
        retention = _retention(plan, escalation_policy)
        window_transfers = window_transfers_module.WindowTransfers(engine)
        decoder_output = decoder_output_module.DecoderOutput(engine)
        copies_the_fold = _copies_the_boundary_fold(settings, escalation_policy)
        gate = decode_requests.WindowInputGate(copies_the_fold)
        builder = decode_requests.DecodeRequestBuilder(engine)
        ledger = committed_rounds.LogicalLedger()
        results = operation_results.OperationResults()
        courier = window_boundaries.BoundaryCourier()
        committer = window_commits.WindowCommitter(engine)
        verdict = _verdict(settings, engine)
        requester = decode_requests.DecodeRequester(
            clock=settings.windows.clock,
            decision_cycles=settings.windows.decision_cycles,
        )
        confidence_signal, gap_join = _confidence(settings, engine)
        regions, shape, pending, strong_redecode = _strong_redecode(
            settings, engine, escalation_policy
        )
        feedback_boundary_mode = settings.workload.feedback_boundary_mode
        window_manager = window_manager_module.WindowManager(
            engine, feedback_boundary_mode=feedback_boundary_mode
        )
        windows = cls(
            escalation_policy=escalation_policy,
            burst_detector=burst_detector,
            models=models,
            planner=planner,
            tracker=tracker,
            retention=retention,
            window_transfers=window_transfers,
            decoder_output=decoder_output,
            gate=gate,
            builder=builder,
            ledger=ledger,
            results=results,
            courier=courier,
            committer=committer,
            verdict=verdict,
            requester=requester,
            confidence_signal=confidence_signal,
            gap_join=gap_join,
            regions=regions,
            shape=shape,
            pending_strong_windows=pending,
            strong_redecode=strong_redecode,
            window_manager=window_manager,
        )
        windows._wire_the_plan(plan)
        windows._wire_the_window_chain(links)
        windows._wire_the_decision()
        windows._wire_the_strong_redecode()
        return windows

    def connect(
        self,
        weak_store: ports.SyndromeBuffer,
        strong_store: Optional[ports.SyndromeBuffer],
        store_output: ports.SyndromeBufferOutput,
        strong_output: Optional[ports.SyndromeBufferOutput],
        strong_receiver: Optional[ports.StrongSyndromeRoundReceiver],
        primary_decoder: Optional[ports.Decoder],
        strong_decoder: Optional[ports.Decoder],
        decode_queue: ports.DecodeQueue,
        strong_decode_queue: Optional[ports.DecodeQueue],
        input_fold: ports.DecoderInputFold,
        frame: Optional[ports.Frame],
        conditional_release: ports.ReleaseReceiver,
        factory: ports.MagicStateFactory,
        detection_events: ports.DetectionEventPlacement,
    ) -> None:
        """Read from the stores, decode on the managers, commit to the frame."""
        self.models.decoder = primary_decoder
        self.models.strong_decoder = strong_decoder
        self.retention.weak_store = weak_store
        self.retention.strong_store = strong_store
        self.retention.detection_events = detection_events
        self.gate.input_fold = input_fold
        self.requester.store_output = store_output
        self.requester.decode_queue = decode_queue
        self.requester.strong_decode_queue = strong_decode_queue
        self.verdict.decode_queue = decode_queue
        self.decoder_output.weak_store = weak_store
        self.decoder_output.frame = frame
        self.results.conditional_release = conditional_release
        self.results.factory = factory
        if self.gap_join is not None:
            self.gap_join.decode_queue = decode_queue
        if self.strong_redecode is not None:
            self.strong_redecode.strong_receiver = strong_receiver
            self.strong_redecode.strong_output = strong_output
            self.strong_redecode.decode_queue = strong_decode_queue

    def start(self) -> None:
        """Build every planned window's model, then its first boundary."""
        self.planner.start()
        self.window_manager.start()

    def check_settled(self) -> None:
        """No strong escalation is pending and every window has closed."""
        strong_redecode = self.strong_redecode
        if strong_redecode is not None and strong_redecode.has_pending():
            pending = strong_redecode.pending_work()
            raise RuntimeError(
                f"the run ended with pending strong escalations: {pending}"
            )
        self.window_manager.check_settled()

    def seed_roots(self) -> tuple:
        """This part's stochastic owners, each by the name its seed hashes."""
        return (("escalation_policy", self.escalation_policy),)

    def _wire_the_plan(self, plan: plan_build.Plan) -> None:
        interaction = plan.window_interaction
        self.models.provider = plan.error_model_provider
        self.planner.scheme = plan.scheme
        self.planner.models = self.models
        self.tracker.scheme = plan.scheme
        self.tracker.planner = self.planner
        self.gate.interaction = interaction
        self.builder.interaction = interaction
        self.courier.interaction = interaction
        self.courier.boundary_policy = plan.boundary_policy
        self.window_manager.window_interaction = interaction
        if self.regions is not None:
            self.regions.interaction = interaction

    def _wire_the_window_chain(self, links: ports.Link) -> None:
        planner = self.planner
        tracker = self.tracker
        retention = self.retention
        retention.planner = planner
        retention.tracker = tracker
        self.window_transfers.link = links
        self.window_transfers.retention = retention
        self.decoder_output.transfers = self.window_transfers
        self.decoder_output.link = links
        self.gate.planner = planner
        self.builder.planner = planner
        self.builder.tracker = tracker
        self.builder.gate = self.gate
        self.results.planner = planner
        self.results.tracker = tracker
        self.results.retention = retention
        self.results.ledger = self.ledger
        self.courier.planner = planner
        self.courier.transfers = self.window_transfers
        self.courier.retention = retention
        self.courier.windows = self.window_manager
        self.committer.courier = self.courier
        self.committer.decoder_output = self.decoder_output
        self.committer.results = self.results
        window_manager = self.window_manager
        window_manager.planner = planner
        window_manager.tracker = tracker
        window_manager.retention = retention
        window_manager.requester = self.requester
        window_manager.courier = self.courier
        window_manager.results = self.results

    def _wire_the_decision(self) -> None:
        verdict = self.verdict
        requester = self.requester
        verdict.planner = self.planner
        verdict.tracker = self.tracker
        verdict.escalation_policy = self.escalation_policy
        verdict.committer = self.committer
        requester.tracker = self.tracker
        requester.retention = self.retention
        requester.builder = self.builder
        requester.escalation_policy = self.escalation_policy
        requester.verdict = verdict
        requester.gap_join = self.gap_join
        if self.burst_detector is not None:
            self.escalation_policy.burst_detector = self.burst_detector
        if self.gap_join is not None:
            self.gap_join.signal = self.confidence_signal
            self.gap_join.verdict = verdict

    def _wire_the_strong_redecode(self) -> None:
        strong_redecode = self.strong_redecode
        if strong_redecode is None:
            return
        regions = self.regions
        regions.planner = self.planner
        regions.tracker = self.tracker
        regions.retention = self.retention
        regions.burst_detector = self.burst_detector
        shape = self.shape
        shape.regions = regions
        shape.planner = self.planner
        shape.retention = self.retention
        shape.builder = self.builder
        shape.requester = self.requester
        shape.ledger = self.ledger
        shape.courier = self.courier
        strong_redecode.shape = shape
        strong_redecode.pending = self.pending_strong_windows
        strong_redecode.retention = self.retention
        strong_redecode.decoder_output = self.decoder_output
        strong_redecode.verdict = self.verdict
        self.committer.strong_redecode = strong_redecode
        self.verdict.strong_redecode = strong_redecode
        self.window_manager.strong_redecode = strong_redecode


def _retention(
    plan: plan_build.Plan, escalation_policy: ports.EscalationPolicy
) -> round_retention_module.RoundRetention:
    """Which store holds a window's rounds, and for how long.

    The reads the run places while it goes hold the rounds before their
    first by the same rule as the reads the plan placed.
    """
    formation_reads = plan.formation_reads
    return round_retention_module.RoundRetention(
        is_strong_context_retained=escalation_policy.requires_strong_context,
        primary_tier=escalation_policy.primary_tier,
        strong_side_seat=formation_reads.strong_side_seat,
    )


def _confidence(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> tuple:
    """The signal a window's confidence is read from, and the join of solves.

    Two Nones on a run that decides on no confidence.
    """
    if not escalation_build.builds_a_confidence_signal(settings.escalation):
        return None, None
    signal = escalation_build.confidence_signal(
        settings.escalation, settings.weak_decoder
    )
    gap_join = gap_join_module.WindowGapJoin(engine)
    return signal, gap_join


def _verdict(
    settings: machine_settings.MachineSettings, engine: engine_module.Engine
) -> window_commits.WindowVerdict:
    """Whether a decoded window is kept, priced on the escalation's clock."""
    escalation = settings.escalation
    return window_commits.WindowVerdict(
        engine,
        clock=escalation.clock,
        threshold_cycles=escalation.threshold_cycles,
        switch_cycles=escalation.switch_cycles,
    )


def _strong_redecode(
    settings: machine_settings.MachineSettings,
    engine: engine_module.Engine,
    escalation_policy: ports.EscalationPolicy,
) -> tuple:
    """The strong tier's four components; four Nones if it never escalates.

    They are the strong regions, the strong window's shape, the ledger
    of pending strong windows and the strong re-decode. The shape is the
    row escalation.strong_window names: the redo window, or the double
    window of Toshio Sec. III C.
    """
    if not escalation_policy.requires_strong_context:
        return None, None, None, None
    shape_row = escalation_build.strong_window_row(settings.escalation)
    regions = strong_regions.StrongRegions()
    shape = shape_row(engine)
    pending = pending_strong_windows.PendingStrongWindows()
    strong_redecode = strong_redecode_module.StrongRedecode(engine)
    return regions, shape, pending, strong_redecode


def _copies_the_boundary_fold(
    settings: machine_settings.MachineSettings,
    escalation_policy: ports.EscalationPolicy,
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
