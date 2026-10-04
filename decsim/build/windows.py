"""The windows part: which windows exist, when each is decoded, where it goes.

The window manager closes each window whose rounds are in, the requester
asks the decoder manager for a decode, the verdict keeps the result or
escalates it, and the committer writes the correction and carries the
boundary to the next window.
"""

import dataclasses
from typing import Optional

import decsim.build.plan as plan_build
import decsim.confidence.gap_join as gap_join_module
import decsim.config as config
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.escalation.settings as escalation_settings
import decsim.frontends.settings as workload_settings
import decsim.links.window_transfers as window_transfers_module
import decsim.ports as ports
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.decode_requests as decode_requests
import decsim.windows.operation_results as operation_results
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_commits as window_commits
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner_module


@dataclasses.dataclass(frozen=True)
class Windows:
    """Every component that turns landed rounds into committed corrections.

    gap_join is None on a run with no switching.
    """

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
    gap_join: Optional[gap_join_module.WindowGapJoin]
    window_manager: window_manager_module.WindowManager

    @classmethod
    def build(
        cls,
        windows_settings: window_settings.WindowSettings,
        workload: workload_settings.WorkloadSettings,
        switching_settings: Optional[escalation_settings.SwitchingSettings],
        window_decoder: Optional[decoder_settings.DecoderPoolSettings],
        strong_decoder: Optional[decoder_settings.DecoderPoolSettings],
        window_tier: window_records.DecoderTier,
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        plan: plan_build.Plan,
        links: ports.Link,
        built_models: built_window_models.BuiltWindowModels,
    ) -> "Windows":
        """Every component of the window side, wired to one another.

        built_models is the window error model cache the shot reads and
        fills.
        """
        run_plan = plan.run_plan
        models = window_planner_module.WindowModels(built_models)
        planner = window_planner_module.WindowPlanner(
            run_plan.resolved_operations,
            run_plan.execution,
            plan.planned_operations,
        )
        tracker = round_tracker_module.RoundTracker()
        is_switching = switching_settings is not None
        retention = _retention(plan, window_tier, is_switching)
        window_transfers = window_transfers_module.WindowTransfers(engine)
        decoder_output = decoder_output_module.DecoderOutput(engine)
        copies_the_window_fold = _copies_the_boundary_fold(window_decoder)
        copies_the_strong_fold = _copies_the_boundary_fold(strong_decoder)
        gate = decode_requests.WindowInputGate(
            copies_the_window_fold, copies_the_strong_fold
        )
        builder = decode_requests.DecodeRequestBuilder(engine)
        ledger = committed_rounds.LogicalLedger()
        results = operation_results.OperationResults()
        courier = window_boundaries.BoundaryCourier()
        committer = window_commits.WindowCommitter(engine)
        verdict = _verdict(switching_settings, machine_clock, engine)
        clocked_windows = config.with_machine_clock(
            windows_settings, machine_clock
        )
        requester = decode_requests.DecodeRequester(
            clock=clocked_windows.clock,
            decision_cycles=clocked_windows.decision_cycles,
        )
        gap_join = _gap_join(is_switching, engine)
        feedback_boundary_mode = workload.feedback_boundary_mode
        window_manager = window_manager_module.WindowManager(
            engine, feedback_boundary_mode=feedback_boundary_mode
        )
        windows = cls(
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
            gap_join=gap_join,
            window_manager=window_manager,
        )
        windows._wire_the_plan(plan)
        windows._wire_the_window_chain(links)
        windows._wire_the_decision()
        return windows

    def connect(
        self,
        weak_store: Optional[ports.SyndromeBuffer],
        strong_store: Optional[ports.SyndromeBuffer],
        store_output: Optional[ports.SyndromeBufferOutput],
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

    def start(self) -> None:
        """Build every planned window's model, then its first boundary."""
        self.planner.start()
        self.window_manager.start()

    def check_settled(self) -> None:
        """Every window has closed."""
        self.window_manager.check_settled()

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
        verdict.committer = self.committer
        requester.tracker = self.tracker
        requester.retention = self.retention
        requester.builder = self.builder
        requester.verdict = verdict
        requester.gap_join = self.gap_join
        if self.gap_join is not None:
            self.gap_join.verdict = verdict


def _retention(
    plan: plan_build.Plan,
    window_tier: window_records.DecoderTier,
    is_switching: bool,
) -> round_retention_module.RoundRetention:
    """Which store holds a window's rounds, and for how long.

    A switching run keeps every round a strong redo would read.
    """
    formation_reads = plan.formation_reads
    return round_retention_module.RoundRetention(
        is_strong_context_retained=is_switching,
        primary_tier=window_tier,
        strong_side_seat=formation_reads.strong_side_seat,
    )


def _gap_join(
    is_switching: bool, engine: engine_module.Engine
) -> Optional[gap_join_module.WindowGapJoin]:
    """The join of a window's solves; None on a run with no switching."""
    if not is_switching:
        return None
    return gap_join_module.WindowGapJoin(engine)


def _verdict(
    switching: Optional[escalation_settings.SwitchingSettings],
    machine_clock: Optional[config.Clock],
    engine: engine_module.Engine,
) -> window_commits.WindowVerdict:
    """Whether a decoded window is kept, priced on the switching clock.

    A run with no switching keeps every result at once.
    """
    if switching is None:
        return window_commits.WindowVerdict(engine)
    clocked_switching = config.with_machine_clock(switching, machine_clock)
    return window_commits.WindowVerdict(
        engine,
        clock=clocked_switching.clock,
        threshold_cycles=clocked_switching.threshold_cycles,
        switch_cycles=clocked_switching.switch_cycles,
    )


def _copies_the_boundary_fold(
    decoder: Optional[decoder_settings.DecoderPoolSettings],
) -> bool:
    """Whether a tier folds its boundary mask into a copy of its input."""
    if decoder is None:
        return True
    return decoder.copies_boundary_fold
