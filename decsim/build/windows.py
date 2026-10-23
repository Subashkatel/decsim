"""The windows part: which windows exist, when each is decoded, where it goes.

The window manager closes each window whose rounds are in, the requester
asks the decoder manager for a decode, the verdict keeps the result or
escalates it, and the committer writes the correction and carries the
boundary to the next window.
"""

import dataclasses
from typing import Optional

import decsim.build.layout as layout_module
import decsim.build.plan as plan_build
import decsim.confidence.gap_join as gap_join_module
import decsim.config as config
import decsim.decoders.decoder_output as decoder_output_module
import decsim.decoders.settings as decoder_settings
import decsim.engine as engine_module
import decsim.frontends.settings as workload_settings
import decsim.links.window_transfers as window_transfers_module
import decsim.ports as ports
import decsim.records.routes as route_records
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.committed_rounds as committed_rounds
import decsim.windows.decode_requests as decode_requests
import decsim.windows.escalate_edges as escalate_edges
import decsim.windows.operation_results as operation_results
import decsim.windows.round_retention as round_retention_module
import decsim.windows.round_tracker as round_tracker_module
import decsim.windows.routers as routers
import decsim.windows.settings as window_settings
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_commits as window_commits
import decsim.windows.window_manager as window_manager_module
import decsim.windows.window_planner as window_planner_module


@dataclasses.dataclass(frozen=True)
class Windows:
    """Every component that turns landed rounds into committed corrections.

    gap_join is None on a run that computes no confidence.
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
    router: routers.Router

    @classmethod
    def build(
        cls,
        windows_settings: window_settings.WindowSettings,
        workload: workload_settings.WorkloadSettings,
        layout: layout_module.Layout,
        machine_clock: Optional[config.Clock],
        engine: engine_module.Engine,
        plan: plan_build.Plan,
        links: ports.Link,
        built_models: built_window_models.BuiltWindowModels,
        recorded_signal: Optional[ports.ConfidenceSignal] = None,
    ) -> "Windows":
        """Every component of the window side, wired to one another.

        The entry pool decodes the planner's windows and the entry edge's
        target re-decodes them (layout). built_models is the window
        error model cache the shot reads and fills. recorded_signal is
        the confidence a run with no switching records
        (recorded_signal below), None for none.
        """
        run_plan = plan.run_plan
        models = window_planner_module.WindowModels(built_models)
        planner = window_planner_module.WindowPlanner(
            run_plan.resolved_operations,
            run_plan.execution,
            plan.planned_operations,
        )
        tracker = round_tracker_module.RoundTracker()
        entry = layout.entry_pool
        window_tier = entry.pool
        is_switching = bool(layout.edges)
        retention = _retention(plan, window_tier, is_switching)
        window_transfers = window_transfers_module.WindowTransfers(engine)
        routes = layout.routes()
        decoder_output = _decoder_output(
            layout.edges, routes, machine_clock, engine
        )
        copies_fold_by_pool = {}
        for pool_plan in layout.pools:
            copies = _copies_the_boundary_fold(pool_plan.settings)
            copies_fold_by_pool[pool_plan.pool] = copies
        gate = decode_requests.WindowInputGate(copies_fold_by_pool)
        router = _router(layout, plan.planned_operations)
        builder = decode_requests.DecodeRequestBuilder(engine)
        ledger = committed_rounds.LogicalLedger()
        results = operation_results.OperationResults()
        rule_settings = window_settings.release_rule_for(
            windows_settings, workload.patch_lives
        )
        results.release_rule = rule_settings.build(plan.operations)
        courier = window_boundaries.BoundaryCourier(routes)
        committer = window_commits.WindowCommitter(engine)
        verdict = window_commits.WindowVerdict(engine, window_tier)
        clocked_windows = config.with_machine_clock(
            windows_settings, machine_clock
        )
        requester = decode_requests.DecodeRequester(
            clock=clocked_windows.clock,
            decision_cycles=clocked_windows.decision_cycles,
        )
        gap_join = _gap_join(layout, recorded_signal, engine)
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
            router=router,
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
        target_decoders: tuple,
        decode_queue: ports.DecodeQueue,
        edges: escalate_edges.EscalateEdges,
        input_fold: ports.DecoderInputFold,
        frame: Optional[ports.Frame],
        conditional_release: ports.ReleaseReceiver,
        factory: ports.MagicStateFactory,
        detection_events: ports.DetectionEventPlacement,
        routed_sides: tuple = (),
    ) -> None:
        """Read from the stores, decode on the managers, commit to the frame.

        routed_sides are the entry pools past the first, each a
        routers.EntrySide, which the router hands each window it sends
        there.
        """
        self.models.decoder = primary_decoder
        for target_decoder in target_decoders:
            self.models.escalates_to(target_decoder)
        self.retention.weak_store = weak_store
        self.retention.strong_store = strong_store
        self.retention.detection_events = detection_events
        self.gate.input_fold = input_fold
        self.requester.store_output = store_output
        self.requester.decode_queue = decode_queue
        self.requester.edges = edges
        self.verdict.decode_queue = decode_queue
        self.verdict.edges = edges
        self.committer.edges = edges
        self.window_manager.edges = edges
        self.decoder_output.frame = frame
        self.results.conditional_release = conditional_release
        self.results.factory = factory
        if self.gap_join is not None:
            self.gap_join.decode_queue = decode_queue
        for side in routed_sides:
            self.router.add_side(side)

    def join_of(
        self, pool: window_records.PoolId
    ) -> Optional[gap_join_module.WindowGapJoin]:
        """The join of an entry pool's solves; None when it has none."""
        side = self.router.side_of(pool)
        if side is None:
            return self.gap_join
        return side.gap_join

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
        if plan.scheme.supports_dynamic_streams:
            self.planner.layouts = plan.scheme.stream_layouts()
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
        router = self.router
        router.tracker = self.tracker
        self.committer.router = router
        requester.router = router
        verdict.router = router
        self.courier.router = router
        self.retention.router = router


def recorded_signal(
    windows_settings: window_settings.WindowSettings,
    layout: layout_module.Layout,
) -> Optional[ports.ConfidenceSignal]:
    """The signal a run with no switching computes on each window, or None.

    It is built on the entry pool's algorithm, as a switching edge builds
    its signal on its source's; it reads no threshold, since it decides
    nothing.
    """
    confidence = windows_settings.recorded_confidence
    if confidence is None:
        return None
    if layout.edges:
        raise ValueError(
            "windows.recorded_confidence records a confidence on a run "
            "with no switching; a switching run records its edge's own "
            "(switching.confidence)"
        )
    if len(layout.entry_pools) > 1:
        raise ValueError(
            "windows.recorded_confidence records the entry pool's "
            "windows, and this run routes its windows to "
            f"{len(layout.entry_pools)} entry pools"
        )
    entry = layout.entry_pool
    algorithm = entry.settings.algorithm
    return confidence.build(algorithm, None)


def _retention(
    plan: plan_build.Plan,
    window_tier: window_records.PoolId,
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


def _router(layout: layout_module.Layout, operations: tuple) -> routers.Router:
    """The router of the layout's entry pools, one_pool for a single one.

    Its rule must send every operation to one of them.
    """
    if layout.router is None:
        entry = layout.entry_pool
        rule = routers.OnePool.Settings(entry.pool.word)
        return routers.Router(rule)
    pool_names = []
    for entry in layout.entry_pools:
        pool_names.append(entry.pool.word)
    rule = layout.router.rule
    rule.check_operations(operations, tuple(pool_names))
    return layout.router.build()


def _gap_join(
    layout: layout_module.Layout,
    recorded_signal: Optional[ports.ConfidenceSignal],
    engine: engine_module.Engine,
) -> Optional[gap_join_module.WindowGapJoin]:
    """The join of the first entry pool's solves; None when it has no gap.

    An entry pool that escalates joins on its edge's signal, bound when
    the machine assembles; a recorded signal is bound here.
    """
    if recorded_signal is not None:
        gap_join = gap_join_module.WindowGapJoin(engine)
        gap_join.signal = recorded_signal
        return gap_join
    entry = layout.entry_pool
    if layout.edge_leaving(entry.pool) is None:
        return None
    return gap_join_module.WindowGapJoin(engine)


def _decoder_output(
    edges: tuple,
    routes: route_records.Routes,
    machine_clock: Optional[config.Clock],
    engine: engine_module.Engine,
) -> decoder_output_module.DecoderOutput:
    """The decoder side's sends, each answer on its generated route.

    A route that commits on an edge's source chip counts its cycles on
    that edge's switching clock, so a run with no clock for it is
    refused.
    """
    clock_by_asker = {}
    for edge in edges:
        switching = edge.settings
        hops = edge.hops
        clocked_switching = config.with_machine_clock(switching, machine_clock)
        route = routes.answer(hops.target, hops.source)
        _refuse_a_chip_commit_with_no_clock(
            edge, route, clocked_switching.clock
        )
        clock_by_asker[hops.source] = clocked_switching.clock
    return decoder_output_module.DecoderOutput(engine, routes, clock_by_asker)


def _refuse_a_chip_commit_with_no_clock(
    edge: layout_module.EdgePlan,
    route: route_records.AnswerRoute,
    clock: Optional[config.Clock],
) -> None:
    if route.commit_cycles == 0 or clock is not None:
        return
    route_name = edge.settings.strong_answer_route
    raise ValueError(
        f"{edge.name}.strong_answer_route {route_name!r} commits on the "
        f"source chip in clock cycles, so it needs {edge.name}.clock or "
        "the machine's clock"
    )


def _copies_the_boundary_fold(
    pool: Optional[decoder_settings.DecoderPoolSettings],
) -> bool:
    """Whether a pool folds its boundary mask into a copy of its input.

    The idle pool of a run with no decoder has no settings and folds
    nothing, so either answer holds; it copies.
    """
    if pool is None:
        return True
    return pool.copies_boundary_fold
