"""One Protocol per pluggable piece of the simulator.

Every swappable part — decoder, scheduler, windowing scheme, controller,
and so on — plugs into the pipeline through exactly one interface below.
RunSpec.build() picks one implementation per port; the port numbers are
the stable names tests and docstrings use to refer to these seams.

This module declares interfaces only (plus the small strategy seam types
Submission / Directive / OutcomeDirective that DecodingStrategy hooks use
to answer the core). It imports nothing from decsim except message types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (Any, Callable, Iterable, Mapping, Optional, Protocol,
                    runtime_checkable)

from .message import (BoundaryDelivery, BoundaryUpdate, DecodeJob,
                      DecodeOutcome, DecodeResult, OperationPlanningView,
                      ResolvedCodeGeometry, RunSeedChild, RunSeedReservation,
                      StrongRegionPlan, SyndromePayload, SyndromeRoundPacket,
                      Window, WindowInfo)


# ------------------------------------------------------ run seed capabilities

@runtime_checkable
class RunSeedConsumer(Protocol):
    """A stochastic runtime owner with a two-phase seed transaction.

    The root reserves every consumer before committing any consumer.  A
    successful reservation therefore establishes a failure-free commit phase:
    preparation that can fail belongs in ``reserve_run_seed``, not in
    ``commit_run_seed``.
    """

    def reserve_run_seed(
        self,
        seed: Optional[int],
    ) -> RunSeedReservation:
        """Prepare replacement state under an exclusive reversible claim.

        This method may fail, but it must not change the active random state or
        draw from it.  It performs all potentially failing preparation and
        returns the exact pending reservation that commit or cancel consumes.
        """
        ...

    def commit_run_seed(
        self,
        reservation: RunSeedReservation,
    ) -> None:
        """Install an exact reservation in the failure-free commit phase.

        For a reservation returned successfully by this consumer, commit must
        be total and must not fail, allocate, draw randomness, invoke a
        callback, or perform validation that can reject.  It may acquire the
        consumer's private lock and install the already-prepared state.
        """
        ...

    def cancel_run_seed(
        self,
        reservation: RunSeedReservation,
    ) -> None:
        """Release the exact pending reservation without changing RNG state.

        Cancellation must be total for the exact pending reservation returned
        by this consumer.
        """
        ...


@runtime_checkable
class RunSeedComposite(Protocol):
    """A runtime owner that exposes semantic stochastic child edges."""

    def run_seed_children(self) -> Iterable[RunSeedChild]: ...


@runtime_checkable
class RunManifestPart(Protocol):
    """A component that declares JSON-safe effective configuration."""

    def run_manifest_config(self) -> Mapping[str, Any]: ...


# --------------------------------------------------------------- strategy seam

@dataclass
class Submission:
    """One decode job a strategy wants enqueued, optionally after a delay
    (e.g. the weak->strong ws hop). A strong redo job gets its
    ready_time stamped when it reaches the ready queue, so the delay does not
    count as queue wait. Its configured deadline remains the physical
    obligation stamped from the source window before the hop."""

    job: DecodeJob
    delay_ticks: int = 0


class Directive(Enum):
    """What the core should do with a decode outcome."""
    FINALIZE = auto()          # accept the weak result; cancel any parallel strong
    AWAIT_STRONG = auto()      # hold the weak result; .extra may carry the strong redo
    FINALIZE_STRONG = auto()   # a strong result landed; core applies hold-or-deliver


@dataclass
class OutcomeDirective:
    """A strategy's verdict on one decode outcome."""
    directive: Directive
    extra: Optional[Submission] = None


@runtime_checkable
class StrategyServices(Protocol):
    """What the core offers a strategy inside its hooks: the clock, strong-job
    construction/cancellation, and strong-result selection transport."""

    @property
    def now(self) -> int: ...

    def make_strong_job(self, weak_job: DecodeJob, n_rounds: int,
                        label: str) -> DecodeJob: ...

    def defer_strong_escalation(
        self, weak_job: DecodeJob,
    ) -> None: ...

    def check_strong_route(
        self, weak_job: DecodeJob, strong_job: DecodeJob,
    ) -> None: ...

    def cancel_strong(self, key: tuple) -> None: ...

    def prepare_strong_selection(
        self, weak_job: DecodeJob, serial_submission: Optional[Submission],
    ) -> int: ...


@runtime_checkable
class DecodingStrategy(Protocol):
    """Port 10. Decides how each window gets decoded: which jobs to submit
    when a window is ready, and what to do with each outcome (accept it,
    escalate it, hold it). For a weak job, on_decode_outcome runs BEFORE
    the core's commit bookkeeping, so its directive decides whether the
    result is held awaiting a strong redo."""

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams: bool,
        static_decode_plan_selected: bool,
        has_frontend: bool,
    ) -> None: ...

    def validate_operations(
        self,
        operations: tuple[OperationPlanningView, ...],
    ) -> None: ...

    def validate_code_geometry(
        self,
        geometry: ResolvedCodeGeometry,
    ) -> None: ...

    def on_window_ready(self, window: Window, weak_job: DecodeJob,
                        services: StrategyServices) -> list[Submission]: ...

    def on_decode_outcome(self, outcome: DecodeOutcome,
                          services: StrategyServices) -> OutcomeDirective: ...

    def metrics(self) -> dict: ...


# ----------------------------------------------------------- runtime policies

@runtime_checkable
class BoundaryPolicy(Protocol):
    """Port 16. Decides when a committed window may ship its boundary
    defects to dependent windows: True = ship now. Eager ships at every
    weak commit (the default); Held ships only once the result is final.
    Implementations may set ``speculative = True`` to request recovery when
    an eagerly sent boundary is later revised. The attribute is optional for
    compatibility; absence means non-speculative delivery."""

    def on_commit(self, window: Window, *, final: bool) -> bool: ...


@runtime_checkable
class WindowInteraction(Protocol):
    """Port 21. Decisions relating adjacent or replaced windows.

    Implementations return data and immutable decisions; the window manager
    remains the sole owner of event ordering, lifecycle mutation, retention,
    logical accounting, and finality.
    """

    def initial_boundary_state(self, window: WindowInfo) -> Any: ...

    def boundary_from_result(
        self, result: Optional[DecodeResult], fallback: Any,
    ) -> Any: ...

    def boundaries_equal(self, left: Any, right: Any) -> bool: ...

    def boundary_targets(
        self, source: WindowInfo, windows: Mapping[tuple, WindowInfo],
    ) -> list:
        """Select unstarted destinations from the source's declared edges."""
        ...

    def merge_boundary(
        self,
        delivery: BoundaryDelivery,
        destination: WindowInfo,
        current_state: Any,
    ) -> BoundaryUpdate: ...

    def apply_boundary(
        self,
        state: Any,
        window: WindowInfo,
        payload,
        round_key: int,
    ): ...

    def invalidated_windows(
        self, source_key: tuple, windows: Mapping[tuple, WindowInfo],
    ) -> list:
        """Select replay roots; the runtime adds their dependent closure."""
        ...

    def plan_strong_region(
        self,
        weak_window: WindowInfo,
        later_windows: list[WindowInfo],
        operation_round_count: int,
    ) -> Optional[StrongRegionPlan]: ...


@runtime_checkable
class IdlePolicy(Protocol):
    """Port 17. How idle rounds are handled while an op waits for feedback
    (see policies.py for the three modes). The reaction gate branches
    on .mode; account() records the idle rounds emitted for an op."""

    mode: str

    def account(self, idle_rounds: int, op) -> None: ...


@runtime_checkable
class DeadlinePolicy(Protocol):
    """Port 13. Stamps DecodeJob.deadline when the job is built; the EDF
    scheduler dispatches by it. Window.t_first_round may be absent; policies
    that require arrival provenance must reject that incomplete input."""

    def deadline(self, op, window: Window, now: int, *,
                 on_reaction_path: bool) -> int: ...


# -------------------------------------------------------------- decode stage

@runtime_checkable
class Decoder(Protocol):
    """Port 8. Joint decode+latency: correctness and timing from one call.

    Contract (decoder_manager): ``latency(job)`` is called ONCE, at
    dispatch, and returns the whole job's service time in ticks; the
    manager schedules completion that many ticks later and then calls
    ``decode(job)`` for the result. Timing evaluation must not mutate routing
    or accuracy-bearing job state; functional decisions belong to decode and
    strategy owners."""

    def decode(self, job: DecodeJob) -> DecodeResult: ...

    def latency(self, job: DecodeJob) -> int: ...


@runtime_checkable
class DecoderRouter(Protocol):
    """Port 9. Selects a decoder for each job."""

    needs_hyperedges: bool

    def route(self, job: DecodeJob) -> Decoder: ...


@runtime_checkable
class Scheduler(Protocol):
    """Port 11. Queue discipline for one decode lane (FIFO or EDF):
    insert() places a job in the ready queue, and pop() picks the next job
    using the dispatch owner's exact current tick."""

    def insert(self, queue: list, job: DecodeJob) -> None: ...

    def pop(self, queue: list, now_ticks: int) -> DecodeJob: ...


@runtime_checkable
class ResourcePool(Protocol):
    """Port 12. The decode units and their ready queues (implemented by
    DecoderManager).

    A DecodeJob is submitted once. enqueue raises, before touching any state,
    on a job it has already admitted, completed or cancelled; admitted spans
    the whole time the job holds a queue slot or a unit, from the moment
    enqueue accepts it through crossing the weak->strong link, queueing and
    execution. A refused submission leaves the job and the pool untouched, so
    a strong request refused as a duplicate may be built again and submitted
    once the destination's result is consumed.

    A destination window owns at most one unconsumed strong result, either a
    live request or a completion held for a demand that has not registered
    yet. enqueue raises on a second strong request while the first is
    unconsumed, and checks nothing else about the destination: whether a
    result will be consumed is decided when it completes, so a strategy may
    return its Submissions in any order and may cancel and replace a request
    from any hook position.

    A completing strong result is delivered to its destination if the
    destination registered a strong demand (AWAIT_STRONG); held if that
    destination's weak decode is still open and may raise one; and otherwise
    raises, because nothing would consume it.

    That ownership is per destination window, not per decode attempt, so it
    needs at most one of a destination's weak decodes open at a time: with two,
    either decode's directive consumes whichever result the destination owns
    and the other attempt is left waiting for one that never comes. enqueue
    therefore refuses a second weak decode for a destination whose first has
    not yet produced a directive. A decode produces its directive by returning
    from on_decode_outcome, so the refusal covers that whole call: a strategy
    calling enqueue from inside its own outcome hook is refused a second weak
    decode of the destination it is deciding, and the destination reopens
    before the returned directive is applied. Per-attempt ownership, which
    would let a destination decode twice at once, needs the attempt carried
    through the request, the hold and the demand, and this port does not carry
    it.

    Two shipped components keep clear of that refusal, and both are
    load-bearing: the window manager queues a window once and re-queues it only
    after its decode has produced a directive, and the Switching strategy lists
    one weak Submission per window. A new strategy is as able to break the rule
    as a change to the window manager.

    check_decode_work_settled raises when the run has gone quiescent with any
    destination still recorded as decoding, waiting for a strong result,
    holding an unclaimed one, or holding a strong request. Each of those is a
    window that never became final, which no metric or view would otherwise
    report.

    cancel_strong is atomic at the event-queue pop: a queued job, or one still
    crossing the weak->strong link, is removed outright and never dispatched;
    an executing job is marked cancelled, releases its modeled unit at the
    cancel, and delivers nothing on completion; a held completion is
    discarded. A batched strong decode with siblings that are still wanted
    keeps running and drops only the cancelled key from delivery. A cancel
    ends one request and passes no verdict on the destination, which may be
    given a replacement."""

    def enqueue(self, job: DecodeJob, delay_ticks: int = 0) -> None: ...

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "") -> None: ...

    def cancel_strong(self, key: tuple) -> None: ...

    def try_dispatch(self) -> None: ...

    def check_decode_work_settled(self) -> None: ...


# ------------------------------------------------------------------ dataflow

@runtime_checkable
class SyndromeSource(Protocol):
    """Port 2. Clocked source of syndrome rounds: start() begins emitting
    an op's rounds on the round clock. Idle rounds are the Chip's job —
    it only asks the source for their payloads (idle_round_payloads)."""

    def start(self, op, round_ticks: int,
              on_body_done: Callable[[Any], None]) -> None: ...

    def idle_round_payloads(self, op, stream_id, global_round: int,
                            patch) -> list: ...


@runtime_checkable
class SyndromeDevice(Protocol):
    """The unclocked syndrome and error-model source supplied to RunSpec."""

    operation_circuit_scope: str

    def begin_operation(self, op, resolved_round_count: int) -> None: ...

    def round_payloads(self, op, round_index: int) -> list: ...

    def idle_round_payloads(
        self, op, stream_id, global_round: int, patch,
    ) -> list: ...

    def register_dynamic_stream(
        self, stream_op, round_count: int, *, belief_matching: bool = False,
    ): ...

    def validate_stream_length(
        self, stream_op, stream_round_count: int,
    ) -> None: ...

    def window_models_for_operation(
        self, op, windows: list, round_count: int, *,
        belief_matching: bool = False,
    ) -> list: ...

    def window_model_for_stream(
        self, stream_id, window, *, is_last: bool,
    ): ...

    def strong_window_model_for_operation(
        self, op, window, round_count: int, *,
        belief_matching: bool = False, exclude_faults_touching=None,
    ): ...


@runtime_checkable
class MultiFaultExclusionSyndromeDevice(Protocol):
    """Optional device capability for disjoint fault-exclusion ranges."""

    def strong_window_model_for_operation_with_exclusions(
        self, op, window, round_count: int, *,
        belief_matching: bool = False, fault_exclusion_ranges: tuple,
    ): ...


@runtime_checkable
class Controller(Protocol):
    """Port 14. Reserves and delivers QC/CWD and OC/CQ reaction-path hops
    through the exact run-owned semantic link fabric."""

    links: Any

    def relay_syndrome(
        self,
        payload: SyndromePayload,
        deliver: Callable[[SyndromeRoundPacket], None],
    ) -> None: ...

    def relay_instruction(self, decision, deliver: Callable) -> None: ...

@runtime_checkable
class Orchestrator(Protocol):
    """Port 15. Turns final decoded measurements into Decisions (the Pauli
    byproduct / S-gate algebra) and releases the operations they block."""

    engine: Any
    frame: Any

    def connect(self, controller, decision_sink: Callable) -> None: ...

    def register_blocked_operation(self, blocked_op_id: int,
                                   blocking_op_id: int) -> None: ...

    def integrate(self, op, outcome) -> None: ...


# ---------------------------------------------------------------- compile side

@runtime_checkable
class InputFrontend(Protocol):
    """Port 1 (compile-time). Builds the list of Operations to simulate."""

    def build(self) -> list: ...


@runtime_checkable
class CodeModel(Protocol):
    """Port 3. Window sizes, cycle length, graph size, and syndrome width."""

    name: str
    distance: int

    def rounds_per_logical_cycle(self) -> int: ...

    def round_period_us(self) -> Optional[float]: ...

    def commit_rounds(self) -> int: ...

    def buffer_rounds(self) -> int: ...

    def buffering_floor(self) -> tuple[int, int]: ...

    def buffer_floor_override_active(self) -> bool: ...

    def spatial_nodes(self, num_patches: int) -> int: ...

    def syndrome_bits_per_round(self, num_patches: int) -> int: ...


@runtime_checkable
class LayoutModel(Protocol):
    """Port 4. Maps operations and patches to their codes, decoding-graph
    sizes, and resource claims.

    ``codes()``, ``code_for_op()``, and ``code_for_patch()`` must remain stable
    for a build. The current runtime accepts one declared planning/runtime code,
    and every reachable selector must return that exact object.
    """

    def code_for_op(self, op): ...

    def code_for_patch(self, patch_id): ...

    def codes(self) -> list: ...

    def spatial_nodes_for(
        self,
        operation,
        *,
        base_spatial_node_count: int,
    ) -> int: ...

    def patch_spatial_nodes_for(
        self,
        patch_identity,
        *,
        base_spatial_node_count: int,
    ) -> int: ...

    def resources_for(self, op) -> list: ...


@runtime_checkable
class RoundsPolicy(Protocol):
    """Port 5. How many syndrome rounds an op runs for. Must return >= 1."""

    def rounds_for(self, op, code) -> int: ...


@runtime_checkable
class DecodingScheme(Protocol):
    """Port 6. Complete static topology plus runtime readiness."""

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ): ...

    def data_complete(self, window: Window, *, rounds_arrived: int,
                      successor_rounds: int, memory_rounds: int,
                      round_count: int, has_successor: bool,
                      operation: OperationPlanningView) -> bool: ...

    def validate_buffer(self, geometry: ResolvedCodeGeometry) -> None: ...


@runtime_checkable
class CrossPartValidator(Protocol):
    """Optional exact capability for parts that reject whole-run combinations."""

    def validate(self, spec, planning) -> None: ...


# ----------------------------------------------------------------- resources

@runtime_checkable
class MagicStateFactory(Protocol):
    """Port 19. Async magic-state supply: request() calls back once a
    distilled state is ready for the op on its declared event engine."""

    engine: Any

    def request(self, op_id: int, callback: Callable[[], None]): ...

    def shutdown(self) -> None: ...


@runtime_checkable
class Metric(Protocol):
    """Port 20. Observes typed views of the run and reports one result;
    never mutates what it observes."""

    name: str
    result_schema_version: int

    def observe(self, view) -> None: ...

    def result(self) -> Any: ...


@runtime_checkable
class MemoryModel(Protocol):
    """Port 18. Observes physical payload storage inside the PayloadStore:
    store()/evict() fire on exactly the fragments held. Optional — when
    absent, storage is unbounded."""

    def store(self, key, payload) -> None: ...

    def evict(self, key) -> None: ...
