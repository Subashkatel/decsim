"""Interfaces for the simulator parts that users may replace.

Each protocol describes one construction seam used by ``RunSpec``. Runtime
state and implementation logic belong in the implementing modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (Any, Callable, Iterable, Mapping, Optional, Protocol,
                    runtime_checkable)

from .message import (BoundaryDelivery, BoundaryUpdate, DecodeJob, Directive, OutcomeDirective, Submission,
                      DecoderRequestKey,
                      DecodeOutcome, DecodeResult, OperationPlanningView,
                      ResolvedCodeGeometry, RunSeedChild, RunSeedReservation,
                      QPUReadout, StrongRegionPlan, SyndromePacketRoute,
                      SyndromePayload,
                      Window, WindowInfo, WindowReadiness)


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


# ------------------------------------------------------ escalation policy seam

@runtime_checkable
class EscalationServices(Protocol):
    """Strong-job construction and strong-result selection offered to an escalation policy."""

    def make_strong_job(self, weak_job: DecodeJob, n_rounds: int,
                        label: str) -> DecodeJob: ...

    def defer_strong_escalation(
        self, weak_job: DecodeJob,
    ) -> DecoderRequestKey: ...

    def check_strong_route(
        self, weak_job: DecodeJob, strong_job: DecodeJob,
    ) -> None: ...

    def prepare_strong_selection(
        self, weak_job: DecodeJob, strong_request_key: DecoderRequestKey,
        serial_strong_job: Optional[DecodeJob], *, deferred: bool,
    ) -> int: ...


@runtime_checkable
class EscalationPolicy(Protocol):
    """Whether and when a window is decoded again by the strong tier: which
    jobs to submit when a window is ready (Baseline: the weak job only), and
    what to do with each outcome (accept it, escalate it, hold it). For a weak
    job, on_decode_outcome runs BEFORE the core's commit bookkeeping, so its
    directive decides whether the result is held awaiting a strong redo."""

    requires_strong_context: bool
    bulk_strong: bool
    double_window: bool

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
                        services: EscalationServices) -> list[Submission]: ...

    def on_decode_outcome(self, outcome: DecodeOutcome,
                          services: EscalationServices) -> OutcomeDirective: ...


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
    """How idle rounds travel while an op waits for feedback (see
    controller/policies.py for the three built-in policies). relay() carries
    one idle round through the controller it is given."""

    def relay(self, controller, operation, patch, round_index: int) -> None: ...


# -------------------------------------------------------------- decode stage

@runtime_checkable
class Decoder(Protocol):
    """Port 8. Joint decode+latency: correctness and timing from one call.

    Contract (decoder_manager): ``latency(job)`` is called ONCE, at
    dispatch, and returns the whole job's service time in ticks; the
    manager schedules completion that many ticks later and then calls
    ``decode(job)`` for the result. Timing evaluation must not mutate routing
    or accuracy-bearing job state; functional decisions belong to decode and
    escalation policy owners.

    A decoder that simulates its own internal stages also offers
    ``run(job, engine, on_done)``: the manager calls it instead of scheduling
    ``latency(job)``, the decoder walks its stages as engine events on the
    unit the manager granted, calls ``on_done`` once when its output is
    released, and ``decode(job)`` then returns the released result."""

    def decode(self, job: DecodeJob) -> DecodeResult: ...

    def latency(self, job: DecodeJob) -> int: ...


@runtime_checkable
class Scheduler(Protocol):
    """Port 11. Select the next ready job from one decoder-pool queue."""

    def pop(self, queue: list[DecodeJob]) -> DecodeJob: ...


@runtime_checkable
class DecoderMemoryTransfer(Protocol):
    """Port 22. Carry one admitted job to the decoder side after a delay.

    Implementations call ``receiver(job)`` exactly once after ``delay_ticks``,
    unless the request is cancelled first. ``cancel(job)`` is idempotent: the
    receiver is never invoked for a request cancelled before its delivery, and
    cancelling an unknown or already delivered request does nothing. Storage
    admission, materialization, stored-input lifetime, link reservation,
    admission, service, and result handling belong elsewhere.
    """

    def deliver(
        self, job: DecodeJob, delay_ticks: int,
        receiver: Callable[[DecodeJob], None],
    ) -> None: ...

    def cancel(self, job: DecodeJob) -> None: ...


@runtime_checkable
class PauliFrame(Protocol):
    """Port 23. Optional final-weak correction sink, inert when absent.

    The caller supplies a stable ``(op_id, window_id)`` identity, the delivered
    logical observables, their decoder-request provenance, and a zero-argument
    continuation. Implementations call the continuation at most once per call
    and exactly once per accepted write, never before the configured write cost
    has elapsed. They never mutate caller-owned values or call the conditional release,
    controller, or execution runtime. ``snapshot`` is immutable and does not
    mutate the frame.
    """

    def commit_weak_correction(
        self,
        *,
        window_key,
        logical_observables,
        request_key,
        on_committed: Callable[[], None],
    ) -> None: ...

    def snapshot(self): ...


@runtime_checkable
class ResourcePool(Protocol):
    """Port 12. Admit decoder jobs and own decoder queues and service units.

    A job may be submitted once. The pool also owns strong-request cancellation
    and the final check that no decoder work is stranded.
    """

    def enqueue(self, job: DecodeJob, reserve_transfer=None) -> None: ...   # reserve_transfer() at dispatch -> transfer ticks

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "") -> None: ...

    def cancel_strong(self, key: tuple) -> None: ...

    def check_decode_work_settled(self) -> None: ...


# ------------------------------------------------------------------ dataflow


@runtime_checkable
class SyndromeDevice(Protocol):
    """Unclocked physical source used only by ``QPUDevice``.

    Payload bits are raw measurement bits per round. A source that carries a
    detector formation table also offers ``form_round(operation_id, round_index,
    raw_bits)``, which Buffer 0 intake calls once per complete round.
    """

    operation_circuit_scope: str
    def begin_operation(
        self, op, segment_round_count: int, source_round_count: int,
    ) -> None: ...
    def round_payloads(self, op, round_index: int) -> list[QPUReadout]: ...
    def finalize_stream_round(
        self, op, source_round_count: int,
    ) -> list[QPUReadout]: ...
    def idle_round_payloads(
        self, op, stream_id, global_round: int, patch,
    ) -> list[QPUReadout]: ...


@runtime_checkable
class ErrorModelProvider(Protocol):
    """Build decoder-facing models without owning physical QPU cadence."""

    def register_dynamic_stream(
        self, stream_op, round_count: int, *, fault_model_requirement,
    ): ...
    def validate_stream_length(
        self, stream_op, stream_round_count: int,
    ) -> None: ...
    def window_models_for_operation(
        self, op, windows: list, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
        window_protocol,
    ) -> list: ...
    def window_model_for_stream(self, stream_id, window): ...
    def strong_window_model_for_operation(
        self, op, window, round_count: int, *,
        fault_model_requirement, exclude_faults_touching=None,
    ): ...


@runtime_checkable
class MultiFaultExclusionSyndromeDevice(Protocol):
    """Optional device capability for disjoint fault-exclusion ranges."""

    def strong_window_model_for_operation_with_exclusions(
        self, op, window, round_count: int, *,
        fault_model_requirement, fault_exclusion_ranges: tuple,
    ): ...


@runtime_checkable
class SyndromeTransport(Protocol):
    """Port 14. Reassembles and forwards transient syndrome packets."""
    def relay_qpu_readout(
        self, payload: SyndromePayload, route: SyndromePacketRoute, *,
        processing_ticks: int,
    ) -> None: ...


@runtime_checkable
class ConditionalReleasePort(Protocol):
    """Receives each operation's final result and releases the operations
    conditioned on it (the OC hop). pauli_frame/conditional_release.py."""
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

    window_floor_justification: Optional[str]

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
class WindowingScheme(Protocol):
    """How an operation's rounds are cut into windows: the static window
    graph of an operation, when a window has its data, and the buffer floor
    the scheme needs."""

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ): ...

    def data_complete(self, window: Window, *, readiness: WindowReadiness,
                      operation: OperationPlanningView) -> bool: ...

    def validate_buffer(self, geometry: ResolvedCodeGeometry) -> None: ...


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
    """Port 18. Observes retained payload storage inside SyndromeBuffer:
    store()/evict() fire on exactly the fragments held. Optional — when
    absent, storage is unbounded."""

    def store(self, key, payload) -> None: ...

    def evict(self, key) -> None: ...
