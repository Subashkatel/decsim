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
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from .message import (BoundaryDelivery, BoundaryUpdate, DecodeJob,
                      DecodeOutcome, DecodeResult, StrongRegionPlan, Window,
                      WindowInfo)


# --------------------------------------------------------------- strategy seam

@dataclass
class Submission:
    """One decode job a strategy wants enqueued, optionally after a delay
    (e.g. the weak->strong ws hop). A strong redo job gets its
    ready_time/deadline re-stamped when it actually enqueues, so the
    delay does not count against its queue-wait accounting."""

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
    construction/cancellation, and the weak->strong link delay."""

    @property
    def now(self) -> int: ...

    def make_strong_job(self, weak_job: DecodeJob, n_rounds: int,
                        label: str) -> DecodeJob: ...

    def defer_strong_escalation(
        self, weak_job: DecodeJob, n_rounds: int, label: str,
    ) -> None: ...

    def check_strong_route(
        self, weak_job: DecodeJob, strong_job: DecodeJob,
    ) -> None: ...

    def cancel_strong(self, key: tuple) -> None: ...

    def ws_delay(self) -> int: ...


@runtime_checkable
class DecodingStrategy(Protocol):
    """Port 10. Decides how each window gets decoded: which jobs to submit
    when a window is ready, and what to do with each outcome (accept it,
    escalate it, hold it). For a weak job, on_decode_outcome runs BEFORE
    the core's commit bookkeeping, so its directive decides whether the
    result is held awaiting a strong redo."""

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
        strong_round_count: int,
        operation_round_count: int,
        buffer_round_count: int,
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
    scheduler dispatches by it."""

    def deadline(self, op, window: Window, now: int, *,
                 on_reaction_path: bool) -> int: ...


# -------------------------------------------------------------- decode stage

@runtime_checkable
class Decoder(Protocol):
    """Port 8. Joint decode+latency: correctness and timing from one call.

    Contract (decoder_manager): ``latency(job)`` is called ONCE, at
    dispatch, and returns the whole job's service time in ticks; the
    manager schedules completion that many ticks later and then calls
    ``decode(job)`` for the result. Implementations may mutate the job
    in latency() to steer routing (SwitchingDecoder sets job.hint)."""

    def decode(self, job: DecodeJob) -> DecodeResult: ...

    def latency(self, job: DecodeJob) -> int: ...


@runtime_checkable
class DecoderRouter(Protocol):
    """Port 9. Selects a decoder for each job."""

    def route(self, job: DecodeJob) -> Decoder: ...


@runtime_checkable
class Scheduler(Protocol):
    """Port 11. Queue discipline for one decode lane (FIFO or EDF):
    insert() places a job in the ready queue, pop() picks the next
    job to dispatch."""

    def insert(self, queue: list, job: DecodeJob) -> None: ...

    def pop(self, queue: list) -> DecodeJob: ...


@runtime_checkable
class ResourcePool(Protocol):
    """Port 12. The decode units and their ready queues (implemented by
    DecoderManager). cancel_strong is atomic at the event-queue pop:
    a queued job is removed outright; an executing job finishes but its
    result is discarded."""

    def enqueue(self, job: DecodeJob, delay_ticks: int = 0) -> None: ...

    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = "") -> None: ...

    def cancel_strong(self, key: tuple) -> None: ...

    def try_dispatch(self) -> None: ...


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

    def begin_operation(self, op) -> None: ...

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
    """Port 14. Delivers messages across the classical network one named
    hop at a time (edges qc, cd, dd, do, oc, cq, ws — priced by
    TimingConfig)."""

    def relay_syndrome(self, payload, deliver: Callable) -> None: ...

    def relay_instruction(self, decision, deliver: Callable) -> None: ...

@runtime_checkable
class Orchestrator(Protocol):
    """Port 15. Turns final decoded measurements into Decisions (the Pauli
    byproduct / S-gate algebra) and releases the operations they block."""

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

    def commit_rounds(self) -> int: ...

    def buffer_rounds(self) -> int: ...

    def spatial_nodes(self, num_patches: int) -> int: ...

    def syndrome_bits_per_round(self, num_patches: int) -> int: ...


@runtime_checkable
class LayoutModel(Protocol):
    """Port 4. Maps operations and patches to their codes, decoding-graph
    sizes, and resource claims."""

    def code_for_op(self, op): ...

    def code_for_patch(self, patch_id): ...

    def codes(self) -> list: ...

    def spatial_nodes_for(self, op) -> int: ...

    def resources_for(self, op) -> list: ...


@runtime_checkable
class RoundsPolicy(Protocol):
    """Port 5. How many syndrome rounds an op runs for. Must return >= 1."""

    def rounds_for(self, op, code) -> int: ...


@runtime_checkable
class DecodingScheme(Protocol):
    """Port 6. The windowing discipline: how an op's rounds split into
    windows (plan_windows), when a window has all the rounds it needs
    (data_complete), and whether the code can support the scheme's
    buffer sizes (validate_buffer)."""

    def plan_windows(self, op_id: int, round_count: int, code) -> list: ...

    def data_complete(self, window: Window, *, rounds_arrived: int,
                      successor_rounds: int, memory_rounds: int,
                      round_count: int, has_successor: bool,
                      op=None, layout=None) -> bool: ...

    def validate_buffer(self, code) -> None: ...


@runtime_checkable
class ExecutionPlanner(Protocol):
    """Port 7. Single owner of windowing resolution: holds scheme + layout +
    rounds policy and plans a whole workload into a WindowPlan."""

    def plan(self, ops: list): ...


# ----------------------------------------------------------------- resources

@runtime_checkable
class MagicStateFactory(Protocol):
    """Port 19. Async magic-state supply: request() calls back once a
    distilled state is ready for the op."""

    def request(self, op_id: int, callback: Callable[[], None]): ...

    def shutdown(self) -> None: ...


@runtime_checkable
class Metric(Protocol):
    """Port 20. Observes typed views of the run and reports one result;
    never mutates what it observes."""

    name: str

    def observe(self, view) -> None: ...

    def result(self) -> Any: ...


@runtime_checkable
class MemoryModel(Protocol):
    """Port 18. Observes physical payload storage inside the PayloadStore:
    store()/evict() fire on exactly the fragments held. Optional — when
    absent, storage is unbounded."""

    def store(self, key, payload) -> None: ...

    def evict(self, key) -> None: ...
