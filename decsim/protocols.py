"""Protocol seams for swappable simulator components."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING

from .message import (Operation, SyndromePayload, DecodeJob, Window, WindowPlan,
                      DecodeResult, Decision, SoftOutput)

if TYPE_CHECKING:
    from .engine import Engine


@runtime_checkable
class InputFrontend(Protocol):
    """Converts an input format into the operation graph the chip runs."""

    def build(self) -> list[Operation]: ...

@runtime_checkable
class SyndromeSource(Protocol):
    """Produces syndrome payloads and matching window error models (timing-only sources return none)."""

    def begin_operation(self, op: Operation) -> None: ...
    def round_payloads(self, op: Operation,
                       round_index: int) -> list[SyndromePayload]: ...
    def idle_round_payloads(self, op: Operation, stream_id: Any,
                            global_round: int,
                            patch: Any) -> list[SyndromePayload]: ...
    def register_dynamic_stream(self, stream_op: Operation, round_count: int,
                                *, belief_matching: bool = False) -> Optional[int]: ...
    def validate_stream_length(self, stream_op: Operation,
                               stream_round_count: int) -> None: ...
    def window_models_for_operation(self, op: Operation, windows: list,
                                    round_count: int,
                                    *, belief_matching: bool = False) -> list: ...
    def window_model_for_stream(self, stream_id: Any, window: Window,
                                *, is_last: bool): ...
    def strong_window_model_for_operation(self, op: Operation, window: Window,
                                          round_count: int,
                                          *, belief_matching: bool = False): ...

@runtime_checkable
class CodeModel(Protocol):
    """Code-specific quantities used by planning, timing, and metrics."""

    @property
    def name(self) -> str: ...

    @property
    def distance(self) -> int: ...

    def rounds_per_logical_cycle(self) -> int: ...
    def commit_rounds(self) -> int: ...
    def buffer_rounds(self) -> int: ...
    def spatial_nodes(self, num_patches: int) -> int: ...
    def syndrome_bits_per_round(self, num_patches: int) -> int: ...

@runtime_checkable
class LayoutModel(Protocol):
    """Maps patches and operations to code models."""

    @property
    def name(self) -> str: ...

    @property
    def distance(self) -> int: ...

    def code_for_patch(self, patch_id: Any) -> CodeModel: ...
    def code_for_op(self, op: Operation) -> CodeModel: ...
    def spatial_nodes_for(self, op: Operation) -> int: ...
    def codes(self) -> list: ...

@runtime_checkable
class DecodingScheme(Protocol):
    """Defines decode windows and readiness; optional hooks wire_deps/entry_windows/exit_windows set dependencies."""
    def plan_windows(self, op_id: int, round_count: int,
                     code: CodeModel) -> list[tuple]: ...
    def data_complete(self, window: Window, rounds_arrived: int, successor_rounds: int,
                      memory_rounds: int, round_count: int, has_successor: bool,
                      op: Operation = None, layout: LayoutModel = None) -> bool: ...


@runtime_checkable
class ExecutionPlanner(Protocol):
    """Builds the compile-time decode plan for an operation graph."""

    def plan(self, ops: list[Operation]) -> WindowPlan: ...

@runtime_checkable
class RoundsPolicy(Protocol):
    """Chooses how many QEC rounds an operation runs."""

    def rounds_for(self, op: "Operation", code: "CodeModel") -> int: ...

@runtime_checkable
class Decoder(Protocol):
    """Decodes one job and separately reports simulated latency."""

    def latency(self, job: DecodeJob) -> int: ...
    def decode(self, job: DecodeJob) -> DecodeResult: ...

@runtime_checkable
class SoftOutputMetric(Protocol):
    """Computes a soft output g per window (smaller g = lower confidence); swappable across metrics."""  # ref: paper Sec. II.B

    @property
    def name(self) -> str: ...

    def evaluate(self, syndrome) -> SoftOutput: ...

@runtime_checkable
class Scheduler(Protocol):
    """Orders ready decode jobs."""

    def insert(self, queue: list, job: DecodeJob) -> None: ...
    def pop(self, queue: list) -> DecodeJob: ...


@runtime_checkable
class DeadlinePolicy(Protocol):
    """Assigns a scheduling deadline to each ready window job."""
    def deadline(self, op: Operation, window: Window, now: int,
                 on_reaction_path: bool) -> int: ...


@runtime_checkable
class DecoderRouter(Protocol):
    """Picks the decoder used for each job at dispatch time."""

    def route(self, job: DecodeJob) -> "Decoder": ...

@runtime_checkable
class DecoderService(Protocol):
    """Allows components to submit external decode jobs to the decoder cluster."""
    def submit_decode(self, round_count: int, on_done: Callable[[], None],
                      label: str = ..., deadline: Optional[int] = ...,
                      code: Optional[str] = ...,
                      spatial_nodes: Optional[int] = ...,
                      hint: Optional[str] = ...) -> None: ...


@runtime_checkable
class WorkloadManager(DecoderService, Protocol):
    """Decoder-cluster contract used by the chip, wiring, and orchestrator."""
    def register_op(self, op: Operation) -> None: ...
    def build_windows(self) -> None: ...
    def load_execution_plan(self, plan: WindowPlan) -> None: ...
    def register_dynamic_stream(self, stream_op: Operation, code: CodeModel) -> None: ...
    def has_dynamic_stream(self, stream_id: Any) -> bool: ...
    def close_stream_boundary(self, stream_id: Any, stream_round_count: int) -> None: ...
    def seal_stream(self, stream_id: Any, stream_round_count: int) -> None: ...
    def committed_stream_round_count(self, stream_id: Any) -> int: ...
    def rounds_for(self, op: Operation) -> int: ...
    def on_syndrome_arrival(self, payload: SyndromePayload) -> None: ...
    def on_memory_round(self, op_id: int) -> None: ...


@runtime_checkable
class Controller(Protocol):
    """Moves syndrome payloads and feedback instructions across links."""

    def relay_syndrome(self, payload: SyndromePayload,
                       deliver: Callable[[SyndromePayload], None]) -> None: ...
    def relay_instruction(self, decision: Decision,
                          deliver: Callable[[Decision], None]) -> None: ...

@runtime_checkable
class Orchestrator(Protocol):
    """Prepares execution, integrates decoded results, and dispatches feedback."""

    def connect(self, controller: "Controller", decision_sink: Callable) -> None: ...
    def register_blocked_operation(self, blocked_op_id: int, blocking_op_id: int) -> None: ...
    def prepare_execution(self, *, operations: list[Operation],
                          cluster: WorkloadManager,
                          planner: ExecutionPlanner,
                          decode_operations: Optional[list[Operation]] = None) -> WindowPlan: ...
    def integrate(self, op: Operation, result: DecodeResult) -> None: ...
    def on_result(self, op: Operation, result: DecodeResult) -> list[Decision]: ...
    

@runtime_checkable
class MagicStateFactory(Protocol):
    """Supplies magic states to operations that need them."""

    def request(self, op_id: int, callback: Callable[[], None]) -> None: ...
    def shutdown(self) -> None: ...

@runtime_checkable
class QuantumProcessor(Protocol):
    """Runs the operation graph and emits syndrome rounds."""

    def load(self, ops: list) -> None: ...
    def on_decision(self, decision: Decision) -> None: ...
 
 
@runtime_checkable
class Metric(Protocol):
    """Read-only observer called after every engine event."""

    def observe(self, engine: "Engine") -> None: ...
    def result(self): ...
