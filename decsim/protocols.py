"""Protocol seams for swappable simulator components."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable, TYPE_CHECKING

from .message import (Operation, SyndromePayload, DecodeJob, Window, WindowPlan,
                      DecodeResult, Decision)

if TYPE_CHECKING:
    from .engine import Engine


@runtime_checkable
class InputFrontend(Protocol):
    """Converts an input format into the operation graph the chip runs."""

    def build(self) -> list[Operation]: ...

@runtime_checkable
class DeviceModel(Protocol):
    """QPU-side syndrome source.

    `round_payload` returns one round payload. A model may also define
    `round_payloads` to emit multiple fragments for a round.
    """
    def begin_operation(self, op: Operation) -> None: ...
    def round_payload(self, op: Operation, round_index: int) -> SyndromePayload: ...

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
    """Defines decode windows and readiness.

    Optional hooks let a scheme define window dependencies:
    `wire_deps`, `entry_windows`, and `exit_windows`.
    """
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
