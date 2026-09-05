"""The ports: the map of a readout's path through the machine.

One Protocol per handoff between neighbours, one method per handoff, and
the record that crosses it in message.py. A component depends on ports,
never on another component's class, so a new implementation plugs in as
one class that fills the port and one row in the root's table
(decsim/machine.py). Read top to bottom, this file is the pipeline: the
QPU emits a readout, the controller receives it, the store holds the
packed round, the window manager closes a window, the decoder manager
schedules a decode, the decoder returns a result, the frame commits the
correction, the frame releases the controller, the controller instructs
the QPU, and every hop between components rides a link.

The pluggable parts (SyndromeSource, RoundStore, Decoder, Link,
EscalationPolicy, WindowingScheme, IdlePolicy, Workload) have their
abstract class here, sinter's Decoder shape
(sinter/_decoding/_decoding_decoder_class.py, one class with the methods
a row of the table must offer), written as a Protocol because the
implementations fill it without inheriting. Observation (metrics, the
traffic ledger, the trace) reaches a component through callbacks it
fires, never through a port, so every component runs with no observer.
"""

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import decsim.message as message

# ------------------------------------------------ the QPU emits a readout


@runtime_checkable
class ReadoutReceiver(Protocol):
    """The controller, as the QPU sees it: it takes every readout."""

    def accept_qpu_readout(
        self, readout: message.QPUReadout, route: message.SyndromePacketRoute
    ) -> None:
        """Take one readout on its route (window input or feedback memory)."""


# --------------------------------------- the store holds the packed round


@runtime_checkable
class RoundStore(Protocol):
    """The upstream round store (Buffer 0), as syndrome packing sees it.

    Table row: round_store. The writer asks has_room before every
    write (gem5's queue answers isFull, then allocates); a packed round
    is written once and kept until every consumer releases it. A store
    never refuses a write: a round that finds no room waits upstream.
    """

    def has_room(self) -> bool:
        """Whether one more round fits now."""

    def accept_packed_round(
        self,
        packet: message.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> None:
        """Keep one finished round; the writer asked has_room first."""

    def release_round(self, round_key: tuple) -> None:
        """Free the round; its consumers are done with it."""

    def require_stored(self, round_keys) -> None:
        """Refuse a read of a round that has not landed here."""


@runtime_checkable
class StrongRoundStore(Protocol):
    """The room-side store (syndrome buffer 1), as syndrome packing sees it.

    The writer counts the rounds still crossing its link as room taken.
    """

    def has_room(self) -> bool:
        """Whether one more write can land."""

    def write(
        self,
        packet: message.SyndromeRoundPacket,
        *,
        packet_bits: Optional[int],
        attribution: message.TransferAttribution,
    ) -> None:
        """Carry the round over the store's link and keep it on arrival."""


# ------------------------------------------- the window manager closes a window


@runtime_checkable
class WindowInput(Protocol):
    """The window manager, as syndrome packing sees it."""

    def accept_window_input(self, packet: message.SyndromeRoundPacket) -> None:
        """Publish one stored round to window readiness; never refused."""


# ----------------------------------- the decoder manager schedules a decode


@runtime_checkable
class DecodeQueue(Protocol):
    """The decoder manager, as the window manager sees it."""

    def enqueue(
        self,
        job: message.DecodeJob,
        send_input: Optional[Callable[[Callable[[], None]], int]] = None,
    ) -> None:
        """Admit a job once; send_input(on_landed) moves its input later."""

    def withdraw_window(self, window_key: tuple) -> None:
        """Take back a window's not-yet-started decode; it is superseded."""

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode."""


# ------------------------------------------- the decoder returns a result


@runtime_checkable
class Decoder(Protocol):
    """One decoder: correctness and timing from one object.

    Table rows: pymatching, unweighted_pymatching, belief_matching,
    union_find, tesseract, relay_bp, bposd, or a number (a preset
    latency on the MWPM path). sinter's abstract class with defaults
    (decsim/decoders/decoder.py, DecoderBase) fills start, cancel,
    occupancy and pipeline_depth from decode and latency, so a row
    writes those two. The manager asks occupancy and pipeline_depth at
    dispatch, calls start once the input has landed, and cancel when the
    request is withdrawn; a decoder measured on the host clock answers
    occupancy with None and start decides its own time.
    """

    fault_model_requirement: Any

    def decode(self, job: message.DecodeJob) -> message.DecodeResult:
        """The window's correction and its logical observables."""

    def latency(self, job: message.DecodeJob) -> int:
        """The whole job's service time in ticks, known at dispatch."""

    def start(
        self,
        job: message.DecodeJob,
        engine,
        on_result: Callable[[Optional[message.DecodeResult]], None],
    ) -> None:
        """Run the job on the unit; on_result runs once at its output."""

    def cancel(self, job: message.DecodeJob) -> None:
        """Stop a started job; on_result never runs for it."""

    def occupancy(self, job: message.DecodeJob) -> Optional[int]:
        """Ticks the unit's compute is held from the start; None if measured."""

    def pipeline_depth(self, job: message.DecodeJob) -> int:
        """Decodes that may be in flight on one unit; one is no pipeline."""


@runtime_checkable
class DecodedWindowReceiver(Protocol):
    """The window manager, as the decoder manager sees it."""

    def on_decode_done(
        self, job: message.DecodeJob, result: message.DecodeResult
    ) -> None:
        """A weak decode finished; the result reaches the window."""

    def on_strong_decode_done(
        self, completion: message.StrongDecodeCompletion
    ) -> None:
        """A strong decode finished; the completion names its request."""


# ------------------------------------------- the frame commits the correction


@runtime_checkable
class Frame(Protocol):
    """The Pauli frame, as the window manager sees it."""

    def commit_correction(
        self,
        *,
        window_key: tuple,
        logical_observables,
        request_key: message.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Accept a window's correction once, charge the write, call back."""


# ------------------------------------- the frame releases the controller


@runtime_checkable
class ReleaseReceiver(Protocol):
    """The conditional release, as the window manager sees it."""

    def release_waiters(self, operation: message.Operation) -> None:
        """A final result is in: send every decision it releases."""


@runtime_checkable
class InstructionReceiver(Protocol):
    """The controller, as the conditional release sees it."""

    def relay_instruction(
        self,
        decision: message.Decision,
        deliver: Callable[[message.Decision], None],
    ) -> None:
        """Carry one decision to the controller; deliver runs on arrival."""


# ------------------------------------------ the controller instructs the QPU


@runtime_checkable
class Qpu(Protocol):
    """The QPU, as the controller sees it."""

    def issue(self, command: message.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""


@runtime_checkable
class SyndromeSource(Protocol):
    """What the QPU reads out each round for an operation.

    Table rows: stim_device, timing_only, syndrome_bits, recorded_stim.
    Payload bits are raw measurement bits per round; a source with a
    detector formation table also offers form_round, which syndrome
    packing calls once per complete round.
    """

    operation_circuit_scope: str

    def begin_operation(
        self,
        operation: message.Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Prepare the operation's rounds (a Stim source samples its shot)."""

    def round_payloads(
        self, operation: message.Operation, round_index: int
    ) -> list[message.QPUReadout]:
        """The readouts of one round, one per patch or fragment."""

    def finalize_stream_round(
        self, operation: message.Operation, source_round_count: int
    ) -> list[message.QPUReadout]:
        """The stream's final data readout, as its own fragment."""

    def idle_round_payloads(
        self,
        operation: message.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> list[message.QPUReadout]:
        """The readouts of one idle round on a patch of a live stream."""


# -------------------------------------------------- every hop rides a link


@runtime_checkable
class Link(Protocol):
    """The link fabric as every sender sees it.

    Table rows: logical_reference, bandwidth_limited (link_profiles.py),
    and the yaml's cards. A path is wired or free; a send on a wired path
    delivers by callback with every tick of the transfer on the record.
    """

    def is_wired(self, path: message.LinkPath) -> bool:
        """Whether the card prices this path; an unwired path is a free hop."""

    def expected_delay_ticks(
        self,
        path: message.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
    ) -> int:
        """What a send now would pay if nothing else reached its channel."""

    def send(
        self,
        path: message.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: message.TransferAttribution,
        on_delivered: Callable[[message.Transfer], None],
    ) -> None:
        """Send one transfer on the path; on_delivered runs at delivery."""


# ------------------------------------ the pluggable policies off the path


@runtime_checkable
class EscalationServices(Protocol):
    """What the window manager offers an escalation policy."""

    def make_strong_job(
        self, weak_job: message.DecodeJob, label: str
    ) -> message.DecodeJob:
        """The strong job that re-decodes the weak job's window."""

    def defer_strong_escalation(
        self, weak_job: message.DecodeJob
    ) -> message.DecoderRequestKey:
        """Reserve a strong request for later, once its boundaries exist."""

    def check_strong_route(
        self, weak_job: message.DecodeJob, strong_job: message.DecodeJob
    ) -> None:
        """Refuse a strong job the decoder manager cannot route."""

    def prepare_strong_selection(
        self,
        weak_job: message.DecodeJob,
        strong_request_key: message.DecoderRequestKey,
        serial_strong_job: Optional[message.DecodeJob],
        *,
        deferred: bool,
        on_selection_delivered: Callable[[], None],
    ) -> None:
        """Send the escalation and call back when the strong side has it."""


@runtime_checkable
class EscalationPolicy(Protocol):
    """Whether and when a window is decoded again by the strong tier.

    Table rows: weak_baseline, strong_only, switching. on_window_ready
    says which jobs to submit when a window is ready; on_decode_outcome
    says what to do with each outcome (accept, escalate, hold). For a
    weak job, on_decode_outcome runs before the core's commit
    bookkeeping, so its directive decides whether the result waits for a
    strong redo.
    """

    requires_strong_context: bool
    bulk_strong: bool
    double_window: bool
    # The tier that decodes the plan's windows; every tier-dependent site
    # (arrival authority, input store and link, request-key tier, output
    # link) follows from this one declaration.
    primary_tier: message.DecoderTier

    def validate_declared_run(
        self,
        *,
        scheme,
        boundary_policy,
        has_dynamic_streams: bool,
        static_decode_plan_selected: bool,
        has_frontend: bool,
    ) -> None:
        """Refuse a run shape the policy does not support."""

    def validate_operations(
        self, operations: tuple[message.OperationPlanningView, ...]
    ) -> None:
        """Refuse a workload the policy does not support."""

    def validate_code_geometry(
        self, geometry: message.ResolvedCodeGeometry
    ) -> None:
        """Refuse a window geometry the policy does not support."""

    def on_window_ready(
        self,
        window: message.Window,
        weak_job: message.DecodeJob,
        services: EscalationServices,
    ) -> list[message.Submission]:
        """The jobs to submit for a window whose data is complete."""

    def on_decode_outcome(
        self, outcome: message.DecodeOutcome, services: EscalationServices
    ) -> message.OutcomeDirective:
        """Accept, escalate or hold one decode outcome."""


@runtime_checkable
class WindowingScheme(Protocol):
    """How an operation's rounds are cut into windows.

    Table rows: sliding, parallel, sandwich, naive_online. The static
    window graph of an operation, when a window has its data, and the
    buffer floor the scheme needs.
    """

    def plan_operation(
        self,
        operation_id: int,
        round_count: int,
        *,
        commit_round_count: int,
        buffer_round_count: int,
    ):
        """The operation's windows and their internal dependencies."""

    def data_complete(
        self,
        window: message.Window,
        *,
        readiness: message.WindowReadiness,
        operation: message.OperationPlanningView,
    ) -> bool:
        """Whether the window has every round it reads."""

    def validate_buffer(self, geometry: message.ResolvedCodeGeometry) -> None:
        """Refuse a buffer below the scheme's floor."""


@runtime_checkable
class IdlePolicy(Protocol):
    """How idle rounds travel while an operation waits for feedback.

    Table rows: separate_decode_jobs, ignore, extend_stream
    (controller/policies.py). relay carries one idle round through the
    controller it is given; end_idle_period runs when an operation
    claims the patch, so rounds the policy has not charged yet can be
    settled.
    """

    def relay(self, controller, operation, patch, round_index: int) -> None:
        """Carry one idle round of the patch through the controller."""

    def end_idle_period(self, controller, operation, patch) -> None:
        """Settle the uncharged rounds when an operation claims the patch."""


@runtime_checkable
class Workload(Protocol):
    """What the machine runs: the operations, wired in program order.

    Table rows: memory_circuit, circuit_list, surgery_ir, qlx.
    """

    def build(self) -> list[message.Operation]:
        """The operations, each with its patches and its predecessors."""
