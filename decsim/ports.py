"""The ports: the map of a readout's path through the machine.

One Protocol per handoff between neighbours, one method per handoff, and
the record that crosses it in decsim/records. A component depends on ports,
never on another component's class, so a new implementation plugs in as
one class that fills the port and one row in the root's table
(decsim/machine.py). Read top to bottom, this file is the pipeline: the
QPU emits a readout, the controller receives it, the store holds the
packed round, the window manager closes a window, the decoder manager
schedules a decode, the decoder returns a result, the frame commits the
correction, the frame releases the controller, the controller instructs
the QPU, and every hop between components rides a link.

The pluggable parts (SyndromeSource, RoundStore, Decoder, Link,
EscalationPolicy, ThresholdSource, ConfidenceSignal, WindowingScheme,
IdlePolicy, Workload) have their abstract class here, sinter's Decoder shape
(sinter/_decoding/_decoding_decoder_class.py, one class with the methods
a row of the table must offer), written as a Protocol because the
implementations fill it without inheriting. Observation (metrics, the
traffic ledger, the trace) reaches a component through callbacks it
fires, never through a port, so every component runs with no observer.
"""

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

# ------------------------------------------------ the QPU emits a readout


@runtime_checkable
class ReadoutReceiver(Protocol):
    """The controller, as the QPU sees it: it takes every readout."""

    def accept_qpu_readout(
        self,
        readout: round_records.QPUReadout,
        route: round_records.SyndromePacketRoute,
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
        packet: round_records.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> None:
        """Keep one finished round; the writer asked has_room first."""

    def release_round(self, round_key: tuple) -> None:
        """Free the round; its consumers are done with it."""


@runtime_checkable
class StrongRoundStore(Protocol):
    """The room-side store (syndrome buffer 1), as syndrome packing sees it.

    The writer counts the rounds still crossing its link as room taken.
    """

    def has_room(self) -> bool:
        """Whether one more write can land."""

    def write(
        self,
        packet: round_records.SyndromeRoundPacket,
        *,
        packet_bits: Optional[int],
        attribution: transfer_records.TransferAttribution,
    ) -> None:
        """Carry the round over the store's link and keep it on arrival."""


# ------------------------------------------- the window manager closes a window


@runtime_checkable
class WindowInput(Protocol):
    """The window manager, as syndrome packing sees it."""

    def accept_window_input(
        self, packet: round_records.SyndromeRoundPacket
    ) -> None:
        """Publish one stored round to window readiness; never refused."""


# ----------------------------------- the decoder manager schedules a decode


@runtime_checkable
class WindowInputGate(Protocol):
    """The window side's say over a job's input, carried on the job.

    The decoder manager asks may_stage before a boundary-blocked job
    takes an input slot, may_start before a landed job decodes, and
    calls mask_input once at the start so the landed input carries the
    window's boundary (qLDPC's net_error folded into the next window).
    """

    def may_stage(self, job: decoding_records.DecodeJob) -> bool:
        """Whether a blocked job may occupy an input slot yet."""

    def may_start(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the landed job owes no boundary and may decode."""

    def mask_input(self, job: decoding_records.DecodeJob) -> None:
        """Fold the window's boundary into the landed input, once."""


@runtime_checkable
class DecodeQueue(Protocol):
    """The decoder manager, as the window manager sees it."""

    def enqueue(
        self,
        job: decoding_records.DecodeJob,
        send_input: Optional[Callable[[Callable[[], None]], int]] = None,
        on_decoded: Optional[
            Callable[
                [decoding_records.DecodeJob, decoding_records.DecodeResult],
                None,
            ]
        ] = None,
    ) -> None:
        """Admit a job once; send_input(on_landed) moves its input later.

        on_decoded(job, result) is the job's return path, SimPy's callback
        on the event (simpy/core.py): the submitting side carries it with
        the job. The job carries its WindowInputGate; a windowless job has
        none.
        """

    def enqueue_without_input(
        self,
        round_count: int,
        on_done: Callable[[], None],
        label: str = "external",
        code: Optional[str] = None,
        spatial_nodes: Optional[int] = None,
    ) -> None:
        """Queue a self-contained decode of the rounds; on_done at its end.

        A factory's correction decode or an idle decode: no syndrome
        data, so no transport and no unit memory.
        """

    def withdraw_window(self, window_key: tuple) -> None:
        """Take back a window's not-yet-started decode; it is superseded."""

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode."""

    def await_strong_result(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The window asked for this request's strong result.

        Its selection is on the weak-to-strong link; the result is held
        for the window until accept_selection.
        """

    def accept_selection(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: the request's result may reach the window."""


# ------------------------------------------- the decoder returns a result


@runtime_checkable
class Decoder(Protocol):
    """One decoder: correctness and timing from one object.

    Table rows: pymatching, unweighted_pymatching, belief_matching,
    union_find, tesseract, relay_bp, bposd, or a number (a preset
    latency on the MWPM path). sinter's abstract class
    with defaults (decsim/decoders/decoder.py, DecoderBase) fills start,
    cancel, occupancy and pipeline_depth from decode and latency, so a row
    writes those two. The manager asks occupancy and pipeline_depth at
    dispatch, calls start once the input has landed, and cancel when the
    request is withdrawn; a decoder measured on the host clock answers
    occupancy with None and start decides its own time.

    stage_recorded is the port's data-side stage callback
    (the data path's hop table): a trace source the row
    fires once per internal stage with a DecoderStageRecord, so an ASIC
    model's engines or a GPU model's kernels reach the trace and the
    stage ledger under their own names. A row with no internal stages
    exposes the silent source DecoderBase gives it and fires nothing;
    the machine connects the stage listeners to every row without asking
    what the row is.

    decoder_evidence declares, as data on the row, what its decode can
    show about itself beyond the correction: a solve pinned to one
    logical class with that class's minimum weight, the growth a
    cluster-based decode did, or neither (decsim/records/decoding.py
    DecoderEvidence). A confidence signal declares the same set as its
    requirement, and the yaml refuses a pairing the row cannot serve.
    missing_evidence_reasons is how a row that a reader would expect to
    produce some evidence says why it does not; the refusal quotes it
    instead of the signal's general sentence. The unit's insides stay
    closed: the port learns what the decoder can answer, never how.
    """

    fault_model_requirement: Any
    stage_recorded: Any
    decoder_evidence: frozenset
    missing_evidence_reasons: dict

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The window's correction and its logical observables.

        A timing-only decoder leaves both unset: the result's two
        fields are optional (decsim/records/decoding.py DecodeResult),
        and every reader of them handles None.
        """

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The whole job's service time in ticks, known at dispatch."""

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine,
        on_result: Callable[[Optional[decoding_records.DecodeResult]], None],
    ) -> None:
        """Run the job on the unit; on_result runs once at its output."""

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop a started job; on_result never runs for it."""

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """Ticks the unit's compute is held from the start; None if measured."""

    def pipeline_depth(self, job: decoding_records.DecodeJob) -> int:
        """Decodes that may be in flight on one unit; one is no pipeline."""


# ------------------------------------------- the frame commits the correction


@runtime_checkable
class Frame(Protocol):
    """The Pauli frame, as the window manager sees it."""

    def commit_correction(
        self,
        *,
        window_key: tuple,
        logical_observables,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Accept a window's correction once, charge the write, call back."""


# ------------------------------------- the frame releases the controller


@runtime_checkable
class ReleaseReceiver(Protocol):
    """The conditional release, as the window manager sees it."""

    def release_waiters(self, operation: program_records.Operation) -> None:
        """A final result is in: send every decision it releases."""


@runtime_checkable
class InstructionReceiver(Protocol):
    """The controller, as the conditional release sees it."""

    def relay_instruction(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Carry one decision to the controller; deliver runs on arrival."""


# ------------------------------------------ the controller instructs the QPU


@runtime_checkable
class Qpu(Protocol):
    """The QPU, as the controller sees it."""

    def issue(self, command: program_records.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""


@runtime_checkable
class SyndromeSource(Protocol):
    """What the QPU reads out each round for an operation.

    Table rows: stim_device, timing_only, syndrome_bits, recorded_stim.
    Payload bits are raw measurement bits per round; a source with a
    detector formation table also offers form_round, which syndrome
    packing calls once per complete round.

    shot_sampled(operation, detection_events) is the port's shot source:
    a row that draws a whole shot fires it once per fresh shot, and a row
    that draws nothing carries the silent source, so a listener connects
    to every row by name.
    """

    operation_circuit_scope: str
    shot_sampled: Any

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
    ) -> None:
        """Prepare the operation's rounds (a Stim source samples its shot)."""

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """The readouts of one round, one per patch or fragment."""

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """The stream's final data readout, as its own fragment."""

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> list[round_records.QPUReadout]:
        """The readouts of one idle round on a patch of a live stream."""


# -------------------------------------------------- every hop rides a link


@runtime_checkable
class Link(Protocol):
    """The link fabric as every sender sees it.

    Table rows: logical_reference, bandwidth_limited (link_profiles.py),
    and the yaml's cards. Every hop of the reaction path is priced, and a
    send delivers by callback with every tick of the transfer on the
    record.
    """

    def expected_delay_ticks(
        self,
        path: transfer_records.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
    ) -> int:
        """What a send now would pay if nothing else reached its channel."""

    def send(
        self,
        path: transfer_records.LinkPath,
        payload_bits: Optional[int],
        now_ticks: int,
        attribution: transfer_records.TransferAttribution,
        on_delivered: Callable[[transfer_records.Transfer], None],
    ) -> None:
        """Send one transfer on the path; on_delivered runs at delivery."""


# ------------------------------------ the pluggable policies off the path


@runtime_checkable
class EscalationPolicy(Protocol):
    """Whether and when a window is decoded again by the strong tier.

    Table rows: weak_baseline, strong_only, switching. The policy decides
    and is told, the shape of gem5's conditional predictor
    (src/cpu/pred/conditional.hh: lookup answers, update teaches, and
    the unit acts on the answer): it builds no job and sends nothing.
    tiers_for_ready_window says which tiers decode a complete window at
    once, verdict_for_weak_result keeps a weak result or escalates its
    window, learn_from_strong_result hears the strong tier's answer, and
    check_plan refuses a run the policy cannot serve, once, at build.
    The strong re-decode itself is the window side's
    (decsim/escalation/strong_redecode.py).
    """

    # The tier that decodes the plan's windows; every tier-dependent site
    # (arrival authority, input store and link, request-key tier, output
    # link) follows from this one declaration.
    primary_tier: window_records.DecoderTier
    # Whether the policy may escalate a window, so the run keeps the
    # room-side store, one buffer of context on each side of every
    # window, and the strong tier's window side.
    requires_strong_context: bool

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """Refuse, with a sentence, a run shape the policy cannot serve."""

    def tiers_for_ready_window(
        self, window: window_records.Window
    ) -> tuple[window_records.DecoderTier, ...]:
        """The tiers that decode the complete window now, primary first."""

    def verdict_for_weak_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> decoding_records.Verdict:
        """Keep the weak result as final, or escalate its window."""

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """The strong tier answered for the window; a source may learn."""


@runtime_checkable
class ThresholdSource(Protocol):
    """Where the switching policy's threshold comes from.

    Table rows: fixed, online (decsim/escalation/threshold_sources.py);
    the yaml's table source is resolved to fixed per sweep point by the
    front. A source that audits by escalating (the online row labels a
    kept window by re-decoding it on the strong tier) needs one serial
    strong re-decode per window, so Switching refuses it beside
    run_both_at_once and the forward strong window.
    """

    audits_by_escalating: bool

    def decide_keep(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> bool:
        """True keeps the weak result; the result carries its soft output."""

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """The strong tier answered for the window; the source may learn."""


@runtime_checkable
class ConfidenceSignal(Protocol):
    """The soft output one window's decodes report, as the join sees it.

    Table rows: complementary_gap and cluster_gap (decsim/confidence/).
    source names the signal so the switching policy can refuse another
    one's, fault_model_requirement is what a window model must offer,
    decoder_evidence_requirement is what the decode itself must show
    (decsim/records/decoding.py DecoderEvidence) and is held against the
    weak row's own declaration at the yaml boundary, evidence_refusal is
    the cited sentence that refusal prints, and forced_logical_classes
    are the classes the window must be decoded in, one job each, empty
    for a signal that reads one ordinary decode.
    The row computes the soft output from the decoder's own output: the
    solves it is given are that window's DecodeResults.
    """

    source: decoding_records.SoftOutputSource
    fault_model_requirement: Any
    decoder_evidence_requirement: frozenset
    evidence_refusal: str
    forced_logical_classes: tuple

    def soft_output_for(
        self, solves: tuple
    ) -> Optional[decoding_records.SoftOutput]:
        """The window's confidence; None when the evidence is missing."""


@runtime_checkable
class BoundaryPayload(Protocol):
    """How a boundary message is written on decoder_to_decoder.

    Table rows: dense_seam_mask and sparse_seam_list
    (decsim/windows/boundary_payloads.py), named by
    windows.boundary_payload. The seam is the destination's oldest round
    layer and the row turns it into the bits the wire carries.
    """

    def bits(self, seam: window_records.BoundarySeam) -> int:
        """The bits one hand-off takes in this representation."""


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
        window: window_records.Window,
        *,
        readiness: window_records.WindowReadiness,
    ) -> bool:
        """Whether the window has every round it reads."""

    def validate_buffer(
        self, geometry: program_records.ResolvedCodeGeometry
    ) -> None:
        """Refuse a buffer below the scheme's floor."""


@runtime_checkable
class IdlePolicy(Protocol):
    """How idle rounds travel while an operation waits for feedback.

    Table rows: separate_decode_jobs, ignore, extend_stream
    (controller/policies.py). relay carries one idle round through the
    idle accounting it is given (controller/idle_rounds.py);
    end_idle_period runs when an operation claims the patch, so rounds
    the policy has not charged yet can be settled.
    """

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        """Carry one idle round of the patch through the idle accounting."""

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        """Settle the uncharged rounds when an operation claims the patch."""


@runtime_checkable
class Workload(Protocol):
    """What the machine runs: the operations, wired in program order.

    Table rows: memory_circuit, circuit_list, surgery_ir, qlx.
    """

    def build(self) -> list[program_records.Operation]:
        """The operations, each with its patches and its predecessors."""
