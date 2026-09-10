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

from collections.abc import Sequence
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

    def mark_publication_tick(
        self, round_key: tuple, publication_tick: int
    ) -> None:
        """The round became readable now; it was written before that."""

    def capacity_rounds(self) -> Optional[int]:
        """The slots this store is bounded to, or None for unbounded.

        A store bounded some other way than by a slot count answers
        None and refuses nothing: the callers that size a plan, a trace
        lane or a room check ask this instead of reading a settings
        record, so the bound stays the store's own to decide.
        """

    def held_rounds_description(self) -> str:
        """The live holds, in one line, for a refusal a reader must debug."""


@runtime_checkable
class RetainedRounds(Protocol):
    """The same store, as the window side that reads and holds it sees it.

    Two neighbours cross a store on two different handoffs: syndrome
    packing writes rounds through RoundStore, and the window side keeps
    the rounds one decode reads alive through this port. A hold names
    the rounds its holder will read from the moment it is placed, so the
    store may hold a round it has not received yet; the holder is any
    record that answers referenced_operation_ids (decsim/records).
    """

    def register_hold(self, holder, round_keys: tuple) -> None:
        """Keep these rounds for this holder until it releases them."""

    def replace_hold(self, holder, round_keys: tuple) -> None:
        """The holder reads a different span now; the old span may go."""

    def transfer_hold(self, old_holder, new_holder) -> None:
        """The same span passes to a new holder, with no gap between."""

    def release_hold(self, holder) -> None:
        """The holder is done; rounds no other holder wants may go."""

    def has_hold(self, holder) -> bool:
        """Whether this holder's span is live."""

    def hold_round_identities(self, holder) -> tuple:
        """The rounds this holder keeps, in the order it named them."""

    def release_round_if_unheld(self, round_key: tuple) -> bool:
        """Free the round when no holder wants it; True when it went."""

    def retained_fragments(self, round_key: tuple) -> Optional[tuple]:
        """The stored round's fragments, or None before and after storage."""

    def is_round_held(self, round_key: tuple) -> bool:
        """Whether a holder wants this round, stored or still expected."""

    def publication_tick(self, round_key: tuple) -> Optional[int]:
        """When the round became readable, or None while it is not."""

    def open_operation(self, operation_id) -> None:
        """This operation may receive rounds from now on."""

    def has_operation(self, operation_id) -> bool:
        """Whether the store still serves this operation."""

    def close_operation(self, operation_id) -> None:
        """The operation sends no more rounds; a closed one never reopens.

        A round still crossing a priced link when its operation closes
        never enters the store: no hold can name a closed operation, so
        the round has no reader and its writer drops it at the landing.
        """

    def has_live_operation_reference(self, operation_id) -> bool:
        """Whether a hold or a stored round still names this operation."""

    def capacity_rounds(self) -> Optional[int]:
        """The slots this store is bounded to, or None for unbounded."""


@runtime_checkable
class StrongRoundStore(Protocol):
    """The room-side store (syndrome buffer 1), as syndrome packing sees it.

    The store counts the rounds still crossing toward it as room taken:
    the controller reserves that room before a round leaves and the
    store keeps the round when it lands.
    """

    def has_room(self) -> bool:
        """Whether one more write can land."""

    def reserve_write(self) -> None:
        """Take the room one crossing round will need, before it leaves."""

    def receive_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        packet_bits: Optional[int],
    ) -> None:
        """Take one round that landed here and keep it on arrival."""


@runtime_checkable
class RoundStoreInput(Protocol):
    """A round store's incoming port, as the controller's transmitter sees it.

    The transfer that carries a round to the store lands here, and this
    end handles the landing: it stamps the store's record and announces
    the published round to whoever waits on it. The same port takes the
    controller's ask for a timing-only round, which leaves by the
    store's own outgoing port.
    """

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Take one round that landed here: publish it, then announce it."""

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Send one timing-only round to the decoder side the store feeds."""


# ------------------------------------------- the window manager closes a window


@runtime_checkable
class WindowInput(Protocol):
    """The window manager, as the controller side sees it.

    Two kinds of thing cross here. A round arrives: the packed round
    itself, or the note that an idle patch produced a timing-only one. A
    stream's shape changes: the controller learns a dynamic stream's
    boundary, its length, or which segment folds into it, and the window
    plan follows. Both stay one-way; the window manager answers nothing
    back except whether it knows a stream at all.
    """

    def accept_window_input(
        self, packet: round_records.SyndromeRoundPacket
    ) -> None:
        """Publish one stored round to window readiness; never refused."""

    def accept_feedback_memory_round(self, source_operation_id) -> None:
        """Record one idle or memory round and re-check waiting windows."""

    def prepend_idle_rounds(self, operation_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation."""

    def has_dynamic_stream(self, stream_id) -> bool:
        """True for a stream whose windows are planned at runtime."""

    def close_stream_boundary(self, stream_id, stream_round_count: int) -> None:
        """Mark a live stream round as a measurement-closed boundary."""

    def seal_stream(self, stream_id, stream_round_count: int) -> None:
        """Close a dynamic stream once its full length has arrived."""

    def bind_stream_operation(
        self, operation_id: int, stream_id, stream_offset: int
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""


@runtime_checkable
class WindowPlan(Protocol):
    """The window plan, as the escalation side reads and reshapes it.

    A strong window covers rounds several weak windows were going to
    commit, so the strong region reads the plan and the shape row changes
    it: one window absorbed, another's read start moved. Everything else
    about the plan (who is ready, what commits) stays the window side's.
    """

    def window_at(self, key: tuple) -> window_records.Window:
        """The window at (operation id, index)."""

    def absorb_window(
        self, key: tuple, restart_key: Optional[tuple]
    ) -> window_records.Window:
        """A window a strong region covers is never weak-decoded."""

    def reslice_window(
        self, key: tuple, buffer_lo: int, model
    ) -> window_records.Window:
        """Move a window's read start and install the model it reads with."""

    def later_windows(self, operation_id: Any, window_index: int) -> list:
        """The operation's windows past that index, in index order."""

    def check_absorbable(self, window_keys) -> None:
        """Every listed window is still undecoded and uncommitted."""

    def strong_window_model(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        fault_exclusion_ranges: tuple,
        prior_faults: Optional[dict],
    ):
        """The error model of one strong window of that operation.

        prior_faults is what a pinned face's neighbour has committed,
        per fault representation, and None when the row pins no face.
        """

    def owned_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults the window at that key commits, per representation.

        A row that pins a face on a neighbour asks for this, so that the
        neighbour's faults are prior faults of its own strong model
        rather than columns it may spend twice (Bombin et al.
        2303.04846 lines 775-788). None when the run builds no error
        models.
        """


@runtime_checkable
class WindowRetention(Protocol):
    """The rounds a window may still read, as the escalation side sees it.

    Which store a strong window reads is the retention's to say, not the
    escalation's, so a caller here asks for holds and for the input
    rather than for a store.
    """

    def hold_strong_input(self, job: decoding_records.DecodeJob) -> None:
        """The strong job's context becomes its input hold."""

    def strong_window_input(self, builder, window) -> list:
        """The room-side payloads of a strong window, first round stamped."""

    def hold_strong_context(
        self, key: tuple, strong_request_key, context_keys
    ) -> None:
        """The window's potential strong read becomes the request's hold."""

    def context_rounds_in_flight(self, key: tuple, read_keys) -> tuple:
        """The context rounds that reached the upstream store and are late."""

    def guard_restart_reads(
        self,
        key: tuple,
        restart_key: Optional[tuple],
        proposed_restart,
        strong_request_key,
        context_keys,
        restart_read_keys,
    ):
        """Hold the restart window's strong context while a plan lands."""

    def replace_window_reads(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Re-point the window's live holds at its reads."""

    def release_restart_reads(self, key: tuple) -> None:
        """No earlier escalation can re-slice the window: its claim ends."""

    def release_hold_if_live(self, owner, store=None) -> None:
        """Drop a hold that is still live; nothing for one already gone."""

    def release_strong_hold_if_live(self, owner) -> None:
        """Drop a room-side hold when it is still registered."""

    def release_absorbed_strong_hold(
        self, key: tuple, restart_key: Optional[tuple], replacement
    ) -> None:
        """Drop the absorbed window's potential read; the request holds it."""

    def require_rounds_retained(
        self, label: str, payloads: list, first_round: int, last_round: int
    ) -> None:
        """A strong window starts only once every round it reads is held."""

    def read_keys_for_bounds(
        self,
        operation_id: Any,
        start_round: int,
        buffer_hi: int,
        window: Optional[window_records.Window] = None,
    ) -> list:
        """The retained round keys of a possibly cross-operation range."""

    def require_retained(
        self, round_keys: list, purpose: str, store=None
    ) -> None:
        """Refuse a new consumer if an already-arrived round was released."""

    def require_strong_retained(self, round_keys, purpose: str) -> None:
        """The same, on the room-side store."""


@runtime_checkable
class WindowJobBuilder(Protocol):
    """Where a strong window shape gets a job's identity and its gate."""

    gate: Any

    def new_request_key(
        self,
        operation_id: Any,
        window_id: int,
        tier: window_records.DecoderTier,
    ) -> window_records.DecoderRequestKey:
        """The next request identity, run-wide ordinal included."""


@runtime_checkable
class WindowRequests(Protocol):
    """The submission side, as a strong window shape steers it."""

    def request_if_ready(
        self, window: window_records.Window, strong_redecode
    ) -> None:
        """If the window has its data, submit it through the policy."""

    def withdraw(self, window: window_records.Window) -> None:
        """Withdraw one window's early-shipped, unstarted decode."""


@runtime_checkable
class LogicalLedger(Protocol):
    """Who commits which rounds, as a strong window shape rewrites it."""

    def owns_strong_window(self, owner_key: tuple) -> bool:
        """Whether a strong window already claims that owner's extent."""

    def replace_contributions(
        self,
        owner_key: tuple,
        commit_lo: int,
        commit_hi: int,
        replaced_keys: tuple,
    ) -> None:
        """A strong window takes the extent of the windows it replaces."""


@runtime_checkable
class BoundaryCourier(Protocol):
    """The committed boundaries a strong window pins a face on.

    A shape row whose face is pinned reads the correction its neighbour
    committed and has it delivered to its own window, which is Bombin
    et al. 2303.04846's input adaptation (lines 775-788): the input to
    the later decoding task is the syndrome of the errors plus the
    corrections already committed. The message is one seam layer on
    decoder_to_decoder, priced against the receiving window's own model.
    Pinning a face is the whole of what the escalation side asks for, so
    it is the whole port; what the courier tells its own package about a
    committed boundary stays a method of the class.
    """

    def pin_strong_face(
        self,
        source_key: tuple,
        destination: window_records.Window,
        model,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
    ) -> None:
        """Ship a committed boundary to a strong window and fold it in."""


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

        Its selection is on the weak-to-strong link; a result that
        finishes first waits in the unit that produced it until
        accept_selection.
        """

    def accept_selection(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: the request's result may reach the window."""

    def charge_soft_output(
        self, job: decoding_records.DecodeJob, ticks: int
    ) -> None:
        """Charge the confidence's own computation on the job's unit.

        The window side computes no confidence: the signal row reports
        what its computation cost and the join asks the decoder side to
        put those ticks on the unit that produced the evidence, whose
        service then carries them (decision D8).
        """

    def resolve_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The window side decided this weak request; close its attempt.

        The verdict is the window's, not the manager's: the manager
        learns here whether the weak answer stands or a strong re-decode
        follows, and closes the attempt either way.
        """

    def read_result(self, job: decoding_records.DecodeJob) -> None:
        """The window side has this job's result in hand.

        A tier whose result blocks its unit gets the unit back here; a
        tier that gave it back at the decode's end has nothing to give.
        """

    def close_companion_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """This forced solve lost; its window is answered by the other.

        The confidence side knows which of a window's solves answered
        it, so the losing solve is closed from there, not by the
        manager's own schedule.
        """


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

    window_checked is the same shape for a row that audits its own
    answer against a referee, and forced_solve_unavailable fires once
    per window model this row cannot pin to a logical class. All three
    sources are on the port because the machine connects the trace, the
    stage ledger and the referee audit to whatever answers this port,
    without asking what the row is; DecoderBase gives a row that fires
    none of them the silent source.

    A row that routes to other rows or wraps one is asked for those rows
    under the seeding protocol, not under this port: decoder_pool's
    routed_decoders walks run_seed_children (decsim/seeding.py
    RunSeedComposite) from the router down, so a routing or wrapping row
    that does not answer it hides the rows inside it from the trace, the
    stage ledger and the referee audit. SwitchingRouter, CodeRouter,
    StagedDecoder, SampledConfidenceDecoder and TesseractCheckedDecoder
    are the shipped rows that answer it.
    """

    fault_model_requirement: Any
    stage_recorded: Any
    window_checked: Any
    forced_solve_unavailable: Any
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
    """The Pauli frame, as the window manager sees it.

    Table row: logical_register (FRAMES,
    pauli_frame/pauli_frame.py), named by pauli_frame.kind; every row
    takes the engine and the write cost in ticks.
    """

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
    """The QPU, as the controller sees it.

    The controller instructs and it emits: an operation body, the end of
    the program, and the two rounds a patch produces while no body runs.
    Where the next body may start is the QPU's own cadence, so the
    controller asks for the boundary rather than computing it.
    """

    def issue(self, command: program_records.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""

    def next_boundary(self) -> int:
        """The cycle boundary at or after now, where an issue would start."""

    def finish(self) -> None:
        """The program is complete: idle patches stop after this cycle."""

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        patch: Any,
    ) -> None:
        """Produce and deliver one idle round of a live stream."""

    def emit_feedback_memory_round(
        self, operation_id: Any, patch: Any, round_index: int
    ) -> None:
        """Deliver the timing-only round of an idle patch."""


@runtime_checkable
class SyndromeSource(Protocol):
    """What the QPU reads out each round for an operation.

    Table rows: stim_device, timing_only, syndrome_bits, recorded_stim.
    Payload bits are raw measurement bits per round; a source with a
    detector formation table also answers DetectionEventFormer below,
    the port the machine forms a round's detection events through.

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

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """The observable flips the source drew, or None when it draws none."""


@runtime_checkable
class DetectionEventFormer(Protocol):
    """Who turns one round's measurement outcomes into its detection events.

    Rows: the run's syndrome source, whose circuit the recipes are read
    off (decsim/detector_error_model/detector_formation.py), and the
    memoising former the decoder side reads through
    (decsim/detector_error_model/detection_event_formation.py). Where the
    machine calls it is controller.detection_events_formed_at: the
    controller's assembler before the round leaves, or the tier that
    reads the round. A former is called once per round, in round order,
    because a detector compares this round's outcomes against the round
    before it (LILLIPUT 2108.06569 lines 499-510) and the formation table
    keeps only the packets its recipes still read.
    """

    def form_round(
        self, operation_id: Any, round_index: int, raw_bits: Sequence[int]
    ) -> tuple[int, ...]:
        """The round's detection events, in detector order."""


@runtime_checkable
class DetectionEventPlacement(Protocol):
    """Where the machine forms a round's detection events.

    Table rows: controller, decoder
    (decsim/controller/settings.py DETECTION_EVENT_FORMATION, built from
    controller.detection_events_formed_at). The two are published
    placements of the same conversion, Google's workstation (2408.13687
    lines 474-476) against the decoder side (Caune 2410.05202 lines
    1252-1256, LILLIPUT 2108.06569 lines 499-510), and the values are
    the same either way, so a row moves the width the round carries, the
    clock its formation is charged on, and nothing else. Every row is
    built with the run's DetectionEventFormer and the controller's own
    formation ticks.

    The controller's assembler asks form_before_departure for the round
    that leaves it and waits departure_ticks before handing it on; the
    root asks decoder_side_former for the former each decoder tier reads
    its rounds through, which is None for a row that has already formed
    them.
    """

    departure_ticks: int

    def form_before_departure(self, fragments: tuple) -> tuple:
        """The round's fragments as they leave the controller."""

    def decoder_side_former(self) -> Optional[DetectionEventFormer]:
        """The former each tier forms through, or None when none does."""


@runtime_checkable
class WindowModelSource(Protocol):
    """Who builds the decoder-facing error model of one window.

    The window planner holds one of these and asks it per window; the
    run's syndrome source is the shipped answer, since the model comes
    from the same circuit the readouts come from, but the plan takes it
    as its own collaborator (qpu.error_model_provider) so a model built
    anywhere else plugs in. A run whose source builds no models supplies
    none at all: the planner holds None and asks nothing.

    The requirement and the returned model are the detector error
    model's records; this port names them by position only, so the
    window side never imports that package to hold the port.
    """

    def window_models_for_operation(
        self,
        operation: program_records.Operation,
        windows: list,
        round_count: int,
        *,
        fault_model_requirement,
        fault_exclusion_ranges: tuple,
        window_protocol,
    ) -> list:
        """One model per window of a planned operation, in window order."""

    def window_model_for_stream(
        self, stream_id: Any, window: window_records.Window
    ):
        """The model of one window of a dynamic stream, laid at runtime."""

    def register_dynamic_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
        *,
        fault_model_requirement,
    ) -> Optional[int]:
        """Note a dynamic stream; the rounds it can supply, or None."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Refuse a stream longer than this source can supply."""

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement,
        exclude_faults_touching=None,
        prior_faults=None,
    ):
        """One strong window's model, with one non-owned range excluded.

        prior_faults names the faults a pinned face's neighbour has
        already committed; they are no columns of this model at all
        (Bombin et al. 2303.04846 lines 775-788).
        """

    def strong_window_model_for_operation_with_exclusions(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement,
        fault_exclusion_ranges: tuple,
        prior_faults=None,
    ):
        """The same, with several non-owned ranges excluded."""


# -------------------------------------------------- every hop rides a link


@runtime_checkable
class Link(Protocol):
    """The link fabric as every sender sees it.

    Table rows: logical_reference, bandwidth_limited (LINK_FABRICS,
    links/link_profiles.py), named by links.kind; a row supplies the
    numbers the section's per-path cards override and builds the fabric
    the root sends on. Every hop of the reaction path is priced, and a
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


@runtime_checkable
class WindowTransfers(Protocol):
    """The link fabric, as a sender that names a window or a job sees it.

    One step below Link: the sender says what it moves (this window's
    boundary, this job's rounds) and the adapter builds the transfer's
    attribution and calls Link.send. The decoder output ports and a
    store's output port hold it by constructor, so it is a port and not
    a class they import. Every method here sends; an input that rides no
    link never reaches this port, because whether an input moves at all
    is the sending store's own decision.
    """

    def send_for_window(
        self,
        path: transfer_records.LinkPath,
        window: window_records.Window,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a window's name; on_delivered runs at the delivery."""

    def send_for_job(
        self,
        path: transfer_records.LinkPath,
        job: decoding_records.DecodeJob,
        *,
        payload_bits: Optional[int],
        request_key: Optional[window_records.DecoderRequestKey] = None,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send in a job's name; returns the delay the link expects."""

    def send_for_round(
        self,
        path: transfer_records.LinkPath,
        packet: round_records.SyndromeRoundPacket,
        payload_bits: Optional[int],
        on_delivered: Callable[[], None],
    ) -> None:
        """Send in a stored round's name; on_delivered runs at the delivery."""

    def send_boundary(
        self,
        path: transfer_records.LinkPath,
        attribution: transfer_records.TransferAttribution,
        payload_bits: Optional[int],
        on_delivered: Callable[[transfer_records.Transfer], None],
    ) -> None:
        """Send one boundary on its path, in its attribution's name."""


# ------------------------------------ the pluggable policies off the path


@runtime_checkable
class BoundaryPolicy(Protocol):
    """When a committed window ships its boundary to the windows after it.

    Table rows: eager, held (BOUNDARY_POLICIES, windows/settings.py),
    named by windows.boundaries. Eager ships at every commit; held ships
    only once the committing result is final. A provisional boundary that
    ships is never revised, so a run that may revise a weak result needs
    a row that holds. ships_provisional_boundaries is that fact, declared
    by the row so a caller reads it rather than the row's class.
    """

    ships_provisional_boundaries: bool

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Whether to ship the boundary now."""


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
    # Whether the policy reads a confidence to decide keep, so the
    # escalation section carries the confidence keys, the weak decoder
    # must serve the run's signal, and the window side joins the solves
    # a signal needs. A row is built from one EscalationCollaborators
    # record (escalation/policies.py) and this says which of its fields
    # the row reads.
    decides_on_a_confidence: bool

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

    Table rows: fixed, table, online (THRESHOLD_SOURCES,
    escalation/settings.py). A source that audits by escalating (the
    online row labels a kept window by re-decoding it on the strong
    tier) needs one serial strong re-decode per window, so Switching
    refuses it beside run_both_at_once and the forward strong window.
    reads_a_calibration_table says the point's number comes from
    calibrate_threshold.py's csv rather than from the section's card, so
    the settings demand threshold_table. built_per_sweep_point says the
    row builds one instance of itself for a whole sweep point, which the
    front hands to every shot, because the row learns across the point's
    windows; every other row is built by the root from the point's
    threshold in nats, which is its one constructor argument.
    """

    audits_by_escalating: bool
    reads_a_calibration_table: bool
    built_per_sweep_point: bool

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

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The window's confidence and what computing it cost.

        The soft output is None when the evidence is missing. The ticks
        are the row's own computation on the weak tier's clock: zero for
        a subtraction, and for a walk over the decode's growth the time
        it took, measured on the host clock the way a measured decoder
        is or declared as a card number (decision D8).
        """


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

    Three facts about the layout are declared rather than read off the
    row's class, so a scheme written outside decsim answers the same
    questions the shipped rows answer (gem5's port API, arXiv 2007.03152
    lines 489-491).

    has_trailing_tail_context: the last window this scheme lays out
        reads rounds past its own commit, which is what the switching
        recovery re-reads when a strong result revises a window.
    commits_in_one_serial_chain: the windows commit one after another in
        stride order, which the forward strong window absorbs.
    supports_dynamic_streams: an operation whose round count is not known
        at build can be windowed by this scheme.
    """

    has_trailing_tail_context: bool
    commits_in_one_serial_chain: bool
    supports_dynamic_streams: bool

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
class RoundsPolicy(Protocol):
    """How many syndrome rounds an operation runs for; always at least 1.

    Rows: fixed_rounds, per_operation_rounds, distance_rounds
    (qpu/round_policies.py); the workload's section names one and the
    QLX frontend fills a per-operation row from each task's duration. The
    lattice-surgery unit of d rounds per step is Horsman 1111.4022 Sec.
    3.1 and Litinski 1808.02892.
    """

    def rounds_for(
        self, operation: program_records.OperationPlanningView, code
    ) -> int:
        """The operation's round count on this code."""


@runtime_checkable
class MagicStateFactory(Protocol):
    """Where a non-Clifford operation gets its magic state.

    Table rows: infinite, distillation, multi_level
    (MAGIC_STATE_FACTORIES, qpu/settings.py), each built from one
    FactoryCollaborators record. The runtime asks and is called back; a
    factory that produces on demand answers at once.
    """

    engine: Any

    def request(self, operation_id: int, callback: Callable[[], None]):
        """Ask for one state; callback runs once it is ready."""

    def shutdown(self) -> None:
        """Stop producing; the workload is complete."""


@runtime_checkable
class IdlePolicy(Protocol):
    """How idle rounds travel while an operation waits for feedback.

    Table rows: separate_decode_jobs, ignore, extend_stream
    (controller/policies.py, beside the accounting they serve). relay
    carries one idle round through the
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
