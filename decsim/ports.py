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

The pluggable parts (SyndromeSource, SyndromeBuffer, Decoder,
StrongBackend, Link, EscalationPolicy, ThresholdSource, ConfidenceSignal,
WindowingScheme, IdlePolicy) have their abstract class here, sinter's
Decoder shape (sinter/_decoding/_decoding_decoder_class.py, one class
with the methods a row of the table must offer), written as a Protocol
because the implementations fill it without inheriting. Observation
(metrics, the traffic ledger, the trace) reaches a component through
callbacks it fires, never through a port, so every component runs with
no observer.

A component names its neighbours by declaring a Port (below) for each
one, and the root binds them by assignment once every component exists.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


class Port:
    """One neighbour a component talks to, named on the class, bound once.

    A component declares a port as a class attribute and reads it as an
    ordinary attribute; the root binds it by assignment once every
    component exists, which is gem5's script assigning one port to
    another (gem5 configs/learning_gem5/part1/simple.py:68).
    The port carries the Protocol its peer answers, so a class points at
    this file rather than the other way about.

    Two refusals, both gem5's. A second bind names the port, the peer it
    holds and the peer offered, as PortRef.connect does
    (gem5 src/python/m5/params/port_params.py:109-114). A
    required port read before it is bound raises, as gem5's default peer
    throws UnboundPortException
    (gem5 src/mem/port.cc:62-65); an optional port reads as
    None instead, which is the neighbour a run does not have.

    Whether the peer answers the Protocol is not asked here. Every bind
    site is decsim's own build code, so no yaml and no call on the
    experiments layer can offer a stranger, and one that a change offered
    would raise at its first call naming the method it lacks. The
    question belongs where the wiring becomes input, which is the root
    reading a table of wires, and it is asked of the protocol this port
    carries.

    The name arrives at class creation rather than at construction,
    because a descriptor learns what it was called only once the class
    body has run; gem5 fills it the same way, from its metaclass
    (gem5 src/python/m5/SimObject.py:353-357).
    """

    def __init__(self, protocol, optional: bool = False) -> None:
        self.protocol = protocol
        self.optional = optional
        self.name = ""

    def __set_name__(self, owner, name: str) -> None:
        """Take the name the class body gave this port."""
        del owner
        self.name = name

    def __get__(self, instance, owner=None):
        """The bound peer; None for an unbound optional port."""
        if instance is None:
            return self
        peer = instance.__dict__.get(self.name)
        if peer is not None:
            return peer
        if self.optional:
            return None
        full_name = self._full_name(instance)
        raise RuntimeError(f"{full_name} was read before it was bound")

    def __set__(self, instance, peer) -> None:
        """Bind one peer, and only one."""
        bound = instance.__dict__.get(self.name)
        if bound is not None:
            full_name = self._full_name(instance)
            held = type(bound)
            offered = type(peer)
            raise ValueError(
                f"{full_name} is already bound to {held.__name__}, "
                f"cannot bind {offered.__name__}"
            )
        instance.__dict__[self.name] = peer

    def _full_name(self, instance) -> str:
        """The port as a refusal names it: the class, then the port."""
        owner = type(instance)
        return f"{owner.__name__}.{self.name}"


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


@runtime_checkable
class IdleRoundReceiver(Protocol):
    """The idle accounting, as the QPU sees it: it takes every idle round.

    A patch nobody is operating on still emits a round every cycle, and
    the cycle hands it here rather than to the readout end, because what
    happens to it is a decision the policy makes and not a readout on
    its route.
    """

    def emit_idle_round(self, operation_id, patch, round_index: int) -> None:
        """Take one idle cycle of a patch nobody is operating on."""


# --------------------------------------- the store holds the packed round


@runtime_checkable
class SyndromeBuffer(Protocol):
    """The weak syndrome buffer, as its own round receiver sees it.

    Table row: syndrome_buffer. A round occupies a slot when its bits are
    in the store, which is at the landing of the hop that carried them, and
    it is readable at that same instant: the store and the publication
    are one call at one tick. The end that takes the landing asks
    has_room first with the round's key and bits, counting the rounds it
    has reserved for the writes in flight (gem5's packet store answers
    `avail() = _maxsize - _size - _reserved` against the packet's own
    length, src/dev/net/pktfifo.hh); a packed round is written once and
    kept until every consumer releases it. A store never refuses a
    write: a round that finds no room waits upstream. occupied_bits is
    the bits the stored rounds hold now, which the strong receiving end
    names when an escalated region finds no room.
    """

    occupied_bits: int

    def has_room(
        self,
        round_key: tuple,
        bits: Optional[int],
        reserved_bits_by_round: Mapping[tuple, int],
    ) -> bool:
        """Whether this round fits beside the stored and the reserved rounds.

        The store is asked with the round itself, as gem5's requester
        asks its responder with the packet (RequestPort::tryTiming(
        PacketPtr pkt), src/mem/port.hh:268 and 617-623), so a store of
        several memories answers for the memory the round goes to, which
        gem5's interleaved ranges pick from the address
        (src/base/addr_range.hh 70-78). reserved_bits_by_round is the
        bits each round still crossing toward the store will take, by
        its key. A bounded store raises on a round that states no size:
        a bound is measured against a size.
        """

    def accept_packed_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> None:
        """Keep one landed round, readable at that tick; None publishes none."""

    def book_write(self, round_key: tuple, bits: Optional[int]) -> int:
        """Take a write of this round's stored bits; the tick it completes.

        The store owns its access timing, so the receiving end asks it
        rather than pricing the write itself. The tick is fixed when the
        write is booked, as gem5's SimpleMemory fixes a response tick at
        acceptance (src/mem/simple_mem.cc:174), and a booking is never
        withdrawn. Nothing is scheduled: the caller schedules its own
        continuation at the tick.
        """

    def book_read(self, round_keys: tuple) -> int:
        """Take a read of these stored rounds; the tick their bits are out.

        Asked by whoever takes the bits out, at the tick they leave: the
        store's outgoing port at dispatch. The store sizes the read from
        the widths it keeps. The tick is fixed and never withdrawn, as
        for book_write.
        """

    def release_round(self, round_key: tuple) -> None:
        """Free the round; its consumers are done with it."""

    def capacity_bits(self) -> Optional[int]:
        """The bits this store is bounded to, or None for unbounded.

        A store bounded some other way than by a bit capacity answers
        None and refuses nothing: the callers that size a trace lane or
        a room check ask this instead of reading a settings record, so
        the bound stays the store's own to decide.
        """

    def held_rounds_description(self) -> str:
        """The stored rounds, in one line, for the I/O trace."""

    def check_settled(self) -> None:
        """At the end of a run no round is stored and no hold is live."""


@runtime_checkable
class RetainedRounds(Protocol):
    """The same store, as the window side that reads and holds it sees it.

    Two neighbours cross a store on two different handoffs: syndrome
    packing writes rounds through SyndromeBuffer, and the window side keeps
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
        the round has no reader and the receiver drops it at the landing.
        """

    def has_live_operation_reference(self, operation_id) -> bool:
        """Whether a live hold still names this operation.

        A stored round that no hold names has no reader, so it does not
        keep its operation's result waiting.
        """


@runtime_checkable
class SyndromeRoundSender(Protocol):
    """The syndrome round sender, as the assembler sees it.

    The one end a packed round leaves the assembler by. The sender
    reserves room in every syndrome buffer the round must reach and
    sends it on; a round that finds no room goes to the waiting line,
    which holds it for a retry or drops it as the controller's overflow
    setting says.

    strong_crossing_count is the strong-primary rounds sent to the
    strong syndrome buffer and not yet landed there, which the packing
    stage's bound on rounds in flight counts (controller/round_assembly.py
    RoundsInFlight), since the windows hear of such a round at its
    landing.
    """

    strong_crossing_count: int

    def admit(self, packed: round_records.PackedRound) -> bool:
        """Write the round where it belongs; False when it found no room."""


@runtime_checkable
class StrongSyndromeRoundReceiver(Protocol):
    """The strong syndrome buffer's receiving end, as its two senders see it.

    The store counts the bits still crossing toward it as room taken.
    A strong-primary run's controller reserves that room before a round
    leaves and the store keeps the round when it lands; a switching
    run's strong redecode reserves the room of an escalated region before
    it leaves the chip and the store keeps every round of it at the
    landing.
    """

    def has_room(self, packed: round_records.PackedRound) -> bool:
        """Whether this round's write can land."""

    def reserve_write(self, packed: round_records.PackedRound) -> None:
        """Take the room this crossing round will need, before it leaves."""

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Store a landed round and deliver it on its canonical route."""

    def reserve_region(self, region: round_records.EscalatedRegion) -> None:
        """Take the room an escalated region's rounds will need, or refuse."""

    def receive_region(self, region: round_records.EscalatedRegion) -> None:
        """Take an escalated region that landed here: every round its slot."""


@runtime_checkable
class WeakSyndromeRoundReceiver(Protocol):
    """The weak syndrome round receiver, as the controller sees it.

    This end owns the store's room and its landing. It answers has_room
    against the bits stored and the bits reserved for the writes still
    in flight, the sender reserves that room before the round leaves,
    and the transfer that carries the round lands here: this end stores
    it, which is the tick it becomes readable, and announces it to
    whoever waits on it. The same port takes the controller's ask for a
    timing-only round, which takes its slot here and leaves by the
    store's own outgoing port.
    """

    def has_room(self, packed: round_records.PackedRound) -> bool:
        """Whether this round fits beside the stored and the crossing ones."""

    def reserve_write(self, packed: round_records.PackedRound) -> None:
        """Take the room this crossing round will need, before it leaves."""

    def receive_round(
        self,
        packed: round_records.PackedRound,
        on_published: Callable[[], None],
    ) -> None:
        """Take one landed round: store it, announce it, then on_published.

        on_published runs once the windows have heard of the round, which
        is where the sender's count of rounds in flight lets it go.
        """

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Write one landed timing-only round, then send it to the decoder."""


@runtime_checkable
class SyndromeBufferOutput(Protocol):
    """A syndrome buffer's outgoing port, as whoever asks for a round sees it.

    A round leaves by the end that holds it, so this end executes every
    send and frees the slot the round held. Two kinds leave: a decode
    job's input, asked for by the window side or by the strong
    re-decode, and a timing-only round the controller packed. Each send
    answers the ticks the link expects, so the asker charges nothing of
    its own.
    """

    def send_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """Read one job's rounds out of the store and move them to its unit.

        The delay expected: the read, then the link. A store whose tier
        reads in place lands the input at the read's end instead.
        """

    def land_held_input(
        self,
        job: decoding_records.DecodeJob,
        on_landed: Callable[[], None],
    ) -> int:
        """Land a resubmitted job whose rounds never left: no delay."""

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Send one timing-only round and free its slot at the delivery."""

    def input_send_for(
        self, job: decoding_records.DecodeJob, is_input_held: bool
    ) -> Callable[[Callable[[], None]], int]:
        """The send the decoder manager calls at dispatch, bound to a job."""

    def name_this_store(self, job: decoding_records.DecodeJob) -> None:
        """Stamp the job with the name of the store its rounds sit in."""


@runtime_checkable
class MemoryRoundArrivals(Protocol):
    """The decoders' end for a timing-only round, as the controller sees it.

    A feedback-memory round carries no syndrome to decode; it lands at
    the decoder side so that the stream stage it occupies is accounted
    for, and that end tells whoever waits on it.
    """

    def receive_memory_round(self, source_operation_id) -> None:
        """Take one timing-only round that landed at the decoder side."""


@runtime_checkable
class HeldRounds(Protocol):
    """The waiting line in front of a store, as the store sees it.

    A store that is full holds nothing back itself: the round waits at
    the sender, and the store tells the line when a slot frees so the
    head can try again. Ruby's MessageBuffer counts that wait as the
    buffer's own statistic
    (gem5 src/mem/ruby/network/MessageBuffer.cc:76-82), and
    ns-3's queue disc stamps the packet at the enqueue
    (ns-3 src/traffic-control/model/queue-disc.cc:851).
    """

    def retry(self) -> None:
        """A slot freed: admit from the head, stop at the first refused."""


# ------------------------------------------- the window manager closes a window


@runtime_checkable
class WindowInput(Protocol):
    """The window manager, as the controller side sees it.

    Two kinds of thing cross here. A round arrives: the packed round
    itself, or the note that an idle patch produced a timing-only one. A
    stream's shape changes: the controller learns a dynamic stream's
    boundary, its length, or which segment folds into it, and the window
    plan follows. Both stay one-way; the window manager answers nothing
    back except whether it knows a stream at all, and which store the
    tier that decodes the plan's windows reads from, which decides where
    the sender publishes a round.
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

    def accept_room_round(self, operation_id, round_index: int) -> None:
        """Record a round that landed in the strong syndrome buffer instead."""

    def accept_boundary(self, window_key: tuple, is_unblocked: bool) -> None:
        """A boundary landed in the window; True when it owed no other."""

    def reads_windows_from(self, store) -> bool:
        """Whether the primary tier's window reads come from this store."""


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

    def crossing_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults it commits that reach behind its commit region.

        A region pinned on the escalated window's own weak commit asks
        for this: that commit stands for the faults crossing the seam
        behind it and for no others, since the rest of the window's
        rounds are exactly what the region decodes again. None when the
        run builds no error models.
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
        """The rounds kept in case the window escalates pass to its request.

        The window's potential strong read becomes the request's hold.
        """

    def context_rounds_in_flight(self, key: tuple, read_keys) -> tuple:
        """The rounds the strong syndrome buffer lacks that the weak one has."""

    def escalated_rounds(self, round_keys) -> tuple:
        """The weak syndrome buffer's packets of these rounds, to carry up."""

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
        """Drop the rounds an absorbed window kept; the strong request has them.

        A strong window covered this window, so its potential read goes.
        """

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
        """The same, on the strong syndrome buffer."""


@runtime_checkable
class WindowRounds(Protocol):
    """The arrivals per operation, as the escalation side reads them.

    A strong region is laid over an operation whose rounds are still
    arriving, so the row asks how long the operation is, how many rounds
    one of its windows reads and how far the strong syndrome buffer has been
    filled. Which windows are ready and what commits stay the window
    side's.
    """

    def operation(self, operation_id) -> program_records.Operation:
        """The operation of that id."""

    def round_count_for_window(
        self, operation_id, window: window_records.Window
    ) -> int:
        """The rounds the window reads of its operation."""

    def strong_rounds_arrived(self, operation_id) -> int:
        """The rounds of the operation stored in the strong syndrome buffer."""


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


@runtime_checkable
class StrongRedecode(Protocol):
    """The strong tier's window side, as the committer and verdict see it.

    Both ends of one escalation cross a package boundary: the window
    that decided sits in windows, the re-decode that carries it out sits
    in escalation. A weak verdict that escalates hands over the job, a
    weak window that commits releases the strong window waiting on it,
    and a weak result that is kept halts the strong request it made
    unnecessary, held here or on the strong side's manager (Toshio et
    al. 2510.25222 lines 606-614).
    """

    def escalate(self, weak_job: decoding_records.DecodeJob) -> None:
        """Ask the strong tier to re-decode the weak job's window."""

    def submit_if_commit_releases(self, window_key: tuple) -> None:
        """A weak window committed: a strong window waiting on it leaves."""

    def cancel_strong_request(self, window_key: tuple) -> None:
        """A kept weak result: its strong request ends, held or submitted."""

    def submit_if_stored_data_releases(self, operation_id) -> None:
        """A round was stored: a window waiting for its tail leaves."""


@runtime_checkable
class WindowVerdict(Protocol):
    """The window side, as a finished decode returns its result to it.

    The return path of one window: a solve of the primary tier, joined
    with the window's others when the run's confidence reads several,
    and the strong re-decode of an escalated window. Either way the side
    that owns the window publishes the correction and finalizes it, so
    the decision and its outcome stay in one place.
    """

    def accept_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """The window's answer: apply the threshold, publish or escalate."""

    def accept_strong_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """A strong decode finished: publish it, then finalize the window."""


# ----------------------------------- the decoder manager schedules a decode


@runtime_checkable
class DecoderInputFold(Protocol):
    """The decoder side's input, as the window's gate hands it a mask.

    The gate says what the landed input must read once the window's
    boundary is folded in; which row the tier declares decides which of
    these two the gate calls, and the decoder side performs the write
    into the storage it owns.
    """

    def fold_into_a_copy(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """Give the job a masked duplicate; the unit's rounds stay raw."""

    def fold_in_place(
        self, job: decoding_records.DecodeJob, masked_input
    ) -> None:
        """Write the masked input into the unit's own memory."""


@runtime_checkable
class WindowInputGate(Protocol):
    """The window side's say over a job's input, carried on the job.

    The decoder manager asks may_stage before a boundary-blocked job
    takes an input slot, may_start before a landed job decodes, and
    calls mask_input once at the start so the landed input carries the
    window's boundary (qLDPC's net_error folded into the next window).
    """

    def may_stage(self, job: decoding_records.DecodeJob) -> bool:
        """Whether a job that cannot decode yet may take a unit's input slot."""

    def may_start(self, job: decoding_records.DecodeJob) -> bool:
        """Whether the landed job owes no boundary and may decode."""

    def mask_input(self, job: decoding_records.DecodeJob) -> None:
        """Fold the window's boundary into the landed input, once."""


@runtime_checkable
class DecoderOutput(Protocol):
    """The decoder side's outgoing sends, as the window side asks them.

    A result leaves the decoder that produced it, so the window side
    asks for the send and never executes it: the correction rides its
    tier's output link to the frame, and an escalated window's selection
    rides the link to the strong decoder.
    """

    def publish(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Send the result on its tier's output link; commit it at delivery."""

    def send_selection(
        self,
        weak_job: decoding_records.DecodeJob,
        strong_request_key: window_records.DecoderRequestKey,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send one window's escalation; returns the delay the link expects."""

    def send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Read a strong window's rounds out of the weak store, send them up.

        The delay expected: the store's read, then the link.
        """


@runtime_checkable
class WindowGapJoin(Protocol):
    """The confidence join, as the requester that admits a solve sees it.

    A run whose confidence reads several forced-class solves joins them
    here rather than in the verdict: every solve of the window returns
    to accept_result, and the join calls the verdict once the window's
    solves are all in. A run whose weak decoder reports its own soft
    output from one decode has no join at all.
    """

    signal: "ConfidenceSignal"

    def accept_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """One solve finished: hold it, or join the window's solves."""

    def unresolved_windows(self) -> tuple:
        """The windows whose solves are still waiting for each other."""


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

    def cancel_strong(self, window_key: tuple) -> None:
        """A kept weak result: the window's strong request ends where it is.

        Queued, crossing the link, running, or done and waiting in the
        unit that produced it; nothing when none is live or done.
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


@runtime_checkable
class DecoderRouter(Protocol):
    """The routing table over the tiers' units, as a caller outside sees it.

    A job goes to one unit and the caller never learns which class that
    is, which is the port API's promise (arXiv 2007.03152 lines 489-491).
    The window side asks the second question rather than the first: what
    a window model must offer for whichever unit would take that code,
    which the planner needs before any job exists.
    """

    def route(self, job: decoding_records.DecodeJob):
        """The decoder this job goes to."""

    def fault_model_requirement_for(self, code: Optional[str]):
        """What a window model must offer for the unit that takes this code."""


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


@runtime_checkable
class StrongBackend(Protocol):
    """The device a strong decode runs on, as the strong decoder sees it.

    Rows: measured_table, a device's measured time law with decsim's own
    answer. A live device (a real GPU process decsim hands the region to
    and waits for) answers the same four methods.

    A strong decode crosses the link in, lands, is noticed, waits, is
    decoded and is written back before the link out. The link cards
    price the two links from echo round trips, which already hold the
    landing, the notice and the write back with no work done: Backline's
    responders poll their memory and "return the received value without
    decoding" (2609.09270 lines 2355-2360), and NVQLink's persistent
    kernel "waits for packet arrival, and loops them back" (2510.25213
    lines 528-533). So a backend reports only the time beyond the echo
    on the same path, the launch included, and never the wait: the wait
    emerges from its capacity in front of the arrivals, a queue decsim
    keeps. IonQ gives each decoding core a fixed set of blocks served in
    turn (2608.25027 lines 507-511), and a device graph "cannot be
    launched twice from the device at the same time" (CUDA C++
    Programming Guide, device graph launch), so one CUDA-Q dispatcher's
    decode ends before its next begins (cudaqx
    docs/sphinx/examples_rst/qec/realtime_relay_bp.rst:49-50).

    submit takes the region with the count of decodes already running on
    the device, because the device's time depends on it (IonQ lines
    550-554 time every decode under full co-running load). The ticket
    lets a device answer after its own call returns; service_ticks and
    result are asked after submit, once each.
    """

    def capacity(self) -> int:
        """Decodes the device runs at once; decsim queues the rest."""

    # The ticket is an opaque identity only the backend that issued it reads.
    def submit(self, request: decoding_records.DecodeJob, running: int) -> Any:
        """Start one region's decode with running others on the device."""

    def service_ticks(self, ticket: Any) -> int:
        """The decode's time beyond the echo on the same path, in ticks."""

    def result(self, ticket: Any) -> decoding_records.DecodeResult:
        """The correction and observables; decode_status marks unconverged."""


# ------------------------------------------- the frame commits the correction


@runtime_checkable
class Frame(Protocol):
    """The Pauli frame, as the decoder output sees it.

    Table row: logical_register (FRAMES,
    pauli_frame/pauli_frame.py), named by pauli_frame.kind; every row
    takes the engine, the clock and the write cost in cycles of it.
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
class DecisionDispatch(Protocol):
    """The frame's dispatch, as the conditional release sees it.

    The decision leaves by the frame's end of frame_to_controller, so
    the send is executed there and the controller is reached at the
    landing.
    """

    def dispatch_decision(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Send one decision to the controller; deliver runs at the QPU."""


@runtime_checkable
class InstructionReceiver(Protocol):
    """The controller, as the frame's dispatch sees it."""

    def relay_instruction(
        self,
        decision: program_records.Decision,
        deliver: Callable[[program_records.Decision], None],
    ) -> None:
        """Take one decision at the landing; deliver runs at the QPU."""


@runtime_checkable
class OperationRuntime(Protocol):
    """What drives an operation's life, as the three that reach it see it.

    An operation is started, its body ends, and a decision releases it,
    and the three events arrive from three different places: the QPU
    when the cycle clock reaches the body's last round, the frame's
    release path when a decision lands, and the protected streams when a
    cadence change frees an operation that was waiting. gem5 keeps the
    same split between the workload's graph and the object that runs it
    (gem5 configs/deprecated/example/se.py builds the
    process list, the system runs it).
    """

    def body_done(self, operation: program_records.Operation) -> None:
        """A body finished: record it, free resources, release successors."""

    def on_decision(self, decision: program_records.Decision) -> None:
        """A decision reached the controller: a release starts its operation."""

    def retry_ready_operations(self) -> None:
        """Retry every state-ready operation after a cadence change."""


@runtime_checkable
class OperationIssuer(Protocol):
    """The controller, as the runtime that drives the operations asks it.

    The runtime owns when an operation may go; what going means is the
    controller's: the streams it begins, the idle rounds it claims, the
    command it prepares, and the boundaries it closes afterwards. The
    start callback rides with the issue, so the return path is the job's
    and neither side holds the other.
    """

    def round_ticks_for(self, operation: program_records.Operation) -> int:
        """The resolved QEC cycle length of one operation, in ticks."""

    def can_start(self, operation: program_records.Operation) -> bool:
        """False while a protected stream holds the operation."""

    def issue_operation(
        self,
        operation: program_records.Operation,
        on_started: Callable[[int], None],
    ) -> None:
        """Prepare one QPU command; on_started hears its start boundary."""

    def before_successor_release(
        self, operation: program_records.Operation
    ) -> None:
        """A body finished: its protected regions close on the boundary."""

    def after_successor_release(
        self,
        operation: program_records.Operation,
        waits_for_blocked: bool,
        is_workload_complete: bool,
    ) -> None:
        """Successors released: close boundaries, seal streams, stop the QPU."""


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

    def are_patches_idle(self, operation_id: Any, patches: tuple) -> bool:
        """Every patch is idle after this same operation, as the QPU owns it.

        The controller queries this before advancing a joint stream from
        per-patch idle callbacks; a busy group member must not execute.
        """

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
    ) -> None:
        """Deliver one protection round, including readout when it is final.

        is_final requests this round's terminal readout from a live source.
        It replaces the bulk fragment rather than adding a second round.
        """

    def emit_feedback_memory_round(
        self, operation_id: Any, patch: Any, round_index: int
    ) -> None:
        """Deliver the timing-only round of an idle patch."""


@runtime_checkable
class SyndromeSource(Protocol):
    """What the QPU reads out each round for an operation.

    Table rows: stim_device, timing_only, syndrome_bits, recorded_stim,
    streaming_stim.
    Payload bits are raw measurement bits per round; a source with a
    detector formation table also answers DetectionEventFormer below,
    the port the machine forms a round's detection events through.

    shot_sampled(operation, detection_events) is the port's shot source:
    a physical source fires it once when its complete shot is available.
    A live source waits for final readout; a source that draws nothing
    carries the silent source, so a listener connects to every row by name.

    takes_code_card says whether the row shapes its payloads by the
    run's code card: such a row is built with the card, so its rounds
    state the code's syndrome width, and a row that reads its widths
    off a circuit is built without it.
    """

    # none means this consumer does not need Operation.circuit; it does not
    # require removal when an independent model provider needs the circuit.
    operation_circuit_scope: str
    takes_code_card: bool
    shot_sampled: Any

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> Optional[int]:
        """Bind physical provenance without sampling or building decode models.

        The root declares each owner after seed binding and before execution.
        Return the physical round limit, or None for an open-ended source.
        This declaration is independent of the selected model provider.
        """

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Require a sealed length consistent with the physical history.

        A finite source requires its declared length. A live source requires
        actual final readout at this length, before decoder models finalize.
        """

    def begin_operation(
        self,
        operation: program_records.Operation,
        segment_round_count: int,
        source_round_count: int,
        *,
        round_period_ticks: int,
    ) -> None:
        """Prepare the operation's rounds at the resolved QPU cadence.

        round_period_ticks is the actual physical round period in integer
        simulator ticks. A duration-dependent source checks it against its
        declared circuit period before executing any instructions.
        """

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """The round's acquisitions in measurement order.

        Each acquisition names its contributing patches. The QPU assigns
        fragment_index from list order before transport, unless the operation
        declares the first slot of a contiguous group in a partitioned round.
        The declared fragment count covers the complete round. Sources
        preserve circuit measurement order here; numbered fragments may
        arrive out of order downstream without changing that record order.
        """

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Final data readout, ordered by the round_payloads contract."""

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """The protection round the controller requests, possibly its last.

        round_period_ticks is the resolved QPU cadence in simulator ticks.
        A duration-dependent source checks it before physical execution.
        For a live source, is_final selects a terminal fragment containing
        this round's checks and final data readout in place of a bulk round.
        A finite source retains its already-declared measurement schedule.
        Return acquisitions in the measurement order of round_payloads.
        """

    def logical_observable_truth(
        self, operation_id: Any
    ) -> Optional[tuple[int, ...]]:
        """The observable flips the source drew, or None when it draws none."""

    def readout_departure_tick(
        self, readout: round_records.QPUReadout, readout_tick: int
    ) -> int:
        """The tick this readout leaves the chip, at or after readout_tick.

        readout_tick is the cycle boundary the clock read the round out
        at, and a source whose readout arrives with a fixed delay or a
        jitter names a later tick, as gem5's queued port lets its owner
        name the absolute tick a response or a request is sent
        (gem5 src/mem/qport.hh:94 schedTimingResp, 150
        schedTimingReq). The clock asks for every readout it hands the
        controller, and sends it at that tick behind every earlier
        readout of its patches.
        """

    def window_model_source(self) -> "WindowModelSource":
        """Where the run's window error models come from, by default.

        A source with a circuit answers itself, since its circuit is the
        model; a source without one answers a component that builds no
        model. The plan wires this unless Python names another provider
        (qpu.error_model_provider), as sinter derives a task's model from
        its circuit only when none is given (sinter/_data/_task.py lines
        71 and 87-89): the sampler is never asked to be the model.
        """


@runtime_checkable
class DetectionEventFormer(Protocol):
    """Who turns one round's measurement outcomes into its detection events.

    Rows: the run's syndrome source, whose circuit the recipes are read
    off (decsim/detector_error_model/detector_formation.py), and the
    memoising former the decoder side reads through
    (decsim/detector_error_model/detection_event_formation.py). Where the
    machine calls it is controller.detection_events_formed_at: the
    controller's assembler before the round leaves, the weak syndrome
    buffer's receiving end before the round is stored, or the tier that
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

    Table rows: controller, weak_syndrome_buffer, decoder
    (decsim/controller/settings.py DETECTION_EVENT_FORMATION, built from
    controller.detection_events_formed_at). They are placements of the
    same conversion, Google's workstation (2408.13687 lines 474-476),
    the decoder's own chip ahead of its store (Maurer 2510.21600 lines
    235-237 for the chip, the order ours), and the decoder side (Caune
    2410.05202 lines 1252-1256, LILLIPUT 2108.06569 lines 499-510), and
    the values are the same in every row, so a row moves the width the
    round carries, the clock its formation is charged on, and nothing
    else. Every row is built with the run's DetectionEventFormer and the
    controller's own formation cycles.

    The controller's assembler asks form_before_departure for the round
    that leaves it and waits detection_event_formation_cycles of the
    controller's clock before handing it on; the weak syndrome buffer's
    receiving end asks form_before_storage for the round it stores; the
    root asks decoder_side_former for the former each decoder tier reads
    its rounds through, which is None for a row that has already formed
    them and for a source that forms nothing. The root reads two facts
    about a row. forms_at_the_weak_syndrome_buffer: whether the run
    depends on its rounds landing in that buffer. forms_at_the_decoder:
    whether each tier has its own event-detection stage in front of its
    core, which is priced whether or not the source has outcomes to
    convert.
    """

    detection_event_formation_cycles: int
    forms_at_the_weak_syndrome_buffer: bool
    forms_at_the_decoder: bool

    def form_before_departure(self, fragments: tuple) -> tuple:
        """The round's fragments as they leave the controller."""

    def form_before_storage(self, fragments: tuple) -> tuple:
        """The round's fragments as the weak syndrome buffer stores them."""

    def decoder_side_former(self) -> Optional[DetectionEventFormer]:
        """The former each tier forms through, or None when none does."""


@runtime_checkable
class WindowModelSource(Protocol):
    """Who builds the decoder-facing error model of one window.

    The window planner holds one of these and asks it per window. The
    run's syndrome source names the shipped answer
    (SyndromeSource.window_model_source): itself when the model comes
    from the same circuit the readouts come from, or a component that
    answers every question with nothing when it has no circuit. The plan
    takes it as its own collaborator (qpu.error_model_provider) so a
    model built anywhere else plugs in.

    The requirement and the returned model are the detector error
    model's records; this port names them by position only, so the
    window side never imports that package to hold the port.
    """

    # per_operation retains the root-owned circuit; none does not require it.
    # Both physical and model consumers declare this independently.
    operation_circuit_scope: str

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

    def finalize_stream_models(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        """Bind the terminal boundary; return whether pending models changed.

        Physical source validation has already succeeded. A finite model
        checks its declared length; an evolving model fixes its final length.
        The provider needs no physical execution state to answer this call.
        True requests rebuilding unqueued models without changing any queued
        or committed model's semantics. False retains the installed models.
        """

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

    Table rows: logical_reference, bandwidth_limited, roce_v2_cpu,
    roce_v2_gpu (LINK_FABRICS, links/link_profiles.py), named by
    links.kind; a row supplies the numbers the section's per-path cards
    override and builds the fabric the root sends on. Every hop of the
    reaction path is priced, and a send delivers by callback with every
    tick of the transfer on the record.

    trace holds transfer_delivered, a trace source the fabric fires once
    per delivered transfer with a TransferRecord. It is on the port
    because the machine connects the traffic ledger, the data-movement
    ledger and the trace writer to whatever answers this port
    (decsim/observe/wiring.py), as the Decoder port carries
    stage_recorded; ns-3's point-to-point device declares its trace
    sources on the device the same way (ns-3
    src/point-to-point/model/point-to-point-net-device.cc, GetTypeId's
    AddTraceSource calls).
    """

    trace: Any

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
    store's output port name it as a port rather than importing the
    class that fills it. Every method here sends; an input that rides no
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

    def send_region(
        self,
        path: transfer_records.LinkPath,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send an escalated region in its request's name; the delay."""


@runtime_checkable
class Channel(Protocol):
    """One physical channel under the fabric: its setup engine and its wire.

    One step below Link: the fabric selects a path's payload, frames it
    with the path's header and hands it to the channel the path's card
    names; the channel decides when the bits cross and calls back at
    the delivery. The split is ns-3's, where the net device frames and
    queues and the channel it is attached to times the crossing (ns-3
    src/point-to-point/model/point-to-point-net-device.cc TransmitStart
    calls point-to-point-channel.cc TransmitStart), and gem5's, where a
    port hands its packets to the packet queue that decides when each
    one goes (gem5 src/mem/port.hh, src/mem/packet_queue.hh:62-63).

    A Link row's build hands the fabric a channel class, which the
    fabric calls once per channel name with that channel's
    ChannelSettings and the engine, so a jittered or credit-limited
    channel is one class and no fabric subclass. framed is a
    FramedPayload (decsim/links/channel.py): the payload bits a
    component sent and the header bits its path adds.
    """

    def send(
        self,
        framed,
        now_ticks: int,
        setup_ticks: int,
        on_delivered: Callable[[transfer_records.Transfer], None],
    ) -> None:
        """Carry one framed payload; on_delivered runs at its delivery."""

    def expected_delay_ticks(
        self, framed, now_ticks: int, setup_ticks: int
    ) -> int:
        """What the transfer would pay if nothing else reached the channel."""


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
    # strong syndrome buffer, one buffer of context on each side of every
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
    reads_a_calibration_table says the point's number comes from an
    offline calibration csv rather than from the section's card, so the
    settings demand threshold_table. built_per_sweep_point says the
    row builds one instance of itself for a whole sweep point, which
    the experiments layer hands to every shot, because the row learns
    across the point's windows; every other row is built by the root
    from the point's threshold in nats, which is its one constructor
    argument.
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

    Table rows: complementary_gap, cluster_gap and extra_cluster_gap
    (decsim/confidence/); each builds itself from the escalation section
    and the weak decoder's settings (from_settings).
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
class RegionProposer(Protocol):
    """The window interaction, as the strong regions see it.

    The escalation package plans a strong region over a weak window it
    is escalating, and that is the one thing it asks of the interaction;
    the rest of WindowInteraction is the windows package's own seam, so
    this port declares the one call and nothing else.
    """

    def plan_strong_region(
        self,
        weak_window: window_records.WindowInfo,
        later_windows: list,
        operation_round_count: int,
    ) -> Optional[window_records.StrongRegionPlan]:
        """The strong window that replaces a weak window, or None."""


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
    window graph of an operation, and when a window has its data.

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


@runtime_checkable
class RoundsPolicy(Protocol):
    """How many syndrome rounds an operation runs for: one or more if decoded.

    An operation the plan does not decode may run none; the planner
    refuses a decoded one with none (frontends/planner.py
    _check_decode_owner_rounds).

    The policies are FixedRounds, PerOperationRounds, CodeRounds,
    GateRounds and TemporalRounds (qpu/round_policies.py), with no table
    and no yaml key: the memory_circuit row fixes its rounds, a Python
    workload may pass its own (a PerOperationRounds may give an operation
    none), and GateRounds is the default. The lattice-surgery unit of d
    rounds per step is Horsman 1111.4022 Sec. 3.1 and Litinski
    1808.02892.
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
    factory that produces on demand answers at once. A row that produces
    ahead of demand queues its first attempt in start, never in its
    constructor, so the order the root builds its components in cannot
    move a tick (gem5's startup, the place to schedule initial events,
    gem5 src/sim/sim_object.hh lines 194 and 280). A row
    declares a decode_queue port (a DecodeQueue), which the root binds
    to the run's decoder manager; a row whose card corrects nothing
    leaves it unread.

    trace holds state_delivered(operation_id, waited_ticks), fired at
    every delivery with the ticks the request waited, which is the
    supply stall; it is on the port because the machine connects the
    run's runtime stamps to whatever answers this port, and a row whose
    requests never wait carries it and never fires it.
    """

    trace: Any

    def start(self) -> None:
        """Queue whatever the factory does before the first request."""

    def request(self, operation_id: int, callback: Callable[[], None]) -> None:
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
    end_idle_period runs when an operation claims the patch and, for
    every idle patch, when the workload completes, so rounds the policy
    has not charged yet can be settled.
    """

    def relay(self, idle_rounds, operation, patch, round_index: int) -> None:
        """Carry one idle round of the patch through the idle accounting."""

    def end_idle_period(self, idle_rounds, operation, patch) -> None:
        """Settle the uncharged rounds: a claim, or the workload's end."""


# ---------------------------- the rows the root reads before it builds


@runtime_checkable
class CodeModel(Protocol):
    """A code card: the numbers the machine reads off a QEC code.

    Table rows: rotated_surface, bivariate_bicycle (CODE_CARDS,
    qpu/settings.py), named by qpu.code_card. A card is a record of
    numbers, not a stabilizer code: the machine prices decoder timing,
    so it asks a card for its name and distance, its window sizes, its
    own round period, the graph size a latency model prices, and the
    bits one round reads out, and for nothing else. The shape is CUDA-Q
    QEC's code base class, a few counts every code implements
    (cudaqx libs/qec/include/cudaq/qec/code.h lines 51-58
    and 140-160), built by name with the code's own options (get_code,
    line 257). The planner, the plan, the round policies, the QPU clock
    and the circuit-less sources call it.
    """

    name: str
    distance: int

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window."""

    def spatial_nodes(self, num_patches: int) -> int:
        """The per-round graph size a latency model prices this card at."""

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits one round of this many patches produces."""


@runtime_checkable
class RowSettings(Protocol):
    """A table row's own yaml keys, read into one record.

    A row with keys of its own declares a nested frozen dataclass named
    Settings that fills this: its fields are the keys, and from_yaml
    reads and checks the ones the yaml wrote, which are all it is
    handed. context is whatever the row's section passes on, nothing
    for most sections and the run's clocks (config.ClockSettings) for a
    decoder tier, whose timing names a clock domain. A row with no keys
    declares no Settings. The section splits its keys from the row's
    and refuses a key neither declares (decsim/tables.py row_settings);
    how the record reaches the row is its table's build call. The shape
    is gem5's: a SimObject's parameters declared on its class
    (gem5 src/mem/SimpleMemory.py:43-53) and handed to
    its constructor as one Params record (src/mem/simple_mem.cc:53).
    """

    @classmethod
    def from_yaml(cls, section, *context) -> "RowSettings":
        """The record, read from the row's own keys the yaml wrote."""


@runtime_checkable
class WorkloadRow(Protocol):
    """A workload row: what the machine runs, as the root reads it.

    Table rows: memory_circuit, circuit_list, surgery_ir (WORKLOADS,
    frontends/settings.py), named by workload.kind. The root never
    builds a workload row; it reads the class. operations turns the
    workload section's record (frontends/settings.py WorkloadSettings,
    whose row_settings holds the row's own RowSettings) and the run's
    code card into the operations and the rounds policy the row fixes,
    or None where the workload's policy applies (build/plan.py
    _operations). has_frontend says whether an operation chain is built
    in front of the run, a fact of the run shape the escalation policy
    checks (build/plan.py build_plan). A row that no yaml can name
    declares a Settings whose from_yaml refuses with a sentence. gem5's
    Workload is the same shape: a SimObject whose parameters sit on its
    class and whose few answers the system reads before it runs
    (gem5 src/sim/Workload.py:46-52,
    src/sim/workload.hh:103-105).
    """

    has_frontend: bool

    def operations(self, settings, code: CodeModel) -> tuple:
        """The operations, and the rounds policy the row fixes or None."""
