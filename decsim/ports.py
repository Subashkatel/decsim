"""The ports: the map of a readout's path through the machine.

One Protocol per handoff between neighbours, one method per handoff, and
the record that crosses it in decsim/records. A component depends on ports,
never on another component's class. Read top to bottom, this file is the
pipeline: the QPU emits a readout, the controller receives it, the store
holds the packed round, the window manager closes a window, the decoder
manager schedules a decode, the decoder returns a result, the frame
commits the correction, the frame releases the controller, the controller
instructs the QPU, and every hop between components rides a link.

A pluggable component's port is sinter's Decoder shape
(sinter/_decoding/_decoding_decoder_class.py), written as a Protocol
because the implementations fill it without inheriting. Observation
reaches a component through callbacks it fires, never through a port, so
every component runs with no observer.
"""

from collections.abc import Callable, Mapping
from typing import Any, Optional, Protocol, Union, runtime_checkable

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records
import decsim.trace_source as trace_source


class Port:
    """One neighbour a component talks to, named on the class, bound once.

    A component declares a port as a class attribute and reads it as an
    ordinary attribute; the part that holds it binds it by assignment,
    as gem5's script assigns one port to another
    (gem5 configs/learning_gem5/part1/simple.py:68). A second bind is
    refused by name, as PortRef.connect does
    (gem5 src/python/m5/params/port_params.py:109-114); a required port
    read before it is bound raises, as gem5's UnboundPortException
    (gem5 src/mem/port.cc:62-65); an optional port reads as None, the
    neighbour a run does not have. Whether the peer answers the Protocol
    is not asked: every bind site is decsim's own build code.
    """

    def __init__(self, protocol: type, optional: bool = False) -> None:
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

    def emit_idle_round(
        self,
        operation_id: Any,  # an opaque identity
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Take one idle cycle of a patch nobody is operating on."""

    def start_command(
        self, command: program_records.RunOperationBody
    ) -> program_records.RunOperationBody:
        """The command as it starts after its patches' idle rounds.

        The operation claims every idle round its patches emitted before
        it started. A segment that declares no stream offset continues
        its stream after every round the stream has had, idle ones
        included, and a feedback source on a protected stream reads it
        from there. Any other command comes back unbound.
        """


# --------------------------------------- the store holds the packed round


@runtime_checkable
class SyndromeBuffer(Protocol):
    """The weak syndrome buffer, as its own round receiver sees it.

    A round occupies a slot from the landing of the hop that carried it,
    and is readable at that same tick. The receiving end asks has_room
    first, counting the writes in flight it reserved (gem5's packet
    store answers `avail() = _maxsize - _size - _reserved`,
    src/dev/net/pktfifo.hh); a round is written once and kept until
    every consumer releases it. A store never refuses a write: a round
    that finds no room waits upstream. occupied_bits is the bits stored
    now.
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
        asks with the packet (src/mem/port.hh:268), so a store of several
        memories answers for the one the round goes to.
        reserved_bits_by_round is the bits each round still crossing
        toward the store will take, by its key. A bounded store raises on
        a round that states no size; a refused round that no free can
        ever make room for stops the run after the action.
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

        The tick is fixed at booking and never withdrawn, as gem5's
        SimpleMemory fixes a response tick at acceptance
        (src/mem/simple_mem.cc:174). Nothing is scheduled: the caller
        schedules its own continuation at the tick.
        """

    def book_read(self, round_keys: tuple) -> int:
        """Take a read of these stored rounds; the tick their bits are out.

        Asked by the store's outgoing port at dispatch; the store sizes
        the read from the widths it keeps. Fixed, as for book_write.
        """

    def release_round(self, round_key: tuple) -> None:
        """Free the round; its consumers are done with it."""

    def retained_fragments(self, round_key: tuple) -> Optional[tuple]:
        """The round's stored fragments; None when it is not stored."""

    def capacity_bits(self) -> Optional[int]:
        """The bits this store is bounded to, or None for unbounded.

        A store bounded some other way than by a bit capacity answers
        None.
        """

    def held_rounds_description(self) -> str:
        """The stored rounds, in one line, for the I/O trace."""

    def check_settled(self) -> None:
        """At the end of a run no round is stored and no hold is live."""


class SyndromeBufferSettings(Protocol):
    """A weak syndrome buffer's settings record, which builds its store.

    bits bounds the store, None for no bound; clock None is the
    machine's. prices_read_bits says whether the store prices a read's
    bits itself, in which case the link out of it may not charge them
    again.
    """

    bits: Optional[int]
    clock: Optional[config.Clock]
    prices_read_bits: bool

    def build(self, engine: engine_module.Engine) -> SyndromeBuffer:
        """A fresh store on these settings."""


@runtime_checkable
class RetainedRounds(Protocol):
    """The same store, as the window side that reads and holds it sees it.

    A hold names the rounds its holder will read from the moment it is
    placed, so the store may hold a round it has not received yet; the
    holder is any record that answers referenced_operation_ids,
    operation_ids_read_at_once and holders_waited_for
    (records/decoding.py).
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

    def open_operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> None:
        """This operation may receive rounds from now on."""

    def has_operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> bool:
        """Whether the store still serves this operation."""

    def close_operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> None:
        """The operation sends no more rounds; a closed one never reopens.

        A round still crossing a priced link when its operation closes
        never enters the store: no hold can name a closed operation, so
        the round has no reader and the receiver drops it at the landing.
        """

    def has_live_operation_reference(
        self,
        operation_id: Any,  # an opaque identity
    ) -> bool:
        """Whether a live hold still names this operation.

        A stored round that no hold names has no reader, so it does not
        keep its operation's result waiting.
        """


@runtime_checkable
class SyndromeRoundSender(Protocol):
    """The syndrome round sender, as the assembler sees it.

    The sender reserves room in every syndrome buffer the round must
    reach and sends it on; a round that finds no room waits in line for
    a retry. strong_crossing_count is the strong-primary rounds sent and
    not yet landed, which the packing stage's bound on rounds in flight
    counts (controller/round_assembly.py RoundsInFlight).
    """

    strong_crossing_count: int

    def admit(self, packed: round_records.PackedRound) -> bool:
        """Write the round where it belongs; False when it found no room."""


@runtime_checkable
class StrongSyndromeRoundReceiver(Protocol):
    """The strong syndrome buffer's receiving end, as its two senders see it.

    The bits still crossing toward the store count as room taken: a
    round's or an escalated region's room is reserved before it leaves
    and kept when it lands.
    """

    def has_room(self, packed: round_records.PackedRound) -> bool:
        """Whether this round's write can land."""

    def reserve_write(self, packed: round_records.PackedRound) -> None:
        """Take the room this crossing round will need, before it leaves."""

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Store a landed round and deliver it on its canonical route."""

    def reserve_region(self, region: round_records.EscalatedRegion) -> None:
        """Take the room an escalated region's rounds will need, or refuse."""

    def receive_region(
        self,
        region: round_records.EscalatedRegion,
        on_stored: Callable[[], None],
    ) -> None:
        """Take an escalated region that landed here: every round its slot.

        on_stored is called once the store holds every round, which is
        later than the landing when this seat forms the rounds first.
        """


@runtime_checkable
class WeakSyndromeRoundReceiver(Protocol):
    """The weak syndrome round receiver, as the controller sees it.

    This end owns the store's room and its landing: has_room counts the
    bits stored and those reserved for writes in flight, and a landed
    round is stored, readable from that tick, and announced.
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
    send, frees the slot, and answers the ticks the link expects.
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

    A feedback-memory round carries no syndrome; it lands at the decoder
    side so that the stream stage it occupies is accounted for.
    """

    def receive_memory_round(
        self,
        source_operation_id: Any,  # an opaque identity
    ) -> None:
        """Take one timing-only round that landed at the decoder side."""


@runtime_checkable
class HeldRounds(Protocol):
    """The waiting line in front of a bounded stage, as the stage sees it.

    A full stage holds nothing back itself: the round waits at the
    sender, and whatever frees a place tells the line so the head can
    try again (gem5 src/mem/port.hh:244-262, sendRetryReq).
    """

    def retry(self) -> None:
        """A slot freed: admit from the head, stop at the first refused."""

    def waiting_round_keys(self) -> set:
        """The keys of the rounds that wait, the head's among them."""


# ------------------------------------------- the window manager closes a window


@runtime_checkable
class WindowInput(Protocol):
    """The window manager, as the controller side sees it.

    Rounds arrive here, and a dynamic stream's shape changes here (its
    boundary, its length, the segments folding into it). Both are
    one-way; the window manager answers only whether it knows a stream
    and which store its windows read.
    """

    def accept_window_input(
        self, packet: round_records.SyndromeRoundPacket
    ) -> None:
        """Publish one stored round to window readiness; never refused."""

    def accept_feedback_memory_round(
        self,
        source_operation_id: Any,  # an opaque identity
    ) -> None:
        """Record one idle or memory round and re-check waiting windows."""

    def prepend_idle_rounds(self, operation_id: int, round_count: int) -> None:
        """Fold pre-gate idle rounds into a batch-style operation."""

    def has_dynamic_stream(
        self,
        stream_id: Any,  # an opaque identity
    ) -> bool:
        """True for a stream whose windows are planned at runtime."""

    def close_stream_boundary(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """Mark a live stream round as a measurement-closed boundary."""

    def seal_stream(
        self,
        stream_id: Any,  # an opaque identity
        stream_round_count: int,
    ) -> None:
        """Close a dynamic stream once its full length has arrived."""

    def bind_stream_operation(
        self,
        operation_id: int,
        stream_id: Any,  # an opaque identity
        stream_offset: int,
    ) -> None:
        """Note which stream and offset a segment's rounds fold into."""

    def bind_required_stream_end(
        self, operation_id: int, required_stream_end: int
    ) -> None:
        """Note the stream round a protected segment's result waits for."""

    def accept_room_round(
        self,
        operation_id: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Record a round that landed in the strong syndrome buffer instead."""

    def accept_boundary(self, window_key: tuple, is_unblocked: bool) -> None:
        """A boundary landed in the window; True when it owed no other."""

    def reads_windows_from(self, store: Optional[SyndromeBuffer]) -> bool:
        """Whether the primary tier's window reads come from this store."""


@runtime_checkable
class WindowPlan(Protocol):
    """The window plan, as the escalation side reads and reshapes it.

    A strong window covers rounds several weak windows were going to
    commit, so the escalation side may absorb a window or move another's
    read start; the rest of the plan stays the window side's.
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

    def later_windows(
        self,
        operation_id: Any,  # an opaque identity
        window_index: int,
    ) -> list:
        """The operation's windows past that index, in index order."""

    def check_absorbable(self, window_keys: tuple) -> None:
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
        per fault representation, and None when no face is pinned.
        """

    def owned_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults the window at that key commits, per representation.

        A strong window pinned on this neighbour takes them as prior
        faults rather than columns it may spend twice (Bombin et al.
        2303.04846 lines 775-788). None when the run builds no error
        models.
        """

    def crossing_faults_of(self, key: tuple) -> Optional[dict]:
        """The faults it commits that reach behind its commit region.

        A region pinned on the escalated window's own weak commit reads
        these: the rest of the window's rounds are what the region
        decodes again. None when the run builds no error models.
        """


@runtime_checkable
class WindowRetention(Protocol):
    """The rounds a window may still read, as the escalation side sees it.

    Which store a strong window reads is the retention's to say, so a
    caller asks for holds and input, never for a store.
    """

    def hold_strong_input(self, job: decoding_records.DecodeJob) -> None:
        """The strong job's context becomes its input hold."""

    def strong_window_input(
        self, builder: "WindowJobBuilder", window: window_records.Window
    ) -> list:
        """The room-side payloads of a strong window, first round stamped."""

    def hold_strong_context(
        self,
        key: tuple,
        strong_request_key: window_records.DecoderRequestKey,
        context_keys: tuple,
        restart_key: Optional[tuple],
    ) -> None:
        """The rounds kept in case the window escalates pass to its request.

        restart_key is the window whose weak commit releases the
        request, or None.
        """

    def context_rounds_in_flight(self, key: tuple, read_keys: tuple) -> tuple:
        """The rounds the strong syndrome buffer lacks that the weak one has."""

    def escalated_rounds(self, round_keys: list) -> tuple:
        """The weak syndrome buffer's packets of these rounds, to carry up."""

    def guard_restart_reads(
        self,
        key: tuple,
        restart_key: Optional[tuple],
        proposed_restart: Optional[window_records.Window],
        strong_request_key: window_records.DecoderRequestKey,
        context_keys: tuple,
        restart_read_keys: tuple,
    ) -> Optional[decoding_records.RephaseGuard]:
        """Hold the restart window's strong context while a plan lands."""

    def replace_window_reads(
        self, key: tuple, window: window_records.Window
    ) -> None:
        """Re-point the window's live holds at its reads."""

    def release_restart_reads(self, key: tuple) -> None:
        """No earlier escalation can re-slice the window: its claim ends."""

    def release_hold_if_live(
        self, owner, store: Optional[RetainedRounds] = None
    ) -> None:
        """Drop a hold that is still live; nothing for one already gone."""

    def release_strong_hold_if_live(self, owner) -> None:
        """Drop a room-side hold when it is still registered."""

    def release_absorbed_strong_hold(
        self,
        key: tuple,
        restart_key: Optional[tuple],
        replacement: decoding_records.PendingStrong,
    ) -> None:
        """Drop an absorbed window's rounds; the strong request holds them."""

    def require_rounds_retained(
        self, label: str, payloads: list, first_round: int, last_round: int
    ) -> None:
        """A strong window starts only once every round it reads is held."""

    def read_keys_for_bounds(
        self,
        operation_id: Any,  # an opaque identity
        start_round: int,
        buffer_hi: int,
        window: Optional[window_records.Window] = None,
    ) -> list:
        """The retained round keys of a possibly cross-operation range."""

    def strong_rounds_before(
        self,
        operation_id: Any,  # an opaque identity
        first_round: int,
        last_round: int,
    ) -> list:
        """The raw rounds before these that a strong read of them carries.

        Those its forming seat has neither formed nor been given.
        """

    def require_retained(
        self,
        round_keys: list,
        purpose: str,
        store: Optional[RetainedRounds] = None,
    ) -> None:
        """Refuse a new consumer if an already-arrived round was released."""

    def require_strong_retained(self, round_keys: list, purpose: str) -> None:
        """The same, on the strong syndrome buffer."""


@runtime_checkable
class WindowRounds(Protocol):
    """The arrivals per operation, as the escalation side reads them.

    A strong region is laid over an operation whose rounds are still
    arriving.
    """

    def operation(
        self,
        operation_id: Any,  # an opaque identity
    ) -> program_records.Operation:
        """The operation of that id."""

    def round_count_for_window(
        self,
        operation_id: Any,  # an opaque identity
        window: window_records.Window,
    ) -> int:
        """The rounds the window reads of its operation."""

    def strong_rounds_arrived(
        self,
        operation_id: Any,  # an opaque identity
    ) -> int:
        """The rounds of the operation stored in the strong syndrome buffer."""


@runtime_checkable
class WindowJobBuilder(Protocol):
    """Where a strong window shape gets a job's identity and its gate."""

    gate: "WindowInputGate"

    def new_request_key(
        self,
        operation_id: Any,  # an opaque identity
        window_id: int,
        tier: window_records.DecoderTier,
    ) -> window_records.DecoderRequestKey:
        """The next request identity, run-wide ordinal included."""


@runtime_checkable
class WindowRequests(Protocol):
    """The submission side, as a strong window shape steers it."""

    def request_if_ready(
        self,
        window: window_records.Window,
        strong_redecode: Optional["StrongRedecode"],
    ) -> None:
        """If the window has its data, submit it through the policy."""

    def withdraw(self, window: window_records.Window) -> None:
        """Withdraw one window's early-shipped, unstarted decode."""


@runtime_checkable
class LogicalLedger(Protocol):
    """Who commits which rounds, as a strong window shape rewrites it."""

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

    A pinned face reads the correction its neighbour committed, Bombin
    et al. 2303.04846's input adaptation (lines 775-788): the later
    decoding task's input is the syndrome plus the corrections already
    committed. The message is one seam layer on decoder_to_decoder.
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

    A kept weak result halts the strong request it made unnecessary,
    held here or on the strong side's manager (Toshio et al. 2510.25222
    lines 606-614).
    """

    def escalate(self, weak_job: decoding_records.DecodeJob) -> None:
        """Ask the strong tier to re-decode the weak job's window."""

    def submit_if_commit_releases(self, window_key: tuple) -> None:
        """A weak window committed: a strong window waiting on it leaves."""

    def cancel_strong_request(self, window_key: tuple) -> None:
        """A kept weak result: its strong request ends, held or submitted."""

    def submit_if_stored_data_releases(
        self,
        operation_id: Any,  # an opaque identity
    ) -> None:
        """A round was stored: a window waiting for its tail leaves."""


@runtime_checkable
class WindowVerdict(Protocol):
    """The window side, as a finished decode returns its result to it.

    The side that owns the window publishes and finalizes every
    correction, primary or strong, so the decision and its outcome stay
    in one place.
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
    boundary is folded in; the decoder side writes it into the storage
    it owns.
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
    asks for the send and never executes it.
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

    A confidence that reads several forced-class solves joins them here
    and calls the verdict once they are all in.
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
        on the event (simpy/core.py).
        """

    def enqueue_without_input(
        self,
        round_count: int,
        on_done: Callable[[], None],
        label: str = "external",
        code: Optional[str] = None,
        spatial_nodes: Optional[int] = None,
    ) -> None:
        """Queue a decode with no syndrome data; on_done at its end."""

    def withdraw_window(self, window_key: tuple) -> None:
        """Take back a window's not-yet-started decode; it is superseded."""

    def release_parked(self, window_key: tuple) -> None:
        """The window's last boundary arrived: start its parked decode."""

    def await_strong_result(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The window asked for this request's strong result.

        A result that finishes before its selection lands waits in its
        unit until accept_selection.
        """

    def accept_selection(
        self, window_key: tuple, request_key: window_records.DecoderRequestKey
    ) -> None:
        """The selection landed: the request's result may reach the window."""

    def charge_soft_output(
        self, job: decoding_records.DecodeJob, ticks: int
    ) -> None:
        """Charge the confidence's own computation on the job's unit.

        The ticks go on the unit that produced the evidence, whose
        service then carries them.
        """

    def resolve_weak_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
        verdict: decoding_records.Verdict,
    ) -> None:
        """The window side decided this weak request; close its attempt."""

    def read_result(self, job: decoding_records.DecodeJob) -> None:
        """The window side, or the confidence join, has this result in hand.

        A tier whose result blocks its unit gets the unit back here.
        """

    def cancel_strong(self, window_key: tuple) -> None:
        """A kept weak result: the window's strong request ends where it is.

        Queued, crossing, running, or done and waiting; nothing when none
        is live or done.
        """

    def close_companion_request(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> None:
        """This forced solve lost; its window is answered by the other."""


# ------------------------------------------- the decoder returns a result


@runtime_checkable
class Decoder(Protocol):
    """One decoder: correctness and timing from one object.

    sinter's abstract class with defaults (decsim/decoders/decoder.py,
    DecoderBase) fills start, cancel and occupancy from decode and
    latency. The manager asks occupancy at dispatch, calls start once
    the input has landed, and cancel when the request is withdrawn; a
    decoder measured on the host clock answers occupancy with None.

    stage_recorded, window_checked and forced_solve_unavailable are
    trace sources (a DecoderStageRecord per internal stage, a referee
    audit, a window model with no forced class); DecoderBase gives a
    decoder that fires none of them the silent source, so the machine
    connects listeners without asking what the decoder is.

    decoder_evidence declares what a decode can show beyond the
    correction (decsim/records/decoding.py DecoderEvidence); a
    confidence signal declares the same set as its requirement, and the
    build refuses a pairing the decoder cannot serve, quoting
    missing_evidence_reasons when it has one. A decoder that wraps
    another answers run_seed_children (decsim/seeding.py
    RunSeedComposite), or the one inside it is hidden from the trace,
    the stage ledger and the referee audit.
    """

    fault_model_requirement: Any
    stage_recorded: Union[trace_source.TraceSource, trace_source.SilentSource]
    window_checked: Union[trace_source.TraceSource, trace_source.SilentSource]
    forced_solve_unavailable: Union[
        trace_source.TraceSource, trace_source.SilentSource
    ]
    decoder_evidence: frozenset
    missing_evidence_reasons: dict

    def decode(
        self, job: decoding_records.DecodeJob
    ) -> decoding_records.DecodeResult:
        """The window's correction and its logical observables.

        A timing-only decoder leaves both None.
        """

    def latency(self, job: decoding_records.DecodeJob) -> int:
        """The whole job's service time in ticks, known at dispatch."""

    def start(
        self,
        job: decoding_records.DecodeJob,
        engine: engine_module.Engine,
        on_result: Callable[[Optional[decoding_records.DecodeResult]], None],
    ) -> None:
        """Run the job on the unit; on_result runs once at its output."""

    def cancel(self, job: decoding_records.DecodeJob) -> None:
        """Stop a started job; on_result never runs for it."""

    def occupancy(self, job: decoding_records.DecodeJob) -> Optional[int]:
        """Ticks the unit's compute is held from the start; None if measured."""


@runtime_checkable
class DecoderSettings(Protocol):
    """A decoder's settings record, as a tier and a confidence see it.

    A decoder that grows clusters also holds a weight_step and a timing,
    which a cluster confidence reads.
    """

    name: str

    def build(self) -> Decoder:
        """A fresh decoder of these settings."""


@runtime_checkable
class StrongBackend(Protocol):
    """The device a strong decode runs on, as the strong decoder sees it.

    The link cards price the two links from echo round trips, which
    already hold the landing, the notice and the write back (Backline
    2609.09270 lines 2355-2360, NVQLink 2510.25213 lines 528-533). So a
    backend reports only the time beyond the echo on the same path, the
    launch included, and never the wait: the wait emerges from its
    capacity, a queue decsim keeps. One CUDA-Q dispatcher's decode ends
    before its next begins (cudaqx
    docs/sphinx/examples_rst/qec/realtime_relay_bp.rst:49-50).

    A decode is a sequence of steps (decoding_records.Step), each holding
    one resource: the dispatcher every request enters by, in slot order
    (cuda-quantum dispatch_kernel.cu v0.15.2 lines 552-555), or a worker
    (host_api.md lines 1065-1113). capacities says how many of each the
    device has; decsim keeps one arrival-order queue per resource, and a
    step on a new resource holds the old one until it ends. The echo's
    own steps appear with zero ticks, so every tick is counted once.

    submit takes the count of decodes already running, since the
    device's time depends on it (IonQ 2608.25027 lines 550-554); it is
    called once the request holds the dispatcher. steps and result are
    asked after submit, once each.
    """

    def capacities(self) -> Mapping[str, int]:
        """Each resource's count; the dispatcher's is always there."""

    # The ticket is an opaque identity only the backend that issued it reads.
    def submit(self, request: decoding_records.DecodeJob, running: int) -> Any:
        """Start one region's decode with running others on the device."""

    def steps(
        self,
        ticket: Any,  # an opaque identity
    ) -> tuple[decoding_records.Step, ...]:
        """The decode's steps beyond the echo on the same path, in order."""

    def result(
        self,
        ticket: Any,  # an opaque identity
    ) -> decoding_records.DecodeResult:
        """The correction and observables; decode_status marks unconverged."""


# ------------------------------------------- the frame commits the correction


@runtime_checkable
class Frame(Protocol):
    """The Pauli frame, as the decoder output sees it."""

    def commit_correction(
        self,
        *,
        window_key: tuple,
        logical_observables: Optional[tuple[int, ...]],
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

    The decision leaves by the frame's end of frame_to_controller.
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

    The QPU ends a body, the frame's release path lands a decision, and
    the protected streams free a waiting operation after a cadence
    change.
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
    controller's. The start callback rides with the issue, so neither
    side holds the other.
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
        is_workload_complete: bool,
    ) -> None:
        """Successors released: close boundaries, seal streams, stop the QPU."""


# ------------------------------------------ the controller instructs the QPU


@runtime_checkable
class Qpu(Protocol):
    """The QPU, as the controller sees it.

    Where the next body may start is the QPU's own cadence, so the
    controller asks for the boundary rather than computing it.
    """

    def issue(self, command: program_records.RunOperationBody) -> None:
        """Queue one operation body; it starts on the next cycle boundary."""

    def next_boundary(self) -> int:
        """The cycle boundary at or after now, where an issue would start."""

    def finish(self) -> None:
        """The program is complete: idle patches stop after this cycle."""

    def are_patches_idle(
        self,
        operation_id: Any,  # an opaque identity
        patches: tuple,
    ) -> bool:
        """Every patch is idle after this same operation, as the QPU owns it.

        Asked before a joint stream advances; a busy member must not run.
        """

    def emit_idle_stream_round(
        self,
        operation: program_records.Operation,
        stream_id: Any,  # an opaque identity
        global_round: int,
        *,
        is_final: bool,
    ) -> None:
        """Deliver one protection round, its terminal readout when is_final.

        The terminal readout replaces the bulk fragment, not a second round.
        """

    def emit_feedback_memory_round(
        self,
        operation_id: Any,  # an opaque identity
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Deliver the timing-only round of an idle patch."""

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Refuse a seal whose length differs from the rounds executed.

        Only the QPU's source can attest that the stream ended there.
        """


@runtime_checkable
class SyndromeSource(Protocol):
    """What the QPU reads out each round for an operation.

    Payload bits are raw measurement bits per round; a source with a
    detector formation table also answers DetectionEventFormer.
    shot_sampled(operation, detection_events) fires once when a physical
    source's complete shot is available (a live source's at final
    readout); a source that draws nothing carries the silent source.
    emits_bit_values says whether payloads carry measured values or
    their sizes alone.
    """

    # none means this consumer does not need Operation.circuit; it does not
    # require removal when an independent model provider needs the circuit.
    operation_circuit_scope: str
    emits_bit_values: bool
    shot_sampled: Union[trace_source.TraceSource, trace_source.SilentSource]

    def declare_stream(
        self,
        stream_operation: program_records.Operation,
        round_count: int,
    ) -> Optional[int]:
        """Bind a stream before execution; its physical round limit or None.

        None is an open-ended source. Nothing is sampled or modelled here.
        """

    def validate_stream_length(
        self,
        stream_operation: program_records.Operation,
        stream_round_count: int,
    ) -> None:
        """Require a sealed length consistent with the physical history.

        A finite source requires its declared length; a live source, a
        final readout at this length.
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

        A duration-dependent source checks round_period_ticks against its
        circuit's period before executing anything.
        """

    def round_payloads(
        self, operation: program_records.Operation, round_index: int
    ) -> list[round_records.QPUReadout]:
        """The round's acquisitions in circuit measurement order.

        The QPU numbers the fragments from list order, unless the
        operation declares the first slot of its group in a partitioned
        round.
        """

    def finalize_stream_round(
        self, operation: program_records.Operation, source_round_count: int
    ) -> list[round_records.QPUReadout]:
        """Final data readout, ordered by the round_payloads contract."""

    def idle_round_payloads(
        self,
        operation: program_records.Operation,
        stream_id: Any,  # an opaque identity
        global_round: int,
        *,
        is_final: bool,
        round_period_ticks: int,
    ) -> list[round_records.QPUReadout]:
        """The protection round the controller requests, possibly its last.

        For a live source, is_final selects a terminal fragment holding
        this round's checks and the final data readout; a finite source
        keeps its declared schedule. Ordered as round_payloads.
        """

    def logical_observable_truth(
        self,
        operation_id: Any,  # an opaque identity
    ) -> Optional[tuple[int, ...]]:
        """The observable flips the source drew, or None when it draws none."""

    def readout_departure_tick(
        self, readout: round_records.QPUReadout, readout_tick: int
    ) -> int:
        """The tick this readout leaves the chip, at or after readout_tick.

        A source with a fixed readout delay or a jitter names a later
        tick, as gem5's queued port names the absolute tick a packet is
        sent (gem5 src/mem/qport.hh:94 schedTimingResp). The clock sends
        it then, behind every earlier readout of its patches.
        """

    def window_model_source(self) -> "WindowModelSource":
        """Where the run's window error models come from, by default.

        A source with a circuit answers itself; one without answers a
        component that builds no model. qpu.error_model_provider
        overrides it, as sinter derives a task's model from its circuit
        only when none is given (sinter/_data/_task.py lines 71, 87-89).
        """


@runtime_checkable
class DetectionEventFormer(Protocol):
    """Who holds the recipes that turn a round's outcomes into its events.

    A source with a circuit reads them off it
    (detector_error_model/detector_formation.py build_formation_table);
    a live stream's table grows as its rounds execute. A detector
    compares this round's outcomes against the round before (LILLIPUT
    2108.06569 lines 499-510).
    """

    def formation_table(
        self,
        operation_id: Any,  # an opaque identity
    ):
        """The operation's formation table, the rounds executed so far."""


@runtime_checkable
class DetectionEventPlacement(Protocol):
    """Where the machine forms a round's detection events, and what it costs.

    The seats are the points on a round's path from the controller to a
    decoder (detector_error_model/settings.py SEATS); each component
    there asks with its own seat name. The values are the same at every
    seat, a parity of the raw outcomes, so a seat moves only the width
    the round carries, the state held and the clock charged (Google
    2408.13687 lines 474-476, Maurer 2510.21600 lines 234-236, Caune
    2410.05202 lines 1252-1255, LILLIPUT 2108.06569 lines 500-509).
    clock is the clock cycles_at counts on; None when nothing is
    charged.
    """

    clock: Optional[config.Clock]

    def forms_at(self, seat: str) -> bool:
        """Whether the seat forms the rounds that cross it."""

    def form_at(
        self, seat: str, fragments: tuple, rounds_before: tuple = ()
    ) -> tuple:
        """The fragments as they leave the seat: formed, or as they came.

        rounds_before are the raw rounds before the first fragment, held
        by the seat for that round's detectors and never returned.
        """

    def width_at(self, seat: str, fragments: tuple) -> Optional[int]:
        """The width one round's fragments take as they leave the seat.

        Forms and holds nothing, so a store can weigh its room before the
        round lands. None when the width is unknown.
        """

    def rounds_needed_before(
        self,
        seat: str,
        operation_id: Any,  # an opaque identity
        first_round: int,
        last_round: int,
    ) -> tuple:
        """The raw rounds before a read's first round the seat must be given.

        Those the read's unformed rounds' recipes read, less those the
        seat holds raw.
        """

    def earlier_rounds_read(
        self,
        operation_id: Any,  # an opaque identity
        first_round: int,
    ) -> tuple[int, ...]:
        """The raw rounds before first_round it or any later round reads.

        On a live stream as far back as its program reaches. None with no
        recipes.
        """

    def retire_round(self, round_key: tuple) -> None:
        """The round left the store the windows read; every seat drops it."""

    def claim_rounds(self, seat: str, round_keys: tuple) -> tuple:
        """The round keys no earlier job of the seat claimed, now claimed."""

    def return_claim(self, seat: str, round_keys: tuple) -> None:
        """A job that never started gives its claimed round keys back."""

    def check_settled(self) -> None:
        """At the end of a run no seat holds a raw round; a leak raises."""

    def cycles_at(self, seat: str, round_count: int) -> int:
        """The cycles of forming round_count rounds together at the seat."""


@runtime_checkable
class WindowModelSource(Protocol):
    """Who builds the decoder-facing error model of one window.

    The window planner asks it per window; the default is the syndrome
    source's own (SyndromeSource.window_model_source). The requirement
    and the model are the detector error model's records, named here by
    position only, so the window side never imports that package.
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
        window_protocol: window_records.WindowProtocol,
    ) -> list:
        """One model per window of a planned operation, in window order."""

    def window_model_for_stream(
        self,
        stream_id: Any,  # an opaque identity
        window: window_records.Window,
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
        """Bind the terminal boundary; whether pending models changed.

        A finite model keeps the length it registered; an evolving one
        fixes its final length. True asks for the unqueued models to be
        rebuilt; queued and committed ones never change.
        """

    def strong_window_model_for_operation(
        self,
        operation: program_records.Operation,
        window: window_records.Window,
        round_count: int,
        *,
        fault_model_requirement,
        fault_exclusion_ranges: tuple = (),
        prior_faults: Optional[dict] = None,
    ):
        """One strong window's model, its non-owned round ranges excluded.

        prior_faults names the faults a pinned face's neighbour has
        already committed; they are no columns of this model at all
        (Bombin et al. 2303.04846 lines 775-788).
        """


# -------------------------------------------------- every hop rides a link


@runtime_checkable
class Link(Protocol):
    """The link fabric as every sender sees it.

    Every hop of the reaction path is priced, and a send delivers by
    callback with every tick of the transfer on the record. trace holds
    transfer_delivered (a TransferRecord per delivered transfer) and
    frame_landed (a FrameRecord per frame a channel moves), declared on
    the port as ns-3's point-to-point device declares its trace sources
    (point-to-point-net-device.cc, GetTypeId's AddTraceSource calls).
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

    The sender says what it moves and the adapter builds the transfer's
    attribution and calls Link.send. Whether an input moves at all is
    the sending store's decision, made before it calls here.
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

    The fabric frames a path's payload with its header and hands it to
    the channel the path's card names; the channel decides when the
    bits cross and calls back at the delivery. The split is ns-3's,
    where the net device frames and the channel times the crossing
    (point-to-point-net-device.cc TransmitStart calls
    point-to-point-channel.cc TransmitStart), and gem5's packet queue
    (src/mem/packet_queue.hh:62-63). framed is a FramedPayload
    (links/channel.py). trace holds frame_landed, one FrameRecord per
    frame.
    """

    trace: Any

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


@runtime_checkable
class Framing(Protocol):
    """How a packet channel cuts one message into its wire's frames.

    A message is the path's header and its payload; the channel
    serializes, credits and acknowledges the frames one by one. The cut
    is Garnet's divCeil(size, width) flits (gem5
    src/mem/ruby/network/garnet/NetworkInterface.cc:382-387) and ns-3's
    header on every packet (point-to-point-net-device.cc:528).
    """

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """The wire bits of each frame, in sending order; at least one."""


# ------------------------------------ the pluggable policies off the path


@runtime_checkable
class BoundaryPolicy(Protocol):
    """When a committed window ships its boundary to the windows after it.

    A provisional boundary that ships is never revised, so a run that may
    revise a weak result needs a policy that holds;
    ships_provisional_boundaries declares that fact.
    """

    ships_provisional_boundaries: bool

    def on_commit(self, window: window_records.Window, *, final: bool) -> bool:
        """Whether to ship the boundary now."""


class BoundaryPolicySettings(Protocol):
    """A boundary policy's settings record, which builds the policy."""

    def build(self) -> BoundaryPolicy:
        """A fresh policy."""


class StrongWindowBoundaries(Protocol):
    """The boundary policy a strong window's settings give the windows.

    A strong window that absorbs the weak windows it covers gives eager,
    since the weak chain must keep committing; one that absorbs nothing
    gives held, since its escalation would revise a shipped boundary.
    """

    boundary_policy: BoundaryPolicySettings


@runtime_checkable
class EscalationPolicy(Protocol):
    """Whether and when a window is decoded again by the strong tier.

    A run with no switching leaves this port unbound. The policy decides
    and is told, the shape of gem5's conditional predictor
    (src/cpu/pred/conditional.hh: lookup answers, update teaches): it
    builds no job and sends nothing. check_plan refuses, once at build,
    a run the policy cannot serve.
    """

    def check_plan(self, plan: decoding_records.RunShape) -> None:
        """Refuse, with a sentence, a run shape the policy cannot serve."""

    def tiers_for_ready_window(
        self, window: window_records.Window
    ) -> tuple[window_records.DecoderTier, ...]:
        """The tiers that decode the complete window now, primary first.

        Asked once per window, when its rounds are complete and its
        decision cycles have passed (windows/decode_requests.py).
        """

    def verdict_for_weak_result(
        self,
        job: decoding_records.DecodeJob,
        result: decoding_records.DecodeResult,
    ) -> decoding_records.Verdict:
        """Keep the weak result as final, or escalate its window.

        Asked once per window, when its weak result arrives
        (windows/window_commits.py).
        """

    def learn_from_strong_result(
        self, window_key: tuple, result: decoding_records.DecodeResult
    ) -> None:
        """The strong tier answered for the window; a source may learn.

        Told once per strong decode, when it ends
        (decoders/decode_outcomes.py). Under bulk_strong one merged
        decode ends several escalations and is told once, under its
        first window's key.
        """


@runtime_checkable
class ThresholdSource(Protocol):
    """Where the switching policy's threshold comes from.

    A source that audits by escalating (it labels a kept window by
    re-decoding it on the strong tier) needs one serial strong re-decode
    per window, so Switching refuses it beside run_both_at_once and the
    double window.
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

    source names the signal so the switching policy can refuse another
    one's; fault_model_requirement is what a window model must offer;
    decoder_evidence_requirement is what the decode must show
    (records/decoding.py DecoderEvidence), held against the weak
    decoder's declaration at build, with evidence_refusal the sentence
    that refusal prints; forced_logical_classes are the classes the
    window is decoded in, one job each, empty for one ordinary decode.
    """

    source: decoding_records.SoftOutputSource
    fault_model_requirement: Any
    decoder_evidence_requirement: frozenset
    evidence_refusal: str
    forced_logical_classes: tuple

    def compute(self, solves: tuple) -> decoding_records.SoftOutputComputation:
        """The window's confidence and what computing it cost.

        The soft output is None when the evidence is missing. The ticks
        are the signal's own computation: zero for a subtraction, the
        measured or declared time of a walk over the decode's growth.
        """


@runtime_checkable
class RegionProposer(Protocol):
    """The window interaction, as the strong regions see it.

    The rest of WindowInteraction is the windows package's own seam.
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

    The seam is the destination's oldest round layer, turned into the
    bits the wire carries.
    """

    def bits(self, seam: window_records.BoundarySeam) -> int:
        """The bits one hand-off takes in this representation."""


@runtime_checkable
class WindowingScheme(Protocol):
    """How an operation's rounds are cut into windows.

    The static window graph of an operation, and when a window has its
    data. Three facts about the layout are declared, never read off the
    class (gem5's port API, arXiv 2007.03152 lines 489-491):

    has_trailing_tail_context: the last window this scheme lays out
        reads rounds past its own commit, which is what the switching
        recovery re-reads when a strong result revises a window.
    commits_in_one_serial_chain: the windows commit one after another in
        stride order, which the double window absorbs.
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
    ) -> window_records.OperationWindowPlan:
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

    An operation the plan does not decode may run none. The
    lattice-surgery unit of d rounds per step is Horsman 1111.4022 Sec.
    3.1 and Litinski 1808.02892.
    """

    def rounds_for(
        self,
        operation: program_records.OperationPlanningView,
        code: "CodeModel",
    ) -> int:
        """The operation's round count on this code."""


@runtime_checkable
class MagicStateFactory(Protocol):
    """Where a non-Clifford operation gets its magic state.

    The runtime asks and is called back. A factory that produces ahead
    of demand queues its first attempt in start, never in its
    constructor, so the build order cannot move a tick (gem5's startup,
    src/sim/sim_object.hh lines 194 and 280). A factory declares a
    decode_queue port, bound to the run's decoder manager. trace holds
    state_delivered(operation_id, waited_ticks), the supply stall of
    each delivery.
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

    end_idle_period runs when an operation starts on the patch and, for
    every idle patch, when the workload completes, so uncharged rounds
    are settled.
    """

    def relay(
        self,
        idle_rounds,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
        round_index: int,
    ) -> None:
        """Carry one idle round of the patch through the idle accounting."""

    def end_idle_period(
        self,
        idle_rounds,
        operation: program_records.Operation,
        patch: Any,  # an opaque identity
    ) -> None:
        """Settle the uncharged rounds: a claim, or the workload's end."""


# ----------------------------- what the root reads before it builds


@runtime_checkable
class CodeModel(Protocol):
    """A code card: the numbers the machine reads off a QEC code.

    A card is a record of numbers, not a stabilizer code: the machine
    prices decoder timing and asks for nothing else. The shape is CUDA-Q
    QEC's code base class, a few counts every code implements (cudaqx
    libs/qec/include/cudaq/qec/code.h lines 51-58 and 140-160).
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

    def spatial_nodes(self, patch_count: int) -> int:
        """The per-round graph size a latency model prices this card at."""

    def syndrome_bits_per_round(self, patch_count: int) -> int:
        """Syndrome bits one round of this many patches produces."""

    def data_bits_per_readout(self, patch_count: int) -> int:
        """Data-qubit bits the final readout of this many patches adds."""
