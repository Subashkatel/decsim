"""A syndrome buffer: finished rounds held until their last hold releases.

The plain store, built from SyndromeBufferSettings. The store's own
round receiver writes each round once at its landing
(accept_packed_round) after asking has_room, the window side keeps it
alive with holds (RoundHolds), and the slot is freed when the last hold
releases; the waiting line then hears it, so a round held for room
enters in order. Capacity is bits, gem5's packet
store: its size is declared on the store itself
(`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
src/dev/net/Ethernet.py) and the room test takes the packet's own
length, `avail()` over the reserved bytes and `reserve(len)` before the
data lands (src/dev/net/pktfifo.hh). The blocked port is gem5's too
(src/mem/cache/base.cc clearBlocked schedules the retry): the store
never refuses a write, it answers room first. A round's status (its
packet, the bits it holds, its publication tick) lives on its record,
gem5's CacheBlk.

The store reports through trace sources (trace_source.py) and
runs with no listener: round_stored(round_key, packet) when a slot is
taken, round_published(round_key, tick) when the round's data is ready
for the windows, round_released(round_key) when the slot frees;
hold_registered(holder, round_keys), hold_transferred(old_holder,
new_holder) and hold_released(holder) for the consumers' tokens;
access_served(direction, port_index, round_keys, arrival_tick,
start_tick, completion_tick) for every write and read a store with ports
books (ported_syndrome_buffer.py). This store holds no port and fires none.
"""

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.round_holds as round_holds
import decsim.trace_source as trace_source


@dataclasses.dataclass(frozen=True)
class SyndromeBufferSettings:
    """The plain store: its capacity, and a flat cost per write and per read.

    bits bounds the store, the capacity a memory is declared in; None is
    unbounded. gem5 sizes its packet store the same way, in bytes on the
    store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
    src/dev/net/Ethernet.py) and answers room against the packet's own
    size (`avail()` over the reserved bytes, `reserve(len)` before the
    data lands, src/dev/net/pktfifo.hh). A full store makes the
    controller hold the finished round and write it in order once a slot
    frees, the backpressure real systems apply to their source (Caune et
    al. 2410.05202: the sequencer stalls on the decoder's status
    register).
    A write of a round costs write_cycles and a read of a decode's
    rounds, or of an idle round leaving for the decoder, read_cycles, on
    clock, whatever their width, and no access waits for another:
    SimpleMemory's latency with no bandwidth term (gem5
    src/mem/SimpleMemory.py:49, simple_mem.cc:174). A zero cost runs
    synchronously without clock-edge alignment. clock None is the
    machine's clock.
    """

    bits: Optional[int] = None
    clock: Optional[config.Clock] = None
    write_cycles: int = 0
    read_cycles: int = 0

    # the read is one flat cost, so a rate on the link out of the store
    # is the only price of the read's bits
    prices_read_bits: ClassVar[bool] = False

    def __post_init__(self) -> None:
        config.check_capacity_bits("syndrome_buffer.bits", self.bits)
        config.check_cycles("syndrome_buffer.write_cycles", self.write_cycles)
        config.check_cycles("syndrome_buffer.read_cycles", self.read_cycles)

    def build(self, engine: engine_module.Engine) -> "SyndromeBuffer":
        """A fresh store on these settings."""
        return SyndromeBuffer(self, engine)


class SyndromeBuffer:
    """The store: rounds by key, their holds, and the operations it serves.

    Its state is the settings, the engine its accesses are timed on,
    the rounds, the holds, the operations and its events (trace).
    """

    # a store built with no waiting line in front of it frees its slots
    # with nobody to tell
    held_rounds = ports.Port(ports.HeldRounds, optional=True)
    # the run's former seats, told when a round leaves the store the
    # plan's windows read, which no read forms after; None on the other
    # store
    detection_events = ports.Port(ports.DetectionEventPlacement, optional=True)

    def __init__(
        self,
        settings: SyndromeBufferSettings,
        engine: engine_module.Engine,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.round_by_key: dict = {}
        self.holds = round_holds.RoundHolds()
        # operation id -> True while it may receive rounds, False once
        # closed; a closed identity never reopens
        self.operations: dict = {}
        self.trace = _TraceSources()
        self._check_costs_have_a_clock()

    # ---- the port

    def has_room(
        self,
        round_key: tuple,
        bits: Optional[int],
        reserved_bits_by_round: Mapping[tuple, int],
    ) -> bool:
        """Whether a round of that many bits fits beside what is taken.

        The reserved bits are the room the crossing rounds have already
        taken, gem5's `_reserved` in `avail() = _maxsize - _size -
        _reserved` (src/dev/net/pktfifo.hh).
        """
        capacity = self.settings.bits
        if capacity is None:
            return True
        if bits is None:
            self._refuse_unsized_round(capacity)
        if bits > capacity:
            self._refuse_round_wider_than_store(round_key, bits, capacity)
        reserved_widths = reserved_bits_by_round.values()
        reserved_bits = sum(reserved_widths)
        taken = self.occupied_bits + reserved_bits
        return taken + bits <= capacity

    def accept_packed_round(
        self,
        packet: round_records.SyndromeRoundPacket,
        *,
        publication_tick: Optional[int],
    ) -> None:
        """Keep one landed round, readable at that tick; None publishes none."""
        packet_bits = round_records.fragment_wire_bits(packet.fragments)
        round_key = (packet.operation_id, packet.round_index)
        assert self.has_room(round_key, packet_bits, {}), (
            "a round was written into a full store"
        )
        assert round_key not in self.round_by_key, (
            f"round {round_key!r} was written twice"
        )
        self._open(packet.operation_id)
        held_bits = round_records.stated_bits(packet_bits)
        stored = _StoredRound(packet, held_bits)
        stored.publication_tick = publication_tick
        self.round_by_key[round_key] = stored
        self.trace.round_stored.fire(round_key, packet)
        if publication_tick is not None:
            self.trace.round_published.fire(round_key, publication_tick)

    def book_write(self, round_key: tuple, bits: Optional[int]) -> int:
        """The tick this round's write completes: write_cycles on its clock.

        This store holds no port, so a write never waits for another
        access: it takes write_cycles from the clock edge at or after
        now, gem5 SimpleMemory's latency with no bandwidth term
        (src/mem/simple_mem.cc:174), and a zero cost completes now.
        """
        del round_key, bits
        return self._access_completion_tick(self.settings.write_cycles)

    def book_read(self, round_keys: tuple) -> int:
        """The tick a read of these rounds completes: read_cycles on its clock.

        One read of a job's rounds costs read_cycles whatever their
        width, and never waits for another access (book_write).
        """
        del round_keys
        return self._access_completion_tick(self.settings.read_cycles)

    def release_round(self, round_key) -> None:
        """Free one unheld round; its consumers are done with it."""
        if round_key not in self.round_by_key:
            raise RuntimeError(f"round {round_key!r} is not stored")
        if self.holds.is_held(round_key):
            raise RuntimeError(f"round {round_key!r} has live consumer holds")
        self._free_round(round_key)

    def capacity_bits(self) -> Optional[int]:
        """The bits this store is bounded to, or None for unbounded.

        The trace lane and the two round receivers ask this instead of
        reading the settings record, so a store bounded some other way
        answers for itself.
        """
        return self.settings.bits

    # ---- reads

    @property
    def occupancy(self) -> int:
        """The rounds stored now."""
        return len(self.round_by_key)

    @property
    def occupied_bits(self) -> int:
        """The bits stored now; a round that states no size holds none.

        Summed over the stored rounds on each ask, as the decoder memory
        sums its inputs, so the rounds are the one record of what is held.
        """
        return self._stored_bits_of(self.round_by_key)

    def retained_fragments(self, round_key) -> Optional[tuple]:
        """The stored round's fragments, or None before and after storage."""
        stored = self.round_by_key.get(round_key)
        if stored is None:
            return None
        return stored.packet.fragments

    def is_round_held(self, round_key) -> bool:
        """Whether a consumer keeps this round, stored or still expected.

        A hold names the rounds its holder reads from the moment it is
        placed, so a round with a hold and no fragments is one the store
        expects and has not received.
        """
        return self.holds.is_held(round_key)

    def publication_tick(self, round_key) -> Optional[int]:
        """The tick the round was published to the windows, or None."""
        stored = self.round_by_key.get(round_key)
        if stored is None:
            return None
        return stored.publication_tick

    # ---- operation scope

    def open_operation(self, operation_id) -> None:
        """Admit rounds and holds of this operation."""
        self._open(operation_id)

    def has_operation(self, operation_id) -> bool:
        """True while the operation may still receive rounds."""
        return self.operations.get(operation_id, False)

    def close_operation(self, operation_id) -> None:
        """Retire an operation once none of its rounds or holds are live."""
        live_rounds = self._stored_rounds_of(operation_id)
        if live_rounds:
            raise RuntimeError(
                f"operation {operation_id!r} has live buffer rounds "
                f"{live_rounds!r}"
            )
        if self.holds.references(operation_id):
            raise RuntimeError(
                f"operation {operation_id!r} has live consumer holds"
            )
        self.operations[operation_id] = False
        open_ids = self._open_operation_ids()
        self.holds.forget_released_outside(open_ids)

    def has_live_operation_reference(self, operation_id) -> bool:
        """True while any live hold refers to this operation."""
        return self.holds.references(operation_id)

    # ---- holds

    def register_hold(self, holder, round_keys) -> None:
        """Keep the listed rounds alive for one consumer token."""
        record = self._hold_record(holder, round_keys)
        self.holds.register(holder, record)
        self.trace.hold_registered.fire(holder, record.round_keys)

    def replace_hold(self, holder, round_keys) -> None:
        """Re-point a live hold at a new set of rounds."""
        record = self._hold_record(holder, round_keys)
        orphaned = self.holds.replace(holder, record)
        self._free_stored(orphaned)

    def transfer_hold(self, old_holder, new_holder) -> None:
        """Move a live hold to a new token without freeing its rounds."""
        self.holds.transfer(old_holder, new_holder)
        self.trace.hold_transferred.fire(old_holder, new_holder)

    def release_hold(self, holder) -> None:
        """Drop a hold; rounds with no remaining holder are freed."""
        orphaned = self.holds.release(holder)
        self.trace.hold_released.fire(holder)
        self._free_stored(orphaned)

    def has_hold(self, holder) -> bool:
        """True while this holder token is live."""
        return self.holds.is_live(holder)

    def hold_round_identities(self, holder) -> tuple:
        """The rounds a live holder keeps."""
        return self.holds.round_keys_of(holder)

    def release_round_if_unheld(self, round_key) -> bool:
        """Free a stored round only when no consumer holds it."""
        if round_key not in self.round_by_key:
            return False
        if self.holds.is_held(round_key):
            return False
        self._free_round(round_key)
        return True

    # ---- settlement and the trace

    def check_settled(self) -> None:
        """At the end of a run nothing may still be held: a leak is a bug.

        A bounded store that ends with rounds under a live hold is too
        small for that hold: the hold waits for a round that needs the
        room its own stored rounds keep, so nothing ever frees. The stop
        names the widest such hold, as gem5 stops a run on a deadlock it
        detects rather than let it idle ("Possible Deadlock detected",
        src/mem/ruby/system/Sequencer.cc:236-239); the size a store needs
        is its widest hold's rounds together, and restart and seam
        re-reads place holds only at run time.
        """
        if self.round_by_key:
            self._refuse_rounds_left()
        held_keys = self.holds.held_round_keys()
        if held_keys:
            raise RuntimeError(
                f"the syndrome buffer has unresolved holds on {held_keys} "
                f"(rounds expected but never written, or holders never "
                f"released)"
            )

    def held_rounds_description(self) -> str:
        """One compact line of the store's contents, for the I/O trace."""
        round_indices_by_operation: dict = {}
        for operation_id, round_index in self.round_by_key:
            indices = round_indices_by_operation.setdefault(operation_id, [])
            indices.append(round_index)
        if not round_indices_by_operation:
            return "empty"
        parts = []
        for operation_id in sorted(round_indices_by_operation, key=str):
            round_indices = round_indices_by_operation[operation_id]
            ordered = sorted(round_indices)
            ranges = _round_ranges_text(ordered)
            count = len(round_indices)
            parts.append(f"op {operation_id} rounds {ranges} ({count})")
        return "; ".join(parts)

    # ---- private

    def _check_costs_have_a_clock(self) -> None:
        charged = self.settings.write_cycles + self.settings.read_cycles
        if charged > 0 and self.settings.clock is None:
            raise ValueError("charged weak_syndrome_buffer costs need a clock")

    def _access_completion_tick(self, cycles: int) -> int:
        """The access's end, cycles after now's edge; a zero cost is now."""
        now = self.engine.now
        if cycles == 0:
            return now
        return self.settings.clock.edge(cycles, now)

    def _refuse_rounds_left(self) -> None:
        held = list(self.round_by_key)
        capacity = self.settings.bits
        widest_holder = self._widest_live_hold()
        if capacity is None or widest_holder is None:
            raise RuntimeError(
                f"the syndrome buffer still holds rounds {held} at the end"
            )
        round_keys = self.holds.round_keys_of(widest_holder)
        kept_bits = self._stored_bits_of(round_keys)
        never_stored = [
            round_key
            for round_key in round_keys
            if round_key not in self.round_by_key
        ]
        raise RuntimeError(
            f"the syndrome buffer still holds rounds {held} at the end, "
            f"{self.occupied_bits} of its {capacity} bits: its widest live "
            f"hold {widest_holder!r} keeps {kept_bits} bits of them and "
            f"waits for {never_stored}, which never found room; a bounded "
            "store must hold its widest hold's rounds at once"
        )

    def _widest_live_hold(self):
        """The live holder whose stored rounds take the most bits, or None."""
        widest_holder = None
        widest_bits = 0
        for holder, record in self.holds.record_by_holder.items():
            kept_bits = self._stored_bits_of(record.round_keys)
            if kept_bits > widest_bits:
                widest_holder = holder
                widest_bits = kept_bits
        return widest_holder

    def _stored_bits_of(self, round_keys) -> int:
        kept_bits = 0
        for round_key in round_keys:
            stored = self.round_by_key.get(round_key)
            if stored is not None:
                kept_bits += stored.held_bits
        return kept_bits

    def _open(self, operation_id) -> None:
        is_open = self.operations.get(operation_id)
        if is_open is False:
            raise RuntimeError("closed operation identities cannot be reused")
        self.operations[operation_id] = True

    def _stored_rounds_of(self, operation_id) -> list:
        """The operation's stored rounds, in stable identity order."""
        stored = []
        for round_key in self.round_by_key:
            if identity_records.same_stable_identity(
                round_key[0], operation_id
            ):
                stored.append(round_key)
        return sorted(stored, key=identity_records.stable_identity_bytes)

    def _open_operation_ids(self) -> set:
        open_ids = set()
        for operation_id, is_open in self.operations.items():
            if is_open:
                open_ids.add(operation_id)
        return open_ids

    def _hold_record(self, holder, round_keys) -> round_holds.HoldRecord:
        """The record of a hold; a hold may name rounds not yet written.

        A hold keeps open the operations its rounds belong to and the
        ones its token names beyond them, which the token answers.
        """
        unique = dict.fromkeys(round_keys)
        keys = tuple(unique)
        references = set()
        for round_key in keys:
            references.add(round_key[0])
        for operation_id in holder.referenced_operation_ids():
            references.add(operation_id)
        for operation_id in references:
            self._open(operation_id)
        referenced = frozenset(references)
        return round_holds.HoldRecord(keys, referenced)

    def _free_stored(self, round_keys) -> None:
        for round_key in round_keys:
            if round_key in self.round_by_key:
                self._free_round(round_key)

    def _refuse_unsized_round(self, capacity: int) -> None:
        """A bound is measured against a size, so a round must state one."""
        raise RuntimeError(
            f"the syndrome buffer holds {capacity} bits and the round "
            "states no size; a bounded syndrome buffer needs sized rounds"
        )

    def _refuse_round_wider_than_store(
        self, round_key, bits: int, capacity: int
    ) -> None:
        """A round wider than the whole store waits for room forever.

        gem5 refuses a message wider than the block that must hold it
        (src/mem/ruby/network/Network.cc:64-65, "data message size >
        cache line size"); round widths come from the source as it runs,
        so the refusal comes at the first ask rather than at the load.
        """
        raise RuntimeError(
            f"the syndrome buffer holds {capacity} bits and round "
            f"{round_key!r} states {bits}: no round leaving it makes room, "
            "so a bounded syndrome buffer holds at least its widest round"
        )

    def _free_round(self, round_key) -> None:
        self.round_by_key.pop(round_key)
        self.trace.round_released.fire(round_key)
        if self.detection_events is not None:
            self.detection_events.retire_round(round_key)
        if self.held_rounds is not None:
            self.held_rounds.retry()


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the store reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (gem5 src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through store.trace.
    """

    round_stored: trace_source.TraceSource = trace_source.new_source()
    round_published: trace_source.TraceSource = trace_source.new_source()
    round_released: trace_source.TraceSource = trace_source.new_source()
    hold_registered: trace_source.TraceSource = trace_source.new_source()
    hold_transferred: trace_source.TraceSource = trace_source.new_source()
    hold_released: trace_source.TraceSource = trace_source.new_source()
    access_served: trace_source.TraceSource = trace_source.new_source()


class _StoredRound:
    """One stored round: its packet, its bits, and once published its tick."""

    def __init__(
        self, packet: round_records.SyndromeRoundPacket, held_bits: int
    ) -> None:
        self.packet = packet
        self.held_bits = held_bits
        self.publication_tick: Optional[int] = None


def _round_ranges_text(sorted_round_indices: list) -> str:
    """[1, 2, 3, 7, 8] -> "1..3, 7..8"."""
    ranges = []
    range_start = sorted_round_indices[0]
    previous = range_start
    for round_index in sorted_round_indices[1:]:
        if round_index != previous + 1:
            ranges.append((range_start, previous))
            range_start = round_index
        previous = round_index
    ranges.append((range_start, previous))
    texts = [_range_text(low, high) for low, high in ranges]
    return ", ".join(texts)


def _range_text(low: int, high: int) -> str:
    if low == high:
        return f"{low}"
    return f"{low}..{high}"
