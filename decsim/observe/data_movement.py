"""How often a run copied bits, referenced them and moved them.

The counters of the data path's hop table, so a
study reads the data path without the trace file. gem5's vocabulary
(section 2 of that note): a copy duplicates bits into a structure the
receiver owns (mem/cache/cache_blk.hh 97-104), a reference is a handle
to bits that stay where they are (mem/packet.hh 1163-1171), a move
crosses a link (dev/dma_device.cc 194-213).

A listener on copy_made on every component that copies, on the fabric's
transfer_delivered for the moves, and on the stores' hold_registered
for the references; it counts and keeps no bits.

The counts are also grouped by the memory class the hop crosses, because
the classical sources make the class the cost and not the count: a DRAM
access is "a couple of orders-of-magnitude higher than the cost of an
internal cache access" (Horowitz, ISSCC 2014 lines 232-247), and on an
accelerator "memory accesses have a cost that is a function of the size
of the memory being accessed" (Dally, CACM 2020 lines 231-234), so a
copy into a register and a copy across a cryostat link must not be
summed. The two tables below are decsim's placement of this machine's
structures and links, read off the sources named on each row; they are
a report-side grouping and no component reads them.
"""

import dataclasses
import enum


class MemoryClass(enum.Enum):
    """The memory a hop crosses, and what it costs to cross it.

    ON_CHIP is a register or an SRAM inside one chip; ON_BOARD is a
    memory two chips of one board share; OFF_BOARD is a link between
    boards or out of the cryostat. UNCLASSIFIED is a structure or a path
    no table row names, so a grouped report still sums to the run's
    total.
    """

    ON_CHIP = "on_chip"
    ON_BOARD = "on_board"
    OFF_BOARD = "off_board"
    UNCLASSIFIED = "unclassified"


# the order the grouped report lists the classes in, cheapest first
_CLASS_ORDER = (
    MemoryClass.ON_CHIP,
    MemoryClass.ON_BOARD,
    MemoryClass.OFF_BOARD,
    MemoryClass.UNCLASSIFIED,
)


# What each link crosses. The readout leaves the cryostat: measurement
# signals "are classified into bits then transmitted to a specialized
# workstation via low-latency Ethernet" (Google 2408.13687 lines
# 471-474), and the instruction returns the same way. The controller
# writes its rounds into memory it shares with the tier beside it, "a
# shared memory buffer" (2408.13687 lines 475-477) and "the decoder
# sequencer's memory" (Caune et al. 2410.05202 lines 1243-1245), and
# that tier reads its input from the same board because its decoder sits
# in the sequencer's own FPGA (Chen 2605.30765 lines 1605-1608). The
# frame is the controller's, so the correction that reaches it stays on
# that board and the conditional release never leaves the chip. The
# strong tier is a separate machine, room side in this tree, given its
# assigned data (Toshio 2510.25222 lines 1248-1250), so the write into
# its store and its answer cross boards while its own store-to-decoder
# hop does not. One decoder's boundary reaches another through the
# shared memory the processing elements write, "implemented using
# registers" (Helios 2301.08419 lines 632-640).
MEMORY_CLASS_BY_LINK_PATH = {
    "qpu_to_controller": MemoryClass.OFF_BOARD,
    "controller_to_qpu": MemoryClass.OFF_BOARD,
    "controller_to_weak_buffer": MemoryClass.ON_BOARD,
    "controller_to_strong_buffer": MemoryClass.OFF_BOARD,
    "weak_buffer_to_weak_decoder": MemoryClass.ON_BOARD,
    "strong_buffer_to_strong_decoder": MemoryClass.ON_BOARD,
    "weak_decoder_to_strong_decoder": MemoryClass.OFF_BOARD,
    "weak_decoder_to_frame": MemoryClass.ON_BOARD,
    "strong_decoder_to_frame": MemoryClass.OFF_BOARD,
    "decoder_to_decoder": MemoryClass.ON_CHIP,
    "frame_to_controller": MemoryClass.ON_CHIP,
}

# Where each copied-into structure lives. A copy is billed where it
# lands, the way gem5 fills the destination block
# (mem/cache/cache_blk.hh 97-104), so the target names the class. The
# controller's intake and its packing workspace are registers of the
# control FPGA: each round's buffer is pushed to "the sequencer's data
# stack" and combined there (Caune et al. 2410.05202 lines 1252-1254). A
# store is memory two chips share (2408.13687 lines 475-477). A unit's
# own input memory and the masked view of it are the decoder's storage
# elements, which its processing elements "can directly access" on chip
# (AFS 2001.06598 lines 528-531; Collision Clustering's Init unit loads
# the syndrome into them, 2309.05558 lines 268-271).
MEMORY_CLASS_BY_STRUCTURE = {
    "controller intake": MemoryClass.ON_CHIP,
    "controller assembler": MemoryClass.ON_CHIP,
    "masked view": MemoryClass.ON_CHIP,
}
_STORE_PREFIX = "Buffer "
_UNIT_PREFIX = "unit "


def memory_class_of_structure(name: str) -> MemoryClass:
    """The class of the structure a copy landed in, by its reported name.

    A name no row and no prefix names is unclassified rather than
    guessed, so the grouped rows still sum to the run's total.
    """
    named = MEMORY_CLASS_BY_STRUCTURE.get(name)
    if named is not None:
        return named
    if name.startswith(_STORE_PREFIX):
        return MemoryClass.ON_BOARD
    if name.startswith(_UNIT_PREFIX):
        return MemoryClass.ON_CHIP
    return MemoryClass.UNCLASSIFIED


def memory_class_of_link_path(path: str) -> MemoryClass:
    """The class of the memory one link's move crosses."""
    return MEMORY_CLASS_BY_LINK_PATH.get(path, MemoryClass.UNCLASSIFIED)


@dataclasses.dataclass
class _Counts:
    """One structure's tally: the events, the rounds they carried, bits.

    data_path.md's hop table counts per round, so one job's input copy
    of six rounds is one event and six rounds; a study that prices the
    hop reads the events, one that counts the table's hops reads the
    rounds.
    """

    events: int = 0
    rounds: int = 0
    bits: int = 0


class DataMovement:
    """Copies, references and moves, in total and per named path."""

    def __init__(self) -> None:
        self.copies = _PathCounts()
        self.moves = _PathCounts()
        self.references = _Counts()
        self.holds = _HoldTallies()
        self.rounds_seen: set = set()

    def copy_made(self, key, bits, source_name: str, target_name: str) -> None:
        """One structure duplicated the bits into another."""
        path = f"{source_name} -> {target_name}"
        rounds = _rounds(key)
        memory_class = memory_class_of_structure(target_name)
        self._add(self.copies, path, bits, rounds, memory_class)

    def transfer_delivered(self, record) -> None:
        """One move landed on its link."""
        bits = record.transfer.payload_bits
        rounds = _attributed_rounds(record.attribution)
        path = record.path.value
        memory_class = memory_class_of_link_path(path)
        self._add(self.moves, path, bits, rounds, memory_class)

    def hold_registered(self, holder, round_keys) -> None:
        """One token referenced the rounds where they already sit."""
        del holder
        self.references.events += 1
        self.references.rounds += len(round_keys)
        self.holds.registered += 1

    def hold_transferred(self, old_holder, new_holder) -> None:
        """A live reference moved to a new token, copying nothing."""
        del old_holder
        del new_holder
        self.holds.transferred += 1

    def hold_released(self, holder) -> None:
        """One reference ended."""
        del holder
        self.holds.released += 1

    def round_emitted(self, readout) -> None:
        """One more round exists, so a per-round rate has a denominator."""
        self.rounds_seen.add((readout.operation_id, readout.round_index))

    @property
    def rounds(self) -> int:
        """The rounds the QPU emitted."""
        return len(self.rounds_seen)

    @property
    def copies_per_round(self) -> float:
        """Copies over rounds; zero when the run emitted none."""
        if not self.rounds_seen:
            return 0.0
        return self.copies.total.events / len(self.rounds_seen)

    def json_value(self) -> dict:
        """The counters as the RunResult carries them."""
        holds = self.holds.total()
        return {
            "rounds": self.rounds,
            "copies": self.copies.total.events,
            "copied_rounds": self.copies.total.rounds,
            "copy_bits": self.copies.total.bits,
            "references": self.references.events,
            "referenced_rounds": self.references.rounds,
            "hold_events": holds,
            "moves": self.moves.total.events,
            "moved_rounds": self.moves.total.rounds,
            "move_bits": self.moves.total.bits,
            "copies_by_path": _as_rows(self.copies.by_path),
            "moves_by_path": _as_rows(self.moves.by_path),
            "copies_by_memory_class": _as_class_rows(self.copies.by_class),
            "moves_by_memory_class": _as_class_rows(self.moves.by_class),
        }

    def _add(
        self,
        kind: "_PathCounts",
        path: str,
        bits,
        rounds: int,
        memory_class: MemoryClass,
    ) -> None:
        row = _row_of(kind.by_path, path)
        class_row = _row_of(kind.by_class, memory_class)
        for counts in (kind.total, row, class_row):
            counts.events += 1
            counts.rounds += rounds
            if bits is not None:
                counts.bits += bits


def _rounds(key) -> int:
    """The rounds one copy carried: a job's whole input, or one round."""
    decoder_input = getattr(key, "decoder_input", None)
    if decoder_input is not None:
        return len(decoder_input.rounds)
    return 1


def _attributed_rounds(attribution) -> int:
    """The rounds one move carried, from its attribution's range."""
    if attribution.first_round is None:
        return 0
    return attribution.last_round - attribution.first_round + 1


def _row_of(rows: dict, key):
    """The tally kept under the key, opened on its first event."""
    counts = rows.get(key)
    if counts is None:
        counts = _Counts()
        rows[key] = counts
    return counts


def _as_rows(by_path: dict) -> dict:
    """One row per path, in path order."""
    rows = {}
    for path in sorted(by_path):
        rows[path] = _as_row(by_path[path])
    return rows


def _as_class_rows(by_class: dict) -> dict:
    """One row per memory class the run crossed, in cost order."""
    rows = {}
    for memory_class in _CLASS_ORDER:
        counts = by_class.get(memory_class)
        if counts is None:
            continue
        rows[memory_class.value] = _as_row(counts)
    return rows


def _as_row(counts: "_Counts") -> dict:
    """One tally as the report carries it."""
    return {
        "events": counts.events,
        "rounds": counts.rounds,
        "bits": counts.bits,
    }


@dataclasses.dataclass
class _PathCounts:
    """One kind's tally: the run's total, one row per path, one per class."""

    total: _Counts = dataclasses.field(default_factory=_Counts)
    by_path: dict = dataclasses.field(default_factory=dict)
    by_class: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _HoldTallies:
    """How often a reference was registered, handed on and ended."""

    registered: int = 0
    transferred: int = 0
    released: int = 0

    def total(self) -> int:
        """Every hold event of the run."""
        return self.registered + self.transferred + self.released
