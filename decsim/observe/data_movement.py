"""How often a run copied bits, referenced them and moved them.

The counters docs/rewrite/notes/data_path.md section 8 asks for, so a
study reads the data path without the trace file. gem5's vocabulary
(section 2 of that note): a copy duplicates bits into a structure the
receiver owns (mem/cache/cache_blk.hh 97-104), a reference is a handle
to bits that stay where they are (mem/packet.hh 1163-1171), a move
crosses a link (dev/dma_device.cc 194-213).

A listener on copy_made on every component that copies, on the fabric's
transfer_delivered for the moves, and on the stores' hold_registered
for the references; it counts and keeps no bits.
"""

import dataclasses


@dataclasses.dataclass
class _Counts:
    """One structure's tally."""

    events: int = 0
    bits: int = 0


class DataMovement:
    """Copies, references and moves, in total and per named path."""

    def __init__(self) -> None:
        self.copies = _Counts()
        self.moves = _Counts()
        self.references = 0
        self.copies_by_path: dict[str, _Counts] = {}
        self.moves_by_path: dict[str, _Counts] = {}
        self.rounds_seen: set = set()

    def copy_made(self, key, bits, source_name: str, target_name: str) -> None:
        """One structure duplicated the bits into another."""
        del key
        path = f"{source_name} -> {target_name}"
        self._add(self.copies, self.copies_by_path, path, bits)

    def transfer_delivered(self, record) -> None:
        """One move landed on its link."""
        bits = record.transfer.payload_bits
        self._add(self.moves, self.moves_by_path, record.path.value, bits)

    def hold_registered(self, holder, round_keys) -> None:
        """One token referenced the rounds where they already sit."""
        del holder
        del round_keys
        self.references += 1

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
        return self.copies.events / len(self.rounds_seen)

    def json_value(self) -> dict:
        """The counters as the RunResult carries them."""
        return {
            "rounds": self.rounds,
            "copies": self.copies.events,
            "copy_bits": self.copies.bits,
            "references": self.references,
            "moves": self.moves.events,
            "move_bits": self.moves.bits,
            "copies_by_path": _as_rows(self.copies_by_path),
            "moves_by_path": _as_rows(self.moves_by_path),
        }

    def _add(self, total: _Counts, by_path: dict, path: str, bits) -> None:
        total.events += 1
        row = by_path.get(path)
        if row is None:
            row = _Counts()
            by_path[path] = row
        row.events += 1
        if bits is None:
            return
        total.bits += bits
        row.bits += bits


def _as_rows(by_path: dict) -> dict:
    """One (events, bits) pair per path, in path order."""
    rows = {}
    for path in sorted(by_path):
        counts = by_path[path]
        rows[path] = {"events": counts.events, "bits": counts.bits}
    return rows
