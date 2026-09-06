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
        self.copies = _Counts()
        self.moves = _Counts()
        self.references = _Counts()
        self.holds_registered = 0
        self.holds_transferred = 0
        self.holds_released = 0
        self.copies_by_path: dict[str, _Counts] = {}
        self.moves_by_path: dict[str, _Counts] = {}
        self.rounds_seen: set = set()

    def copy_made(self, key, bits, source_name: str, target_name: str) -> None:
        """One structure duplicated the bits into another."""
        path = f"{source_name} -> {target_name}"
        rounds = _rounds(key)
        self._add(self.copies, self.copies_by_path, path, bits, rounds)

    def transfer_delivered(self, record) -> None:
        """One move landed on its link."""
        bits = record.transfer.payload_bits
        rounds = _attributed_rounds(record.attribution)
        self._add(
            self.moves, self.moves_by_path, record.path.value, bits, rounds
        )

    def hold_registered(self, holder, round_keys) -> None:
        """One token referenced the rounds where they already sit."""
        del holder
        self.references.events += 1
        self.references.rounds += len(round_keys)
        self.holds_registered += 1

    def hold_transferred(self, old_holder, new_holder) -> None:
        """A live reference moved to a new token, copying nothing."""
        del old_holder
        del new_holder
        self.holds_transferred += 1

    def hold_released(self, holder) -> None:
        """One reference ended."""
        del holder
        self.holds_released += 1

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
        holds = (
            self.holds_registered + self.holds_transferred + self.holds_released
        )
        return {
            "rounds": self.rounds,
            "copies": self.copies.events,
            "copied_rounds": self.copies.rounds,
            "copy_bits": self.copies.bits,
            "references": self.references.events,
            "referenced_rounds": self.references.rounds,
            "hold_events": holds,
            "moves": self.moves.events,
            "moved_rounds": self.moves.rounds,
            "move_bits": self.moves.bits,
            "copies_by_path": _as_rows(self.copies_by_path),
            "moves_by_path": _as_rows(self.moves_by_path),
        }

    def _add(
        self, total: _Counts, by_path: dict, path: str, bits, rounds: int
    ) -> None:
        row = by_path.get(path)
        if row is None:
            row = _Counts()
            by_path[path] = row
        for counts in (total, row):
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


def _as_rows(by_path: dict) -> dict:
    """One row per path, in path order."""
    rows = {}
    for path in sorted(by_path):
        counts = by_path[path]
        rows[path] = {
            "events": counts.events,
            "rounds": counts.rounds,
            "bits": counts.bits,
        }
    return rows
