"""Where the controller writes each round: its stream's store, and copies.

A round goes to the store of the entry pool its stream is routed to,
and to every other entry store that holds it, since a window of that
store's pool reads it: a merge's partner region, or the first rounds of
a successor another pool decodes. Real systems route raw syndromes at
write time to each decoder that needs them: DecoNet's leaf router is
"statically configure[d] ... to route measurement data to the
appropriate decoder instances" (Liyanage et al. 2504.11805 Sec. VI-B.2),
and each of Lin et al.'s overlapping windows "receives information from
the control system" (2403.01353 Sec. VI). No source has a decoder fetch
a neighbour's raw syndromes when it decodes; decoders exchange only
boundaries after a commit (windows/window_boundaries.py).

The holds decide: a store holds a round from the moment its reader is
known, before the round is stored (syndrome_buffer.py is_round_held),
so who reads what has one owner, the retention, and no table here.

A round copied to several stores leaves each when its readers there
are done, at different times. The seats form a round's events until it
leaves the last of them, so each store tells the destinations, which
know how many stores the round went to, and the seats hear it once.
"""

import dataclasses
from typing import Any

import decsim.ports as ports
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records


@dataclasses.dataclass(frozen=True)
class StoreSide:
    """One entry store as the controller writes it.

    receiver is its receiving end; write_path the controller's hop into
    it; is_landed whether a round lands there over that hop, rather than
    being written through the transmitter.
    """

    name: str
    store: ports.RetainedRounds
    receiver: ports.SyndromeRoundReceiver
    write_path: transfer_records.LinkPath
    is_landed: bool


class RoundDestinations:
    """Each round's stores, its own first, and how far each stream went."""

    # which entry pool decodes a stream, so which store is a round's own
    router = ports.Port(ports.StreamRouting)
    # the run's former seats, told when a round left every store it went
    # to; bound only with several entry stores, the only ones that tell
    # these destinations rather than the seats
    detection_events = ports.Port(ports.RoundRetirement)

    def __init__(self, side_by_pool: dict) -> None:
        self.side_by_pool = side_by_pool
        self.sides = _distinct_sides(side_by_pool)
        self.last_written_by_stream: dict = {}
        # with several entry stores, how many still keep each round
        self.stores_left_by_round: dict = {}

    def stores_for(self, packed: round_records.PackedRound) -> tuple:
        """The round's own store side first, then each that holds it.

        One store takes every round of a machine whose entry pools share
        it, so nothing there is routed or copied.
        """
        if len(self.sides) == 1:
            return self.sides
        stream_id = _routed_stream(packed)
        pool = self.router.pool_of_stream(stream_id)
        own = self.side_by_pool[pool]
        sides = [own]
        for side in self.sides:
            if side is own:
                continue
            if side.store.is_round_held(packed.round_key):
                sides.append(side)
        return tuple(sides)

    def note_written(
        self, packed: round_records.PackedRound, store_count: int
    ) -> None:
        """The round left for store_count stores; a later reader gets no copy.

        One store tells the seats itself, so nothing is counted for it.
        """
        stream_id, round_index = packed.round_key
        last_written = self.last_written_by_stream.get(stream_id, 0)
        self.last_written_by_stream[stream_id] = max(last_written, round_index)
        if len(self.sides) == 1:
            return
        self.stores_left_by_round[packed.round_key] = store_count

    def retire_round(self, round_key: tuple) -> None:
        """One store let the round go; the seats hear when the last does."""
        stores_left = self.stores_left_by_round.get(round_key)
        if stores_left is None:
            raise RuntimeError(
                f"round {round_key} left an entry store it was never written to"
            )
        if stores_left > 1:
            self.stores_left_by_round[round_key] = stores_left - 1
            return
        del self.stores_left_by_round[round_key]
        self.detection_events.retire_round(round_key)

    def last_round_written(
        self,
        stream_id: Any,  # an opaque identity
    ) -> int:
        """The stream's highest round written to any store; 0 for none."""
        return self.last_written_by_stream.get(stream_id, 0)


def _routed_stream(packed: round_records.PackedRound) -> Any:
    """The stream whose pool's store is the round's own.

    A window-input round is its own stream's; a feedback-memory round
    times the windows of the stream it stands in for.
    """
    window_input = round_records.SyndromePacketRouteKind.WINDOW_INPUT
    if packed.route.kind is window_input:
        return packed.round_key[0]
    return packed.route.source_stream_id


def _distinct_sides(side_by_pool: dict) -> tuple:
    """Each store's side once, in the order its first pool names it."""
    sides = []
    for side in side_by_pool.values():
        if side not in sides:
            sides.append(side)
    return tuple(sides)
