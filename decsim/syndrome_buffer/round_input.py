"""A round store's incoming port: the store handles what lands in it.

Whoever receives a transfer handles its landing, and the end a round
lands at is the store that keeps it. gem5's requesting port hands the
packet to the peer's own receive method instead of writing the peer's
state itself (`tmp/resources/gem5/src/mem/port.hh:603-614`,
RequestPort::sendTimingReq, whose `src/mem/protocol/timing.cc:49-53`
calls `peer->recvTimingReq(pkt)`, the method every receiving port
declares at `src/mem/protocol/timing.hh:170-172`), and it bills a
transfer to the port it left by (`packet.hh:424-431`). OMNeT++ hands the
message's ownership to the destination module before that module's
handleMessage runs (`tmp/resources/omnetpp/src/sim/csimplemodule.cc`,
doMessageEvent at 777-799, take(msg) at 783, omnetpp-6.1.0). And ns-3's
point-to-point channel schedules the receive on the destination device
(`tmp/resources/l5_buffers/ns3-point-to-point/point-to-point-channel.cc`
at 88-92), where that receive is the device's own method
(`point-to-point-net-device.cc:324`).

Two calls of the controller's transmitter arrive here, both about
Buffer 0. A packed round lands over controller_to_weak_buffer: its
record was written before the wire was used, so that the store could
refuse for room first, the way gem5's queue reserves an entry before the
send (`src/mem/cache/queue.hh:150-152`); this port stamps the
publication tick on that record and announces the published round to the
window manager. And a timing-only feedback-memory round is asked for: it
leaves the store, so the store's own outgoing port executes that send
(round_output.py).
"""

import dataclasses
from typing import Callable

import decsim.records.rounds as round_records
import decsim.trace_source as trace_source


class RoundStoreInput:
    """One store's port toward the controller, bound once by the root.

    Trace source: round_event(RoundEvent) with kind PUBLISHED.
    """

    def __init__(self, engine, store, output, windows) -> None:
        self.engine = engine
        self.store = store
        # the store's outgoing port, which sends what leaves the store
        self.output = output
        self.windows = windows
        self.trace = _TraceSources()

    def receive_round(self, packed: round_records.PackedRound) -> None:
        """Take one landed round: publish it, then tell the windows.

        The publication tick is this end's own clock at the landing, and
        the announcement follows the stamp, so the window manager never
        hears of a round before the store's record says it is readable
        (validation buffer_contract.md, Buffer 0).
        """
        publication_tick = self.engine.now
        self.store.mark_publication_tick(packed.round_key, publication_tick)
        self._fire_published(packed)
        self.windows.accept_window_input(packed.packet)

    def send_memory_round(
        self,
        packed: round_records.PackedRound,
        on_delivered: Callable[[], None],
    ) -> None:
        """Ask the store to send one timing-only round to its decoder."""
        self.output.send_memory_round(packed, on_delivered)

    def _fire_published(self, packed: round_records.PackedRound) -> None:
        operation_id, round_index = packed.round_key
        event = round_records.RoundEvent.of(
            "PUBLISHED",
            self.engine.now,
            operation_id,
            round_index,
            packed.route,
        )
        self.trace.round_event.fire(event)


@dataclasses.dataclass(frozen=True)
class _TraceSources:
    """Every event the store's incoming port reports, as one member.

    gem5 groups a component's statistics into one nested Group member
    (tmp/resources/gem5/src/base/stats/group.hh:60-92) rather than one
    member per counter; a component's events are the same shape, so a
    listener reaches all of them through one name.
    """

    round_event: trace_source.TraceSource = trace_source.new_source()
