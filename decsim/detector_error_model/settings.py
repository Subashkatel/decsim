"""The detection events' settings: where the former sits, and its cost.

The published seats are four: the controller's workstation (Google
2408.13687), the decoder chip's input path (Maurer 2510.21600), inside
the decoder (Caune 2410.05202, LILLIPUT 2108.06569, cudaqx) and offline
(Deltakit through Stim's compile_m2d_converter).
"""

import dataclasses
from typing import Optional

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class DetectionEventSettings:
    """How a round's detection events are formed.

    formed_at lists the seats, each once, by the part each is: the
    controller, a store or a pool; every path a round takes to a pool
    crosses exactly one (build/readout.py). One conversion costs
    latency_cycles, plus cycles_per_round for each further round a seat
    forms together, on the clock of the forming logic: a pipelined stage
    takes a round a cycle after its fixed latency (Yang et al. 2605.04892:
    syndrome calculation "at 20 ns (5 FPGA clock cycles)", "fully pipelined
    operation"). No source publishes a controller-side or buffer-side
    figure, so both costs default to zero. clock None is the machine's.
    """

    formed_at: tuple = ("controller",)
    clock: Optional[config.Clock] = None
    latency_cycles: int = 0
    cycles_per_round: int = 0

    def __post_init__(self) -> None:
        config.check_cycles(
            "detection_events.latency_cycles", self.latency_cycles
        )
        config.check_cycles(
            "detection_events.cycles_per_round", self.cycles_per_round
        )
        _check_seats(self.formed_at)

    def landed_seat(self, landed_seats: tuple) -> Optional[str]:
        """The seat among landed_seats that forms; None for none.

        landed_seats are the stores a round lands in and the pools that
        read them (build/layout.py Layout.landed_seats): the rounds of an
        escalated region reach them from a store another pool already
        read, so they read the raw round before the region too. Every
        path crosses exactly one forming seat (formed_at), so at most one
        of them forms.
        """
        for seat in self.formed_at:
            if seat in landed_seats:
                return seat
        return None

    def cycles_for(self, round_count: int) -> int:
        """The cycles of forming round_count rounds together; none for none."""
        if round_count == 0:
            return 0
        after_the_first = round_count - 1
        return self.latency_cycles + self.cycles_per_round * after_the_first


def _check_seats(formed_at: tuple) -> None:
    """No seat is named twice; build/readout.py checks each one's path."""
    if len(set(formed_at)) != len(formed_at):
        raise ValueError(
            f"detection_events.formed_at names a seat twice: {list(formed_at)}"
        )
