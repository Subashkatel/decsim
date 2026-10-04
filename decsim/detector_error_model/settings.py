"""The detection events' settings: where the former sits, and its cost.

The published seats are four: the controller's workstation (Google
2408.13687), the decoder chip's input path (Maurer 2510.21600), inside
the decoder (Caune 2410.05202, LILLIPUT 2108.06569, cudaqx) and offline
(Deltakit through Stim's compile_m2d_converter).
"""

import dataclasses
from typing import Optional

import decsim.config as config

# The points on the path from the controller to a decoder where the
# former may sit, in path order: the controller's assembler, the two
# stores' receiving ends, and the two decoder units' input.
SEATS = (
    "controller",
    "weak_syndrome_buffer",
    "strong_syndrome_buffer",
    "weak_decoder",
    "strong_decoder",
)
# The seats inside a decoder unit: they form a job's rounds as they land
# in the unit's memory, which was written the raw round.
DECODER_SEATS = ("weak_decoder", "strong_decoder")
# The paths a round takes from the controller to a decoder, as seats in
# path order: to the weak decoder; from the weak syndrome buffer to the
# strong decoder when a switching run escalates a region
# (strong_syndrome_round_receiver.py, weak_decoder_to_strong_decoder);
# and straight to the strong decoder when the strong tier decodes the
# plan's windows (controller_to_strong_buffer).
WEAK_PATH = ("controller", "weak_syndrome_buffer", "weak_decoder")
ESCALATION_PATH = (
    "controller",
    "weak_syndrome_buffer",
    "strong_syndrome_buffer",
    "strong_decoder",
)
STRONG_PATH = ("controller", "strong_syndrome_buffer", "strong_decoder")
# The seats past the weak syndrome buffer on the escalation path: the
# rounds of an escalated region reach them from a store the weak side
# already read, so they read the raw round before the region too.
STRONG_SIDE_SEATS = ("strong_syndrome_buffer", "strong_decoder")


@dataclasses.dataclass(frozen=True)
class DetectionEventSettings:
    """The seats that form a round's detection events, and what it costs.

    formed_at lists the seats, each once; every path a round takes to a
    decoder crosses exactly one (build/readout.py). One conversion costs
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

    def strong_side_seat(self) -> Optional[str]:
        """The seat past the weak syndrome buffer that forms; None for none.

        Every path crosses exactly one forming seat (formed_at), so the
        escalation path has at most one past the weak syndrome buffer.
        """
        for seat in self.formed_at:
            if seat in STRONG_SIDE_SEATS:
                return seat
        return None

    def cycles_for(self, round_count: int) -> int:
        """The cycles of forming round_count rounds together; none for none."""
        if round_count == 0:
            return 0
        after_the_first = round_count - 1
        return self.latency_cycles + self.cycles_per_round * after_the_first


def _check_seats(formed_at: tuple) -> None:
    """Every seat is one of SEATS, and none is named twice."""
    for seat in formed_at:
        if seat not in SEATS:
            raise ValueError(
                f"detection_events.formed_at names {seat!r}, which is not "
                f"a seat; the seats are {list(SEATS)}"
            )
    if len(set(formed_at)) != len(formed_at):
        raise ValueError(
            f"detection_events.formed_at names a seat twice: {list(formed_at)}"
        )
