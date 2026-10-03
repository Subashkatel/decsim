"""The yaml's `detection_events` section: where the former sits, and its cost.

A detection event is a parity of raw measurement outcomes
(detector_formation.py), so its value is the same wherever the machine
forms it; the seat moves the width every hop after it carries, the
state the seat holds and the clock the conversion is charged on. The
published seats are four: the controller's workstation (Google
2408.13687 lines 472-477), the decoder chip's input path (Maurer
2510.21600 lines 234-236), inside the decoder (Caune 2410.05202 lines
1252-1255, LILLIPUT 2108.06569 lines 500-509, cudaqx
libs/qec/lib/decoder.cpp:426-432) and offline (Deltakit through Stim's
compile_m2d_converter). This section names the seats on decsim's path
that do it.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config
import decsim.tables as tables

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

_KEYS = ("formed_at", "clock", "latency_cycles", "cycles_per_round")


@dataclasses.dataclass(frozen=True)
class DetectionEventSettings:
    """The seats that form a round's detection events, and what it costs.

    formed_at lists the seats, each named once; every path a round takes
    to a decoder crosses exactly one of them, which the build checks
    against the run's paths (build/readout.py). One conversion
    costs latency_cycles, plus cycles_per_round for every round after
    the first when a seat forms several rounds together, on clock, the
    clock of the logic that forms them: a pipelined stage takes a round
    a cycle after its fixed latency (Yang et al. 2605.04892 lines
    1273-1275, "The total latency of the preprocessing stage for
    syndrome calculation ... at 20 ns (5 FPGA clock cycles)", with "All
    variables ... in FPGA registers, enabling fully pipelined
    operation"). A seat that forms one round at a time pays the latency
    for each. No source publishes a controller-side or buffer-side
    figure, so both costs are zero by default. clock None is the
    machine's clock.
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

    @classmethod
    def from_yaml(
        cls, section: Mapping, clocks: config.ClockSettings
    ) -> "DetectionEventSettings":
        """The section's seats and cost, on its clock when it names one."""
        tables.refuse_unknown_keys("detection_events", section, _KEYS)
        formed_at = _formed_at(section)
        clock = None
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        latency_cycles = section.get("latency_cycles", 0)
        cycles_per_round = section.get("cycles_per_round", 0)
        return cls(
            formed_at=formed_at,
            clock=clock,
            latency_cycles=latency_cycles,
            cycles_per_round=cycles_per_round,
        )

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


def _formed_at(section: Mapping) -> tuple:
    """The formed_at list as a tuple; a single seat is refused as a list."""
    formed_at = section.get("formed_at", ["controller"])
    if not isinstance(formed_at, list):
        raise ValueError(
            f"detection_events.formed_at is a list of seats, as in "
            f"[{formed_at}]; got {formed_at!r}"
        )
    return tuple(formed_at)


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


# Yang et al.'s syndrome preprocessing, fully pipelined in FPGA registers
# at 20 ns, 5 cycles of their 250 MHz clock (2605.04892 lines 1273-1275,
# Table I line 1063). Yang form the syndromes in the CFM, the FPGA that
# also decodes them (lines 199-200); decsim forms them at the
# controller, the record's default seat, by its own choice.
_YANG_CLOCK = config.Clock.from_megahertz(250.0)
YANG_PREPROCESSING = DetectionEventSettings(
    clock=_YANG_CLOCK,
    latency_cycles=5,  # lines 1274-1275
    cycles_per_round=1,  # a round a cycle, fully pipelined, line 1273
)
