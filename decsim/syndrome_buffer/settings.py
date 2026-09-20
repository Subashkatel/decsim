"""A syndrome buffer's capacity and access costs on its named clock."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class SyndromeBufferSettings:
    """The yaml's `weak_syndrome_buffer` and `strong_syndrome_buffer` sections.

    Table row (SYNDROME_BUFFERS, syndrome_buffer.py): syndrome_buffer.
    rounds bounds the store; None is unbounded. A full store makes the
    controller hold the finished round and write it in order once a slot frees,
    the backpressure real systems apply to their source (Caune et al.
    2410.05202: the sequencer stalls on the decoder's status register).
    The weak syndrome buffer charges write_cycles before publication and
    read_cycles once per primary window before assembling its input. The stages
    follow gem5's frontend and forward latencies (src/mem/XBar.py).
    detection_event_cycles_per_round is what forming a round's detection
    events costs the weak decoder chip ahead of that write, read under
    controller.detection_events_formed_at weak_syndrome_buffer alone;
    Yang et al. fix their syndrome calculation at 5 FPGA clock cycles
    (2605.04892 lines 1273-1275).
    A zero cost runs synchronously without clock-edge alignment.
    """

    kind: str = "syndrome_buffer"
    rounds: Optional[int] = None
    clock: Optional[config.Clock] = None
    write_cycles: int = 0
    read_cycles: int = 0
    detection_event_cycles_per_round: int = 0

    def __post_init__(self) -> None:
        config.check_cycles(
            "weak_syndrome_buffer.write_cycles", self.write_cycles
        )
        config.check_cycles(
            "weak_syndrome_buffer.read_cycles", self.read_cycles
        )
        config.check_cycles(
            "weak_syndrome_buffer.detection_event_cycles_per_round",
            self.detection_event_cycles_per_round,
        )
        charged = self.write_cycles + self.read_cycles
        charged += self.detection_event_cycles_per_round
        if charged > 0 and self.clock is None:
            raise ValueError("charged weak_syndrome_buffer costs need a clock")

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        default_clock: Optional[config.Clock] = None,
    ) -> "SyndromeBufferSettings":
        """A store section: its kind, and `rounds`, a positive count or null."""
        kind = section.get("kind", "syndrome_buffer")
        rounds = section.get("rounds")
        if rounds is not None and rounds < 1:
            raise ValueError(
                f"a syndrome buffer holds at least one round, got {rounds!r}"
            )
        clock = default_clock
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        write_cycles = section.get("write_cycles", 0)
        read_cycles = section.get("read_cycles", 0)
        formation_cycles = section.get("detection_event_cycles_per_round", 0)
        return cls(
            kind=kind,
            rounds=rounds,
            clock=clock,
            write_cycles=write_cycles,
            read_cycles=read_cycles,
            detection_event_cycles_per_round=formation_cycles,
        )


# the costs the weak syndrome buffer charges; the strong one charges none
_WEAK_BUFFER_COSTS = (
    "write_cycles",
    "read_cycles",
    "detection_event_cycles_per_round",
)


def check_strong_section_charges_nothing(section: Mapping) -> None:
    """The strong syndrome buffer's section prices no access.

    Its receiving end stores a round at the tick it lands
    (strong_syndrome_round_receiver.py), so a cost written there would
    be read and never paid.
    """
    for key in _WEAK_BUFFER_COSTS:
        cycles = section.get(key, 0)
        if cycles > 0:
            raise ValueError(
                f"strong_syndrome_buffer.{key} is a cost of the weak "
                "syndrome buffer; the strong syndrome buffer stores a "
                "round as it lands and charges nothing, so leave it out"
            )
