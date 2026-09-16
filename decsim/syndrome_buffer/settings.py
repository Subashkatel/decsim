"""A round store's capacity and access costs on its named clock."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.config as config


@dataclasses.dataclass(frozen=True)
class RoundStoreSettings:
    """The yaml's `round_store` and `strong_round_store` sections.

    Table row (ROUND_STORES, round_store.py): round_store. rounds bounds
    the store; None is unbounded. A full store makes the controller hold
    the finished round and write it in order once a slot frees, the
    backpressure real systems apply to their source (Caune et al.
    2410.05202: the sequencer stalls on the decoder's status register).
    Buffer 0 charges write_cycles before publication and read_cycles
    once per primary window before assembling its input. The stages
    follow gem5's frontend and forward latencies (src/mem/XBar.py).
    A zero cost runs synchronously without clock-edge alignment.
    """

    kind: str = "round_store"
    rounds: Optional[int] = None
    clock: Optional[config.Clock] = None
    write_cycles: int = 0
    read_cycles: int = 0

    def __post_init__(self) -> None:
        config.check_cycles("round_store.write_cycles", self.write_cycles)
        config.check_cycles("round_store.read_cycles", self.read_cycles)
        charged = self.write_cycles + self.read_cycles
        if charged > 0 and self.clock is None:
            raise ValueError("charged round_store costs need a clock")

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        clocks: config.ClockSettings,
        default_clock: Optional[config.Clock] = None,
    ) -> "RoundStoreSettings":
        """A store section: its kind, and `rounds`, a positive count or null."""
        kind = section.get("kind", "round_store")
        rounds = section.get("rounds")
        if rounds is not None and rounds < 1:
            raise ValueError(
                f"a round store holds at least one round, got {rounds!r}"
            )
        clock = default_clock
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        write_cycles = section.get("write_cycles", 0)
        read_cycles = section.get("read_cycles", 0)
        return cls(
            kind=kind,
            rounds=rounds,
            clock=clock,
            write_cycles=write_cycles,
            read_cycles=read_cycles,
        )
