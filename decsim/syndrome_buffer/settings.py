"""A syndrome buffer's capacity and access costs on its named clock."""

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import decsim.config as config
import decsim.tables as tables

# the keys every row of a syndrome buffer section shares; any other key is
# the row's own (its Settings, decsim/tables.py row_settings)
SYNDROME_BUFFER_KEYS = (
    "kind",
    "bits",
    "clock",
    "detection_event_cycles_per_round",
)


@dataclasses.dataclass(frozen=True)
class SyndromeBufferSettings:
    """The yaml's `weak_syndrome_buffer` and `strong_syndrome_buffer` sections.

    Table rows (SYNDROME_BUFFERS, ported_syndrome_buffer.py):
    syndrome_buffer and ported_syndrome_buffer.
    bits bounds the store, the capacity a memory is declared in; None is
    unbounded. gem5 sizes its packet store the same way, in bytes on the
    store itself (`rx_fifo_size = Param.MemorySize("384KiB", ...)`,
    src/dev/net/Ethernet.py) and answers room against the packet's own
    size (`avail()` over the reserved bytes, `reserve(len)` before the
    data lands, src/dev/net/pktfifo.hh). A full store makes the
    controller hold the finished round and write it in order once a slot frees,
    the backpressure real systems apply to their source (Caune et al.
    2410.05202: the sequencer stalls on the decoder's status register).
    The store prices its own accesses (book_write, book_read) on this
    clock, each row by its own keys (the default row's write_cycles and
    read_cycles, syndrome_buffer.py). The stages follow gem5's frontend
    and forward latencies (src/mem/XBar.py).
    detection_event_cycles_per_round is what forming a round's detection
    events costs the weak decoder chip ahead of that write, read under
    controller.detection_events_formed_at weak_syndrome_buffer alone;
    Yang et al. fix their syndrome calculation at 5 FPGA clock cycles
    (2605.04892 lines 1273-1275).
    A zero cost runs synchronously without clock-edge alignment.
    row_settings is the row's own Settings, read from the section's keys
    outside SYNDROME_BUFFER_KEYS, or None for a row that declares none;
    the row reads it off this record, which its constructor takes whole.
    """

    kind: str = "syndrome_buffer"
    bits: Optional[int] = None
    clock: Optional[config.Clock] = None
    detection_event_cycles_per_round: int = 0
    # the row's own Settings record, opaque to the section
    row_settings: Optional[Any] = None

    def __post_init__(self) -> None:
        config.check_cycles(
            "weak_syndrome_buffer.detection_event_cycles_per_round",
            self.detection_event_cycles_per_round,
        )
        charged = self.detection_event_cycles_per_round
        if charged > 0 and self.clock is None:
            raise ValueError("charged weak_syndrome_buffer costs need a clock")

    @classmethod
    def from_yaml(
        cls,
        section: Mapping,
        section_name: str,
        clocks: config.ClockSettings,
        buffer_rows: Mapping,
        default_clock: Optional[config.Clock] = None,
    ) -> "SyndromeBufferSettings":
        """A store section: its kind, and `bits`, a bit capacity or null.

        buffer_rows is SYNDROME_BUFFERS, which sits beside the store
        classes and so cannot be imported here
        (ported_syndrome_buffer.py).
        """
        kind = section.get("kind", "syndrome_buffer")
        row = tables.row(buffer_rows, f"{section_name}.kind", kind)
        row_settings = tables.row_settings(
            row, section_name, section, SYNDROME_BUFFER_KEYS
        )
        bits = section.get("bits")
        key = f"{section_name}.bits"
        config.check_capacity_bits(key, bits)
        clock = default_clock
        if "clock" in section:
            clock = clocks.clock(section["clock"])
        formation_cycles = section.get("detection_event_cycles_per_round", 0)
        return cls(
            kind=kind,
            bits=bits,
            clock=clock,
            detection_event_cycles_per_round=formation_cycles,
            row_settings=row_settings,
        )


# the keys only the weak syndrome buffer reads: its costs and the clock
# they are charged on
_WEAK_BUFFER_ONLY_KEYS = (
    "clock",
    "write_cycles",
    "read_cycles",
    "detection_event_cycles_per_round",
)


def check_strong_section_charges_nothing(section: Mapping) -> None:
    """The strong syndrome buffer's section prices no access.

    Its receiving end stores a round at the tick it lands
    (strong_syndrome_round_receiver.py), so a cost or its clock written
    there would be read and never paid, whatever its value.
    """
    kind = section.get("kind")
    check_strong_store_kind(kind)
    for key in _WEAK_BUFFER_ONLY_KEYS:
        if key in section:
            raise ValueError(
                f"strong_syndrome_buffer.{key} belongs to the weak "
                "syndrome buffer's costs; the strong syndrome buffer "
                "stores a round as it lands and charges nothing, so leave "
                "it out"
            )


def check_strong_store_kind(kind: Optional[str]) -> None:
    """The strong syndrome buffer is not a store with ports.

    Its receiving end stores a round as it lands and never books the
    write, while a ported store books its reads, so a ported strong
    store would price half its accesses. The strong side is a latency
    model behind its backend, so the kind is refused wherever the
    settings come from: the yaml load and the machine's build both ask
    here, as gem5 refuses a wrong neighbour when it binds the port
    whatever script built it (src/mem/port.cc:152, fatal_if).
    """
    if kind != "ported_syndrome_buffer":
        return
    raise ValueError(
        "strong_syndrome_buffer.kind ported_syndrome_buffer prices its "
        "accesses on ports; the strong syndrome buffer stores a round "
        "as it lands and charges nothing, so name syndrome_buffer"
    )
