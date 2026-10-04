"""The ported syndrome buffer: the store of rounds behind memory ports.

The plain store's rounds, holds and room, with accesses that take words
on ports. A round is stored in whole words, so a request needs k = sum
over its rounds of ceil(bits / word_bits) accesses: gem5's crossbar
charges `divCeil(pkt->getSize(), width)` per packet
(src/mem/xbar.cc:135), and Helios reads its syndrome FIFO a byte a clock
with every round on a byte boundary (Helios_scalable_QEC
design/stage_controller/control_node_single_FPGA.v lines 35-36 and
152-167). Requests are served in arrival order on the first free port
their direction may use (gem5 src/mem/xbar.cc:206 and 288-289). A
request holds its port for k x cycles_per_access cycles from the later
of the clock edge at or after its arrival and the tick the port frees,
SimpleMemory's busy time (src/mem/simple_mem.cc:154-161), and its data
is usable access_latency_cycles after that (simple_mem.cc:174). No later
request overtakes a booked one, so the completion is known at booking.

The port kinds are OpenRAM's read, write and read/write ports
(compiler/options.py num_r_ports, num_w_ports, num_rw_ports). IBM's
decoder FPGA puts the syndrome "into a FIFO" (2510.21600 lines 506-509)
and states no width.
"""

import dataclasses
from typing import ClassVar, Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.records.rounds as round_records
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module

# the port kinds, each by the directions it serves
_READ = "read"
_WRITE = "write"
_READ_WRITE = "read_write"


@dataclasses.dataclass(frozen=True)
class PortedSyndromeBufferSettings:
    """The ported store's settings.

    bits bounds the store; None is unbounded. read_ports, write_ports
    and read_write_ports count OpenRAM's three port kinds. word_bits is
    what one access moves, cycles_per_access how long one access holds
    its port, and access_latency_cycles how long after its last access
    the data is usable, both on clock; clock None is the machine's
    clock. AFS's memory read, "a readout time of four cycles to read
    32-bit data" per read (2001.06598 lines 529-531, 1107-1110), is
    word_bits 32, cycles_per_access 4 and access_latency_cycles 0. The
    defaults are the sky130 pseudo dual port byte FIFO (VLSIDA
    sky130_sram_macros at 965df150, sky130_sram_1kbyte_1r1w_8x1024_8.py
    lines 6 and 14-16), a word a clock with no added latency as Helios's
    fall-through FIFO (design/generics/fifo_fwft.v lines 95, 105 and
    115).
    """

    bits: Optional[int] = None
    clock: Optional[config.Clock] = None
    read_ports: int = 1
    write_ports: int = 1
    read_write_ports: int = 0
    word_bits: int = 8
    cycles_per_access: int = 1
    access_latency_cycles: int = 0

    # the read port moves a read's bits by words, so a rate on the link
    # out of the store would charge them twice
    prices_read_bits: ClassVar[bool] = True

    def __post_init__(self) -> None:
        config.check_capacity_bits("ported_syndrome_buffer.bits", self.bits)
        config.check_cycles(
            "ported_syndrome_buffer.read_ports", self.read_ports
        )
        config.check_cycles(
            "ported_syndrome_buffer.write_ports", self.write_ports
        )
        config.check_cycles(
            "ported_syndrome_buffer.read_write_ports", self.read_write_ports
        )
        _check_word_bits(self.word_bits)
        _check_at_least_one_cycle(self.cycles_per_access)
        config.check_cycles(
            "ported_syndrome_buffer.access_latency_cycles",
            self.access_latency_cycles,
        )
        self._check_both_directions_have_a_port()

    def build(self, engine: engine_module.Engine) -> "PortedSyndromeBuffer":
        """A fresh store on these settings."""
        return PortedSyndromeBuffer(self, engine)

    def _check_both_directions_have_a_port(self) -> None:
        reading_ports = self.read_ports + self.read_write_ports
        writing_ports = self.write_ports + self.read_write_ports
        if reading_ports >= 1 and writing_ports >= 1:
            return
        raise ValueError(
            "a ported_syndrome_buffer needs a port that reads and a port "
            "that writes: read_ports plus read_write_ports, and write_ports "
            "plus read_write_ports, must each be at least one"
        )


class PortedSyndromeBuffer(syndrome_buffer_module.SyndromeBuffer):
    """The store whose accesses take words on ports in arrival order."""

    def __init__(
        self,
        settings: PortedSyndromeBufferSettings,
        engine: engine_module.Engine,
    ) -> None:
        syndrome_buffer_module.SyndromeBuffer.__init__(self, settings, engine)
        self.memory_ports = _memory_ports(settings)

    def book_write(self, round_key: tuple, bits: Optional[int]) -> int:
        """Take the first free port that writes, for the round's words."""
        stored_bits = round_records.stated_bits(bits)
        word_count = self._words(stored_bits)
        round_keys = (round_key,)
        return self._serve(_WRITE, round_keys, word_count)

    def book_read(self, round_keys: tuple) -> int:
        """Take the first free port that reads, for every round's words."""
        word_count = 0
        for round_key in round_keys:
            stored = self.round_by_key.get(round_key)
            assert stored is not None, (
                f"round {round_key!r} is read and is not stored"
            )
            word_count += self._words(stored.held_bits)
        return self._serve(_READ, round_keys, word_count)

    def _words(self, bits: int) -> int:
        """A round stored in whole words: one more word for a part word."""
        word_count, part_word_bits = divmod(bits, self.settings.word_bits)
        if part_word_bits:
            word_count += 1
        return word_count

    def _serve(self, direction: str, round_keys: tuple, word_count: int) -> int:
        """Book one request on the port that frees first; its completion."""
        period_ticks = self.settings.clock.period_ticks
        arrival_ticks = self.engine.now
        port_index = self._first_free_port(direction)
        port = self.memory_ports[port_index]
        edge_ticks = self.settings.clock.edge(0, arrival_ticks)
        start_ticks = max(edge_ticks, port.free_ticks)
        busy_cycles = word_count * self.settings.cycles_per_access
        port.free_ticks = start_ticks + busy_cycles * period_ticks
        latency_cycles = self.settings.access_latency_cycles
        completion_ticks = port.free_ticks + latency_cycles * period_ticks
        self.trace.access_served.fire(
            direction,
            port_index,
            round_keys,
            arrival_ticks,
            start_ticks,
            completion_ticks,
        )
        return completion_ticks

    def _first_free_port(self, direction: str) -> int:
        """The port this direction may use that frees first, lowest first."""
        chosen_index = None
        for index, port in enumerate(self.memory_ports):
            if not port.serves(direction):
                continue
            if chosen_index is None:
                chosen_index = index
                continue
            chosen = self.memory_ports[chosen_index]
            if port.free_ticks < chosen.free_ticks:
                chosen_index = index
        return chosen_index


class _MemoryPort:
    """One memory port of the store."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.free_ticks = 0

    def serves(self, direction: str) -> bool:
        if self.kind == _READ_WRITE:
            return True
        return self.kind == direction


def _memory_ports(settings: PortedSyndromeBufferSettings) -> list:
    """The store's ports in a fixed order: reads, writes, then read/writes."""
    kinds = [_READ] * settings.read_ports
    kinds += [_WRITE] * settings.write_ports
    kinds += [_READ_WRITE] * settings.read_write_ports
    return [_MemoryPort(kind) for kind in kinds]


def _check_word_bits(word_bits: int) -> None:
    """A word is a whole number of bits, at least one; never unbounded."""
    key = "ported_syndrome_buffer.word_bits"
    config.check_capacity_bits(key, word_bits)
    if word_bits is not None:
        return
    raise ValueError(f"{key} must be at least one bit (got None)")


def _check_at_least_one_cycle(cycles_per_access: int) -> None:
    """An access that holds its port no time is not a memory access."""
    config.check_cycles(
        "ported_syndrome_buffer.cycles_per_access", cycles_per_access
    )
    if cycles_per_access >= 1:
        return
    raise ValueError(
        "ported_syndrome_buffer.cycles_per_access must be at least one cycle"
    )
