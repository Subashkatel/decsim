"""How a link cuts one message into the frames its wire carries.

A message is the path's header and its payload; the frames add the
protocol's own framing. A message of no bits is one frame, because a
message arrives only when a frame lands. The bits are those the
channel's rate is written for: framings that count a line code
(aurora_64b66b, flits) at the line rate, those that count bytes
(pcie_tlp, roce_v2, ethernet_udp) at the data rate after line coding,
as pcie-bench prices them (model/pcie.py:40-48, model/eth.py:29-41).
"""

import dataclasses
from typing import ClassVar

import decsim.links.settings as link_settings

BITS_PER_BYTE = 8

# Aurora 64B/66B (Xilinx SP011 v1.2): a block is 64 data bits behind a
# 2-bit sync header (section 5.2, page 39); "Data blocks carry eight
# octets", and a frame ends with a Separator block carrying "0 to 6
# octets" or a Separator-7 carrying "exactly seven" (section 2.1, page
# 22; section 2.5, page 26).
AURORA_BLOCK_BITS = 66
AURORA_OCTETS_PER_BLOCK = 8

# A PCIe memory write TLP with a 64-bit address and no ECRC: 2 bytes of
# framing and 6 of data link layer header (DLLP_Hdr), a 4-byte TLP header
# and a 12-byte memory write header (pcie-bench model/pcie.py:220-240;
# Neugebauer et al., SIGCOMM 2018, lines 385-389, "MWr_Hdr is 24B").
PCIE_MEMORY_WRITE_OVERHEAD_BYTES = 24

# Ethernet on the wire: preamble 7, start of frame 1, header 14, FCS 4
# and the standard 12-byte interframe gap, with a 46-byte minimum
# payload (pcie-bench model/eth.py:29-41, no VLAN tag).
ETHERNET_OVERHEAD_BYTES = 38
ETHERNET_MINIMUM_PAYLOAD_BYTES = 46
IPV4_HEADER_BYTES = 20
UDP_HEADER_BYTES = 8

# RoCE v2 inside the UDP datagram (UDP port 4791, Linux rxe_net.c:296):
# the 12-byte base transport header (rxe_hdr.h:61-67), the 16-byte RDMA
# extended transport header on the first packet of a write
# (rxe_hdr.h:526-530), the payload padded to four bytes, "pad =
# (-payload) & 0x3" (rxe_req.c:423), and the 4-byte invariant CRC
# (rxe_hdr.h:55; paylen at rxe_req.c:430).
ROCE_BASE_TRANSPORT_HEADER_BYTES = 12
ROCE_RDMA_EXTENDED_HEADER_BYTES = 16
# An acknowledgement is its own packet, opcode RC_ACKNOWLEDGE: the base
# header and the 4-byte ACK extended transport header, no payload, then
# the invariant CRC (rxe_opcode.c:317-321, "RXE_BTH_BYTES +
# RXE_AETH_BYTES"; rxe_hdr.h:773-775; paylen at rxe_resp.c:779-780).
ROCE_ACK_EXTENDED_HEADER_BYTES = 4
ROCE_INVARIANT_CRC_BYTES = 4
ROCE_PAYLOAD_ALIGNMENT_BYTES = 4


class Whole:
    """One frame of the whole message, the header with it.

    ns-3's point-to-point device adds its header and times the packet as
    one (point-to-point-net-device.cc:528 and :243).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The whole framing has no keys of its own."""

        # no acknowledgement packet of its own (RoceV2.Settings has one)
        has_acknowledgement_packet: ClassVar[bool] = False

        def build(self) -> "Whole":
            """A fresh framing."""
            return Whole()

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """The one frame."""
        message_bits = payload_bits + header_bits
        return (message_bits,)


class Flits:
    """Flits of flit_bits each; the last one is padded to the full width.

    Garnet cuts a message into divCeil(size, bitWidth) flits of the
    link's width (gem5 src/mem/ruby/network/garnet/NetworkInterface.cc
    lines 382-387).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The width of one flit, a positive whole number of bits."""

        flit_bits: int

        has_acknowledgement_packet: ClassVar[bool] = False

        def __post_init__(self) -> None:
            link_settings.check_positive_count("flit_bits", self.flit_bits)

        def build(self) -> "Flits":
            """A fresh framing on these settings."""
            return Flits(self)

    def __init__(self, settings: "Flits.Settings") -> None:
        self._flit_bits = settings.flit_bits

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """One flit per flit_bits of the message, at least one."""
        message_bits = payload_bits + header_bits
        flit_count = _chunk_count(message_bits, self._flit_bits)
        return (self._flit_bits,) * flit_count


class Aurora64b66b:
    """66-bit blocks: floor(n / 8) + 1 of them for a frame of n octets.

    A block is one entry of the receiver's FIFO, the unit Aurora's native
    flow control counts (SP011 section 3.1, page 29): full blocks of eight
    octets, then one separator block with the last zero to seven.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """Aurora's blocks have no keys of their own."""

        has_acknowledgement_packet: ClassVar[bool] = False

        def build(self) -> "Aurora64b66b":
            """A fresh framing."""
            return Aurora64b66b()

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """One 66-bit frame per block of the Aurora frame."""
        message_bits = payload_bits + header_bits
        octet_count = _ceiling_division(message_bits, BITS_PER_BYTE)
        full_blocks = octet_count // AURORA_OCTETS_PER_BLOCK
        block_count = full_blocks + 1
        return (AURORA_BLOCK_BITS,) * block_count


class PcieTlp:
    """Memory write TLPs of at most max_payload_bytes, 24 bytes on each.

    A write of n bytes costs ceil(n / MPS) x 24 + n bytes (pcie-bench
    model/mem_bw.py:40-41; Neugebauer lines 385-389); one of no bytes is one
    TLP of overhead, where pcie-bench counts none. Data link ACKs,
    flow-control updates and replay are left out: this row pairs with the
    credit protocol only.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The link's Maximum Payload Size, a positive whole number of bytes."""

        max_payload_bytes: int

        has_acknowledgement_packet: ClassVar[bool] = False

        def __post_init__(self) -> None:
            link_settings.check_positive_count(
                "max_payload_bytes", self.max_payload_bytes
            )

        def build(self) -> "PcieTlp":
            """A fresh framing on these settings."""
            return PcieTlp(self)

    def __init__(self, settings: "PcieTlp.Settings") -> None:
        self._max_payload_bytes = settings.max_payload_bytes

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """One TLP per Maximum Payload Size of the message's bytes."""
        message_bits = payload_bits + header_bits
        message_bytes = _ceiling_division(message_bits, BITS_PER_BYTE)
        chunks = _chunks(message_bytes, self._max_payload_bytes)
        frames = []
        for chunk_bytes in chunks:
            tlp_bytes = chunk_bytes + PCIE_MEMORY_WRITE_OVERHEAD_BYTES
            tlp_bits = tlp_bytes * BITS_PER_BYTE
            frames.append(tlp_bits)
        return tuple(frames)


class RoceV2:
    """RDMA write packets of at most path_mtu_bytes, each an Ethernet frame.

    The first packet carries the RDMA extended header; every packet carries
    IPv4, UDP, the base transport header, its payload padded to four bytes
    and the invariant CRC, inside Ethernet's overhead.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The connection's path MTU, a positive whole number of bytes."""

        path_mtu_bytes: int

        # an ACK or a NAK is a RoCE packet (acknowledgement_bits)
        has_acknowledgement_packet: ClassVar[bool] = True

        def __post_init__(self) -> None:
            link_settings.check_positive_count(
                "path_mtu_bytes", self.path_mtu_bytes
            )

        def build(self) -> "RoceV2":
            """A fresh framing on these settings."""
            return RoceV2(self)

    def __init__(self, settings: "RoceV2.Settings") -> None:
        self._path_mtu_bytes = settings.path_mtu_bytes

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """One Ethernet frame per path MTU of the message's bytes."""
        message_bits = payload_bits + header_bits
        message_bytes = _ceiling_division(message_bits, BITS_PER_BYTE)
        chunks = _chunks(message_bytes, self._path_mtu_bytes)
        frames = []
        for index, chunk_bytes in enumerate(chunks):
            is_first = index == 0
            datagram_bytes = _roce_datagram_bytes(chunk_bytes, is_first)
            frame_bytes = _ethernet_frame_bytes(datagram_bytes)
            frame_bits = frame_bytes * BITS_PER_BYTE
            frames.append(frame_bits)
        return tuple(frames)

    def acknowledgement_bits(self) -> int:
        """The wire bits of one ACK or NAK, whatever the path MTU."""
        transport_bytes = (
            ROCE_BASE_TRANSPORT_HEADER_BYTES
            + ROCE_ACK_EXTENDED_HEADER_BYTES
            + ROCE_INVARIANT_CRC_BYTES
        )
        headers_bytes = IPV4_HEADER_BYTES + UDP_HEADER_BYTES
        datagram_bytes = headers_bytes + transport_bytes
        frame_bytes = _ethernet_frame_bytes(datagram_bytes)
        return frame_bytes * BITS_PER_BYTE


class EthernetUdp:
    """UDP datagrams of at most mtu_bytes, IPv4 and UDP headers inside it.

    NVQLink chose an unreliable connection over UDP on purpose (2510.25213
    lines 376-388).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The link's MTU, a whole number of bytes above the two headers."""

        mtu_bytes: int

        has_acknowledgement_packet: ClassVar[bool] = False

        def __post_init__(self) -> None:
            link_settings.check_positive_count("mtu_bytes", self.mtu_bytes)
            headers_bytes = IPV4_HEADER_BYTES + UDP_HEADER_BYTES
            if self.mtu_bytes > headers_bytes:
                return
            raise ValueError(
                f"mtu_bytes is {self.mtu_bytes}; the MTU holds the "
                f"{headers_bytes} bytes of IPv4 and UDP headers and at "
                f"least one byte of data"
            )

        def build(self) -> "EthernetUdp":
            """A fresh framing on these settings."""
            return EthernetUdp(self)

    def __init__(self, settings: "EthernetUdp.Settings") -> None:
        self._mtu_bytes = settings.mtu_bytes

    def frames(self, payload_bits: int, header_bits: int) -> tuple[int, ...]:
        """One Ethernet frame per datagram."""
        message_bits = payload_bits + header_bits
        message_bytes = _ceiling_division(message_bits, BITS_PER_BYTE)
        headers_bytes = IPV4_HEADER_BYTES + UDP_HEADER_BYTES
        data_per_datagram = self._mtu_bytes - headers_bytes
        chunks = _chunks(message_bytes, data_per_datagram)
        frames = []
        for chunk_bytes in chunks:
            datagram_bytes = headers_bytes + chunk_bytes
            frame_bytes = _ethernet_frame_bytes(datagram_bytes)
            frame_bits = frame_bytes * BITS_PER_BYTE
            frames.append(frame_bits)
        return tuple(frames)


def _roce_datagram_bytes(chunk_bytes: int, is_first: bool) -> int:
    """IPv4, UDP and RoCE headers, the padded payload and the ICRC."""
    padding_bytes = -chunk_bytes % ROCE_PAYLOAD_ALIGNMENT_BYTES
    transport_bytes = ROCE_BASE_TRANSPORT_HEADER_BYTES
    if is_first:
        transport_bytes += ROCE_RDMA_EXTENDED_HEADER_BYTES
    headers_bytes = IPV4_HEADER_BYTES + UDP_HEADER_BYTES + transport_bytes
    padded_bytes = chunk_bytes + padding_bytes
    return headers_bytes + padded_bytes + ROCE_INVARIANT_CRC_BYTES


def _ethernet_frame_bytes(ethernet_payload_bytes: int) -> int:
    """The bytes an Ethernet frame holds the wire for, gap included."""
    padded_bytes = max(ethernet_payload_bytes, ETHERNET_MINIMUM_PAYLOAD_BYTES)
    return ETHERNET_OVERHEAD_BYTES + padded_bytes


def _chunks(total: int, chunk_size: int) -> tuple[int, ...]:
    """Total cut into chunk_size pieces, the last one short, at least one."""
    chunk_count = _chunk_count(total, chunk_size)
    chunks = []
    remaining = total
    for _index in range(chunk_count):
        chunk = min(remaining, chunk_size)
        chunks.append(chunk)
        remaining -= chunk
    return tuple(chunks)


def _chunk_count(total: int, chunk_size: int) -> int:
    """ceil(total / chunk_size), and one for an empty message."""
    count = _ceiling_division(total, chunk_size)
    return max(count, 1)


def _ceiling_division(numerator: int, denominator: int) -> int:
    """gem5's divCeil: the whole units that hold numerator."""
    return -(-numerator // denominator)
