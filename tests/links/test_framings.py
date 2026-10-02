"""A framing row cuts a message into the frames its protocol puts on a wire.

Sources: gem5 src/mem/ruby/network/garnet/NetworkInterface.cc:382-387
(divCeil(size, bitWidth) flits); Xilinx SP011 v1.2 sections 2.1, 2.5
and 5.2 (eight octets per 66-bit data block, a separator block for the
last zero to seven); pcie-bench model/mem_bw.py:40-41 and
model/pcie.py:220-240 (ceil(n / MPS) x 24 + n bytes for a write);
pcie-bench model/eth.py:29-41 (Ethernet's 38 bytes of overhead and
46-byte minimum payload); Linux drivers/infiniband/sw/rxe rxe_hdr.h and
rxe_req.c:423-430 (the RoCE v2 headers, the four-byte pad, the ICRC).
"""

import pytest

import decsim.links.framings as framings
import decsim.ports as ports


def framing(kind, **keys):
    record = framings.FRAMINGS[kind]
    settings = record(**keys)
    return settings.build()


def aurora_blocks(aurora, octets):
    message_bits = octets * 8
    frames = aurora.frames(message_bits, 0)
    return len(frames)


def test_every_framing_row_fills_the_framing_port():
    rows = [
        framing("whole"),
        framing("flits", flit_bits=128),
        framing("aurora_64b66b"),
        framing("pcie_tlp", max_payload_bytes=256),
        framing("roce_v2", path_mtu_bytes=4096),
        framing("ethernet_udp", mtu_bytes=1500),
    ]

    assert all(isinstance(row, ports.Framing) for row in rows)


def test_the_whole_framing_is_one_frame_of_payload_and_header():
    whole = framing("whole")

    assert whole.frames(100, 24) == (124,)
    assert whole.frames(0, 0) == (0,)


def test_flits_are_the_garnet_div_ceil_of_the_message_at_least_one():
    flits = framing("flits", flit_bits=128)

    assert flits.frames(576, 0) == (128,) * 5
    assert flits.frames(512, 0) == (128,) * 4
    assert flits.frames(500, 12) == (128,) * 4
    assert flits.frames(0, 0) == (128,)


def test_an_aurora_frame_of_n_octets_is_n_div_8_plus_1_blocks():
    """Hand-traced from SP011: full blocks of eight, then a separator."""
    aurora = framing("aurora_64b66b")
    octet_counts = [0, 1, 6, 7, 8, 9, 15, 16, 23, 24]
    expected_blocks = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]

    observed_blocks = [aurora_blocks(aurora, octets) for octets in octet_counts]

    assert observed_blocks == expected_blocks
    assert aurora.frames(80, 0) == (66, 66)


def test_a_pcie_write_is_24_bytes_on_every_tlp_of_at_most_mps_bytes():
    pcie = framing("pcie_tlp", max_payload_bytes=256)

    assert pcie.frames(64, 0) == (256,)
    assert pcie.frames(4800, 0) == (2240, 2240, 896)
    assert pcie.frames(0, 0) == (192,)


def test_a_roce_v2_write_carries_its_extended_header_on_the_first_packet():
    """Ethernet 38, IPv4 20, UDP 8, BTH 12, RETH 16, padded data, ICRC 4."""
    roce = framing("roce_v2", path_mtu_bytes=1024)

    first_bytes = 38 + 20 + 8 + 12 + 16 + 1024 + 4
    last_bytes = 38 + 20 + 8 + 12 + 8 + 4
    frames = roce.frames(8240, 0)
    assert frames == (first_bytes * 8, last_bytes * 8)


def test_a_roce_v2_payload_is_padded_to_four_bytes():
    roce = framing("roce_v2", path_mtu_bytes=4096)

    padded_bytes = 38 + 20 + 8 + 12 + 16 + 12 + 4
    frames = roce.frames(72, 0)
    assert frames == (padded_bytes * 8,)


def test_a_roce_v2_acknowledgement_is_bth_aeth_and_icrc_in_a_frame():
    """rxe_opcode.c:317-321: RC_ACKNOWLEDGE is BTH 12 and AETH 4."""
    roce = framing("roce_v2", path_mtu_bytes=1024)

    acknowledgement_bytes = 38 + 20 + 8 + 12 + 4 + 4
    assert roce.acknowledgement_bits() == acknowledgement_bytes * 8


def test_a_short_udp_datagram_is_padded_to_ethernets_minimum_payload():
    udp = framing("ethernet_udp", mtu_bytes=1500)

    assert udp.frames(64, 0) == (84 * 8,)
    assert udp.frames(11776, 0) == (1538 * 8,)
    assert udp.frames(11784, 0) == (1538 * 8, 84 * 8)


def test_a_framing_that_is_not_a_row_is_refused_with_the_rows():
    with pytest.raises(ValueError, match="is not a row of its table"):
        framings.framing_settings_from_yaml({"kind": "morse"}, "path")


def test_a_framing_key_its_row_does_not_declare_is_refused():
    section = {"kind": "flits", "flit_bits": 64, "flit_count": 3}

    with pytest.raises(ValueError, match="does not know"):
        framings.framing_settings_from_yaml(section, "path")


def test_a_flit_width_of_zero_is_refused():
    section = {"kind": "flits", "flit_bits": 0}

    with pytest.raises(ValueError, match="positive whole number"):
        framings.framing_settings_from_yaml(section, "path")
