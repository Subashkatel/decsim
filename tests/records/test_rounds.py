"""The round records of decsim/records/rounds.py.

A timing-only readout keeps no bits, an escalated region rides under
one name, and the packet's defect text is the sparse form the I/O trace
prints.
"""

import decsim.records.rounds as round_records
import decsim.records.windows as window_records


def make_fragment(**overrides):
    values = {
        "operation_id": 7,
        "patch_ids": ("patch-a",),
        "round_index": 3,
        "bits": (0, 1),
        "size_bits": 2,
        "fragment_index": 0,
    }
    values.update(overrides)
    return round_records.RetainedSyndromeFragment(**values)


def test_a_timing_only_readout_retains_no_bits():
    """A readout with no bits stays a timing-only fragment."""
    readout = round_records.QPUReadout(
        operation_id="operation",
        patch_ids=("patch",),
        round_index=2,
        bits=None,
        size_bits=2,
    )
    fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
    assert fragment.bits is None


def test_a_regions_rounds_before_ride_under_its_one_name():
    """Two raw rounds before round 3 add their bits and no second name.

    One gem5 DMA request covers its whole range (src/dev/dma_device.cc
    lines 195-207), so the message is the name and six bits.
    """
    request_key = window_records.DecoderRequestKey(
        7, 0, window_records.DecoderTier.STRONG, 1
    )
    two_bits = make_fragment()
    first = round_records.SyndromeRoundPacket(7, 1, (two_bits,))
    second = round_records.SyndromeRoundPacket(7, 2, (two_bits,))
    third = round_records.SyndromeRoundPacket(7, 3, (two_bits,))

    region = round_records.EscalatedRegion.of(
        request_key, (first, second, third), 3
    )

    assert region.rounds_before == (first, second)
    assert region.packets == (third,)
    assert region.message_bits() == 64 + 6


def test_a_successors_rounds_are_never_rounds_before():
    """Round 1 of operation 8 follows the window; round 2 of 7 precedes it."""
    request_key = window_records.DecoderRequestKey(
        7, 0, window_records.DecoderTier.STRONG, 1
    )
    two_bits = make_fragment()
    before = round_records.SyndromeRoundPacket(7, 2, (two_bits,))
    own = round_records.SyndromeRoundPacket(7, 3, (two_bits,))
    successor = round_records.SyndromeRoundPacket(8, 1, (two_bits,))

    region = round_records.EscalatedRegion.of(
        request_key, (before, own, successor), 3
    )

    assert region.rounds_before == (before,)
    assert region.packets == (own, successor)
    assert region.round_keys == ((7, 2), (7, 3), (8, 1))


def test_defect_text_counts_positions_across_the_fragments_in_order():
    """The defect indices run across the fragments, in the packet's order."""
    first_fragment = make_fragment(bits=(0, 1), fragment_index=0)
    second_fragment = make_fragment(bits=(1, 0, 1), fragment_index=1)
    packet = round_records.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(first_fragment, second_fragment),
    )
    assert packet.defects_text() == "defects {1, 2, 4}"
