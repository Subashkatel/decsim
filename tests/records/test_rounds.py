"""The round records of decsim/records/rounds.py.

The readout carries no analog waveform: its bits are the sampled outcome,
normalized once on the way into a retained fragment, and the packet's
defect text is the sparse form the I/O trace prints.
"""

import numpy as np

import decsim.records.rounds as round_records


def make_fragment(**overrides):
    values = {
        "operation_id": 7,
        "patch_id": "patch-a",
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
        patch_id="patch",
        round_index=2,
        bits=None,
        size_bits=2,
    )
    fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
    assert fragment.bits is None


def test_readout_bits_from_a_boolean_array_become_zero_one_integers():
    """A one-dimensional NumPy boolean readout normalizes to 0/1 integers."""
    bits = np.array([True, False, True], dtype=bool)
    readout = round_records.QPUReadout(
        operation_id="operation",
        patch_id="patch",
        round_index=2,
        bits=bits,
        size_bits=3,
    )
    fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
    assert fragment.bits == (1, 0, 1)


def test_retained_fragment_normalizes_readout_bits():
    """A fragment normalizes the bits and copies the transport metadata."""
    readout = round_records.QPUReadout(
        operation_id="operation",
        patch_id="patch",
        round_index=2,
        bits=[True, 0],
        code="code",
        fragment_index=3,
        size_bits=2,
    )
    fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
    assert fragment.bits == (1, 0)
    assert fragment.operation_id == "operation"
    assert fragment.fragment_index == 3


def test_round_packet_preserves_supplied_fragment_order():
    """Round packets preserve supplied fragment order without sorting."""
    later_fragment = make_fragment(patch_id="patch-b", fragment_index=1)
    earlier_fragment = make_fragment(patch_id="patch-a", fragment_index=0)
    packet = round_records.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(later_fragment, earlier_fragment),
    )
    assert packet.fragments == (later_fragment, earlier_fragment)


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


def test_a_round_with_no_set_bit_reads_as_no_defects():
    """A round of zeros is printed as no defects, not as an empty set."""
    fragment = make_fragment(bits=(0, 0))
    packet = round_records.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(fragment,),
    )
    assert packet.defects_text() == "no defects"


def test_a_round_missing_one_fragments_bits_reads_as_timing_only():
    """One fragment without bits makes the whole round timing-only."""
    with_bits = make_fragment(bits=(1, 1))
    without_bits = make_fragment(bits=None)
    packet = round_records.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(with_bits, without_bits),
    )
    assert packet.defects_text() == "timing-only"
