"""The round records of decsim/records/rounds.py.

The readout carries no analog waveform: its bits are the sampled outcome,
normalized once on the way into a retained fragment, and the packet's
defect text is the sparse form the I/O trace prints.
"""

import numpy as np
import pytest

import decsim.records.rounds as round_records


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


def test_measurement_partitions_preserve_interleaved_patch_order() -> None:
    readout = round_records.QPUReadout(
        7, ("left", "right"), 1, bits=(1, 0, 1, 1), size_bits=4
    )
    partitions = (
        round_records.MeasurementPartition(("left",), 1),
        round_records.MeasurementPartition(("right",), 2),
        round_records.MeasurementPartition(("left",), 1),
    )

    readouts = round_records.partition_measurements(readout, partitions)

    measurement_groups = [part.bits for part in readouts]
    assert measurement_groups == [(1,), (0, 1), (1,)]
    footprints = [part.patch_ids for part in readouts]
    assert footprints == [
        ("left",),
        ("right",),
        ("left",),
    ]
    widths = [part.size_bits for part in readouts]
    assert widths == [1, 2, 1]


def test_measurement_partitions_cannot_omit_raw_outcomes() -> None:
    readout = round_records.QPUReadout(7, ("left",), 1, bits=(1, 0))
    partitions = (round_records.MeasurementPartition(("left",), 1),)

    with pytest.raises(ValueError, match="cover every measurement"):
        round_records.partition_measurements(readout, partitions)


def test_measurement_partitions_cannot_name_an_unrelated_patch() -> None:
    readout = round_records.QPUReadout(7, ("left",), 1, bits=(1,))
    partitions = (round_records.MeasurementPartition(("right",), 1),)

    with pytest.raises(ValueError, match="outside the readout footprint"):
        round_records.partition_measurements(readout, partitions)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_partition_counts_are_nonnegative_integers(count: object) -> None:
    with pytest.raises(ValueError, match="must be a nonnegative integer"):
        round_records.MeasurementPartition((0,), count)


@pytest.mark.parametrize("patches", [(True,), (1.0,), ([1],)])
def test_partition_footprints_use_stable_identities(patches: tuple) -> None:
    with pytest.raises(ValueError, match="must be stable identities"):
        round_records.MeasurementPartition(patches, 1)


@pytest.mark.parametrize("patches", [(), (0, 0)])
def test_partition_footprints_require_distinct_patches(patches: tuple) -> None:
    with pytest.raises(ValueError, match="needs unique patches"):
        round_records.MeasurementPartition(patches, 1)


def test_partitioning_requires_physical_measurement_bits() -> None:
    readout = round_records.QPUReadout(7, (0,), 1)
    partitions = (round_records.MeasurementPartition((0,), 1),)
    with pytest.raises(ValueError, match="require raw outcomes"):
        round_records.partition_measurements(readout, partitions)


def test_an_empty_measurement_round_keeps_its_declared_footprint() -> None:
    readout = round_records.QPUReadout(7, (0,), 1, bits=())
    partitions = (round_records.MeasurementPartition((0,), 0),)
    readouts = round_records.partition_measurements(readout, partitions)
    expected = round_records.QPUReadout(7, (0,), 1, bits=(), size_bits=0)
    assert readouts == [expected]


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


def test_readout_bits_from_a_boolean_array_become_zero_one_integers():
    """A one-dimensional NumPy boolean readout normalizes to 0/1 integers."""
    bits = np.array([True, False, True], dtype=bool)
    readout = round_records.QPUReadout(
        operation_id="operation",
        patch_ids=("patch",),
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
        patch_ids=("patch",),
        round_index=2,
        bits=[True, 0],
        fragment_index=3,
        size_bits=2,
    )
    fragment = round_records.RetainedSyndromeFragment.from_readout(readout)
    assert fragment.bits == (1, 0)
    assert fragment.operation_id == "operation"
    assert fragment.fragment_index == 3


def test_round_packet_preserves_supplied_fragment_order():
    """Round packets preserve supplied fragment order without sorting."""
    later_fragment = make_fragment(patch_ids=("patch-b",), fragment_index=1)
    earlier_fragment = make_fragment(patch_ids=("patch-a",), fragment_index=0)
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
