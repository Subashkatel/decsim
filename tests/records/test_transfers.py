"""The transfer records of decsim/records/transfers.py.

An attribution says whose bits moved. Its patches are ordered by the
stable identity bytes, not by Python's comparison, so a traffic ledger
reads the same whatever identity types a front end chose.
"""

import decsim.records.identity as identity_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


def make_fragment(patch_id):
    return round_records.RetainedSyndromeFragment(
        operation_id=7,
        patch_ids=(patch_id,),
        round_index=3,
        bits=(0, 1),
        size_bits=2,
        fragment_index=0,
    )


def test_a_rounds_patches_are_ordered_by_the_stable_identity_bytes():
    """Mixed identity types order by their bytes, not by Python's compare."""
    attribution = transfer_records.TransferAttribution.for_round(
        7, ("patch-b", 2, "patch-a", 10), 3
    )
    expected = tuple(
        sorted(
            ("patch-b", 2, "patch-a", 10),
            key=identity_records.stable_identity_bytes,
        )
    )
    assert attribution.patch_ids == expected
    assert attribution.first_round == 3
    assert attribution.last_round == 3
    assert attribution.window_id is None


def test_a_regions_rounds_are_named_under_each_packets_own_operation():
    """A strong window past its operation's end reads the next one's rounds."""
    patch_a = make_fragment("patch-a")
    patch_b = make_fragment("patch-b")
    last_of_the_first = round_records.SyndromeRoundPacket(
        operation_id=1, round_index=6, fragments=(patch_a,)
    )
    first_of_the_next = round_records.SyndromeRoundPacket(
        operation_id=2, round_index=1, fragments=(patch_b,)
    )
    request_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 0
    )
    region = round_records.EscalatedRegion(
        request_key, (last_of_the_first, first_of_the_next), wire_bits=4
    )
    attribution = transfer_records.TransferAttribution.for_region(region)
    assert attribution.round_keys == ((1, 6), (2, 1))
