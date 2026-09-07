"""The transfer records of decsim/records/transfers.py.

An attribution says whose bits moved. Its patches are ordered by the
stable identity bytes, not by Python's comparison, so a traffic ledger
reads the same whatever identity types a front end chose.
"""

import decsim.records.identity as identity_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


def make_fragment(patch_id):
    return round_records.RetainedSyndromeFragment(
        operation_id=7,
        patch_id=patch_id,
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
            key=identity_records.stable_identity_order_key,
        )
    )
    assert attribution.patch_ids == expected
    assert attribution.first_round == 3
    assert attribution.last_round == 3
    assert attribution.window_id is None


def test_a_packets_attribution_names_every_patch_of_the_round():
    """The packed round moves one round of every patch that reported."""
    later_patch = make_fragment("patch-b")
    earlier_patch = make_fragment("patch-a")
    packet = round_records.SyndromeRoundPacket(
        operation_id=7,
        round_index=3,
        fragments=(later_patch, earlier_patch),
    )
    attribution = transfer_records.TransferAttribution.for_packet(packet)
    assert attribution.patch_ids == ("patch-a", "patch-b")
    assert attribution.operation_id == 7
    assert attribution.first_round == 3


def test_a_windows_attribution_covers_the_rounds_it_reads():
    """The range is the window's reads, both buffers included."""
    window = window_records.Window(4, 2, 3, 5, 6, 6, buffer_lo=1)
    operation = program_records.Operation(id=4, name="memory", qubits=("q0",))
    operation.patches = ("patch-a",)
    request_key = window_records.DecoderRequestKey(
        4, 2, window_records.DecoderTier.WEAK, 1
    )
    attribution = transfer_records.TransferAttribution.for_window(
        window, operation, request_key
    )
    assert attribution.first_round == 1
    assert attribution.last_round == 6
    assert attribution.window_id == 2
    assert attribution.relation.request_key is request_key
