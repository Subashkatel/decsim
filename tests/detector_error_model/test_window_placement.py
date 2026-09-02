"""A plan entry, a window's rows and the exclusion ranges, read once at entry.

Sources: Skoric et al. 2209.08552, section I.B (a window is a commit
region inside a decoding region; a fault the decoder may use but may not
commit stays outside the commit region) and the distance-3 rotated
surface code of stim.Circuit.generated, whose four-round circuit holds
4, 8, 8 and 12 detectors in rounds 1 to 4 (test_detector_chronology).
"""

import pytest
import stim

from decsim.detector_error_model import detector_chronology, window_placement


def surface_code_detectors_by_round(rounds):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=0.001,
    )
    round_by_detector = detector_chronology.resolve_detector_rounds(
        circuit, None, rounds
    )
    return detector_chronology.detectors_by_round(round_by_detector)


def test_a_four_bound_entry_is_returned_as_its_four_ints():
    bounds = window_placement.parse_window_entry((1, 2, 3, 4))
    assert bounds == (1, 2, 3, 4)


def test_a_three_bound_entry_starts_its_buffer_at_its_first_commit_round():
    bounds = window_placement.parse_window_entry((2, 3, 4))
    assert bounds == (2, 2, 3, 4)


def test_a_windows_rows_are_the_detectors_of_rounds_two_and_three():
    detectors_by_round = surface_code_detectors_by_round(4)
    rows = window_placement.detectors_in_window(detectors_by_round, 2, 3)
    assert rows == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_ranges_given_as_a_list_of_lists_come_back_as_a_tuple_of_int_pairs():
    checked = window_placement.checked_fault_exclusion_ranges(
        [[1, 1], [2, 3]], 4
    )
    assert checked == ((1, 1), (2, 3))
    assert type(checked) is tuple
    assert type(checked[0]) is tuple
    assert type(checked[1]) is tuple
    assert type(checked[0][0]) is int
    assert type(checked[1][1]) is int


def test_an_empty_str_is_not_an_empty_sequence_of_ranges():
    with pytest.raises(
        ValueError, match="must be a sequence of ranges, got ''"
    ):
        window_placement.checked_fault_exclusion_ranges("", 4)


def test_a_bytes_container_is_refused():
    with pytest.raises(
        ValueError, match="must be a sequence of ranges, got b''"
    ):
        window_placement.checked_fault_exclusion_ranges(b"", 4)


def test_a_str_pair_is_refused():
    with pytest.raises(
        ValueError, match="must be a pair of built-in integers, got '12'"
    ):
        window_placement.checked_fault_exclusion_ranges(("12",), 4)


def test_a_bytes_pair_is_refused_although_its_values_are_ints():
    with pytest.raises(
        ValueError,
        match=r"must be a pair of built-in integers, got b'\\x01\\x01'",
    ):
        window_placement.checked_fault_exclusion_ranges((b"\x01\x01",), 4)
