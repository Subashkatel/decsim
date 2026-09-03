"""A plan entry and a window's rows, read off the distance-3 surface code.

Sources: Skoric et al. 2209.08552, section I.B (a window is a commit
region inside a decoding region) and the distance-3 rotated surface code
of stim.Circuit.generated, whose four-round circuit holds 4, 8, 8 and 12
detectors in rounds 1 to 4 (test_detector_chronology).
"""

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
