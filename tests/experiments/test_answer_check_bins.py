"""Windows binned by the weak gap, against hand counts.

A row counts when its chain's kept answer on the window before passed,
and a one-sided row whose answers share a logical class counts as the
boundary's, not the class's (experiments/answer_check_bins.py).
"""

import math

import decsim.experiments.answer_check_bins as answer_check_bins

EDGES = (0.0, 2.0, 4.0, math.inf)


def _row(
    gap, weak_failed, strong_failed, kept="weak", escalates=False, same=False
):
    return {
        "weak_gap": gap,
        "weak_failed": weak_failed,
        "strong_failed": strong_failed,
        "same_logical_class": same,
        "kept": kept,
        "would_escalate": escalates,
    }


def test_each_bin_counts_its_windows_by_which_answer_failed():
    stream = [
        _row(1.0, True, False, kept="strong", escalates=True),
        _row(1.5, False, True, escalates=True),
        _row(3.0, False, False),
        _row(math.inf, False, False),
    ]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert [gap_bin.windows for gap_bin in bins] == [2, 1, 1]
    assert [gap_bin.escalated for gap_bin in bins] == [2, 0, 0]
    low = bins[0]
    assert (low.weak_only.rate, low.strong_only.rate) == (0.5, 0.5)
    assert low.both.rate is None
    assert bins[1].weak_only.rate is None


def test_a_row_whose_window_before_kept_a_failing_answer_is_left_out():
    """Row 2 follows a failed check; row 3 follows a passing one again."""
    stream = [
        _row(1.0, False, False),
        _row(1.0, True, True),
        _row(1.0, False, False),
        _row(1.0, True, False, kept="strong"),
    ]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert bins[0].windows == 3
    assert bins[0].both.rate == 1 / 3
    assert bins[0].weak_only.rate == 1 / 3


def test_a_row_with_no_gap_counts_in_no_bin_and_keeps_the_rows_after_it():
    stream = [_row(None, False, False), _row(3.0, True, False)]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert [gap_bin.windows for gap_bin in bins] == [0, 1, 0]


def test_a_one_sided_row_in_one_logical_class_counts_as_the_boundarys():
    """Rows 1 and 2 differ in class; rows 3 and 4 share one."""
    stream = [
        _row(1.0, True, False, kept="strong"),
        _row(1.0, False, True),
        _row(1.0, True, False, kept="strong", same=True),
        _row(1.0, False, True, same=True),
    ]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    low = bins[0]
    assert low.windows == 4
    assert (low.weak_only.rate, low.strong_only.rate) == (0.25, 0.25)
    assert low.boundary_decides.rate == 0.5


def test_a_row_with_no_logical_class_counts_in_no_bin():
    stream = [_row(1.0, True, False, kept="strong", same=None)]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert bins[0].windows == 0
