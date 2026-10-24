"""Windows binned by the weak gap, against hand counts.

A row counts only while its chain's earlier kept answers all passed,
so a failed check is the window's own (experiments/answer_check_bins.py).
"""

import math

import decsim.experiments.answer_check_bins as answer_check_bins

EDGES = (0.0, 2.0, 4.0, math.inf)


def _row(gap, weak_failed, strong_failed, kept="weak", escalates=False):
    return {
        "weak_gap": gap,
        "weak_failed": weak_failed,
        "strong_failed": strong_failed,
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


def test_rows_after_a_kept_answer_that_failed_are_left_out():
    """The second row's kept weak answer fails; the third would count both."""
    stream = [
        _row(1.0, False, False),
        _row(1.0, True, True),
        _row(1.0, True, True),
    ]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert bins[0].windows == 2
    assert bins[0].both.rate == 0.5


def test_a_row_with_no_gap_counts_in_no_bin_and_keeps_the_rows_after_it():
    stream = [_row(None, False, False), _row(3.0, True, False)]

    bins = answer_check_bins.gap_bins([stream], EDGES)

    assert [gap_bin.windows for gap_bin in bins] == [0, 1, 0]
