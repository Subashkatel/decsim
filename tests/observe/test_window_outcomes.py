"""Each window's answer beside its label, the parity of the errors it owns.

Source: Zhang et al. 2509.03815 Eq. (1) and (2), lines 497-508 and
576-580: a window's label is the parity, on the logical observable, of
the true errors in its own region, and the labels add up to the truth.
"""

import decsim.observe.window_outcomes as window_outcomes_module
import decsim.records.decoding as decoding_records
import decsim.records.sampled_errors as sampled_error_records


class _Operation:
    """The one field the listener reads off the operation that asked."""

    def __init__(self, operation_id: int) -> None:
        self.id = operation_id


def contribution(window_index, commit_lo, commit_hi, answer):
    return decoding_records.LogicalContribution(
        owner_key=(1, window_index),
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        ownership_kind="ordinary_window",
        logical_observables=answer,
    )


def fired(first_round, flip):
    return sampled_error_records.FiredError(
        first_round=first_round, logical_observables=(flip,)
    )


def test_a_window_is_labelled_by_the_errors_whose_first_round_it_commits():
    outcomes = window_outcomes_module.WindowOutcomes()
    errors = (fired(2, 1), fired(3, 1), fired(4, 1), fired(6, 0))
    operation = _Operation(1)
    outcomes.errors_sampled(operation, errors)
    tiles = (contribution(0, 1, 3, (0,)), contribution(1, 4, 6, (0,)))

    outcomes.contributions_delivered(1, tiles)
    first, second = outcomes.outcomes_by_operation[1]

    assert first.label == (0,)
    assert second.label == (1,)
    assert first.answer == (0,)


def test_a_timing_only_window_has_no_label():
    outcomes = window_outcomes_module.WindowOutcomes()
    operation = _Operation(1)
    errors = (fired(1, 1),)
    outcomes.errors_sampled(operation, errors)
    tiles = (contribution(0, 1, 3, None),)

    outcomes.contributions_delivered(1, tiles)
    only = outcomes.outcomes_by_operation[1][0]

    assert only.label is None


def test_a_source_that_kept_no_errors_leaves_the_answers_unlabelled():
    outcomes = window_outcomes_module.WindowOutcomes()
    tiles = (contribution(0, 1, 3, (1,)),)

    outcomes.contributions_delivered(1, tiles)
    only = outcomes.outcomes_by_operation[1][0]

    assert only.answer == (1,)
    assert only.label is None
