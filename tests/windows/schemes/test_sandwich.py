"""The sandwich row: Tan's type-1 cores, one-layer type-2 seams, refusals.

Tan et al. 2209.09219 supplement S8: type-1 cores of width w = s + 2b
every s rounds with a one-layer type-2 seam between each adjacent pair.
The row's own refusals come from that schedule: a step below 2 leaves no
room for a seam and a buffer below 1 leaves the cores non-overlapping.
"""

import pytest

import decsim.records.windows as window_records
import decsim.windows.schemes.sandwich as sandwich_scheme


def test_a_core_commits_s_minus_one_rounds_and_the_seam_takes_the_other():
    """The zero-seam construction: the round left over is the type-2 seam."""
    row = sandwich_scheme.TanSandwichScheme()

    plan = row.plan_operation(1, 30, commit_round_count=5, buffer_round_count=2)
    interior_core = plan.windows[2]
    seam_before_it = plan.windows[1]

    assert (interior_core.commit_lo, interior_core.commit_hi) == (8, 11)
    assert (seam_before_it.commit_lo, seam_before_it.commit_hi) == (7, 7)


def test_a_core_reads_s_plus_two_buffers_and_a_seam_reads_one_layer():
    row = sandwich_scheme.TanSandwichScheme()

    plan = row.plan_operation(1, 30, commit_round_count=5, buffer_round_count=2)
    interior_core = plan.windows[2]
    seam = plan.windows[3]

    assert interior_core.buffer_lo == 6
    assert interior_core.buffer_hi == 14
    assert (seam.buffer_lo, seam.buffer_hi) == (12, 12)
    assert seam.closed_temporal_boundaries is True


def test_the_first_core_commits_from_round_one_and_the_last_to_the_end():
    """No seam precedes the first core and none follows the last."""
    row = sandwich_scheme.TanSandwichScheme()

    plan = row.plan_operation(1, 30, commit_round_count=5, buffer_round_count=2)

    assert (plan.windows[0].commit_lo, plan.windows[0].commit_hi) == (1, 6)
    assert (plan.windows[-1].commit_lo, plan.windows[-1].commit_hi) == (28, 30)


def test_each_seam_waits_on_the_core_on_either_side_of_it():
    row = sandwich_scheme.TanSandwichScheme()

    plan = row.plan_operation(1, 30, commit_round_count=5, buffer_round_count=2)

    assert plan.internal_dependencies == (
        (0, 1),
        (2, 1),
        (2, 3),
        (4, 3),
        (4, 5),
        (6, 5),
        (6, 7),
        (8, 7),
        (8, 9),
        (10, 9),
    )
    assert row.commits_in_one_serial_chain is False
    assert plan.protocol is (
        window_records.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE
    )


def test_a_step_below_two_is_refused():
    """A step of 1 leaves no round for a seam between adjacent cores."""
    row = sandwich_scheme.TanSandwichScheme()

    with pytest.raises(ValueError) as refusal:
        row.plan_operation(1, 30, commit_round_count=1, buffer_round_count=2)

    assert "needs a step of at least 2" in str(refusal.value)


def test_a_buffer_below_one_is_refused():
    """A buffer of 0 leaves the cores non-overlapping, not the schedule."""
    row = sandwich_scheme.TanSandwichScheme()

    with pytest.raises(ValueError) as refusal:
        row.plan_operation(1, 30, commit_round_count=5, buffer_round_count=0)

    assert "overlapping windows (b >= 1)" in str(refusal.value)
