"""The parallel row's block A/B schedule and its two terminal branches.

Skoric et al. 2209.08552 section I.C: A blocks decode at the same time
and B blocks reconcile the seams between them, so the commit graph is
depth two rather than a chain, and the published construction fixes
ncom = nbuf = d. The two ways a stream can end are the row's own rules:
a tail of at most d rounds is absorbed into the preceding A commit, and
a tail of at most 3d is a terminal B with only its left A behind it.
"""

import pytest

import decsim.records.program as program_records
import decsim.windows.schemes.parallel as parallel_scheme


def _geometries(plan) -> list:
    laid_out = []
    for window in plan.windows:
        laid_out.append(
            (
                window.buffer_lo,
                window.commit_lo,
                window.commit_hi,
                window.buffer_hi,
            )
        )
    return laid_out


def test_the_first_a_block_commits_the_first_2d_rounds_and_reads_3d():
    row = parallel_scheme.ParallelWindowScheme()

    plan = row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=3)
    first = plan.windows[0]

    assert (first.buffer_lo, first.commit_lo) == (1, 1)
    assert first.commit_hi == 6
    assert first.buffer_hi == 9


def test_an_interior_b_block_commits_the_region_between_two_a_commits():
    """A B block reads nothing past what it commits and closes both ends."""
    row = parallel_scheme.ParallelWindowScheme()

    plan = row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=3)
    interior_b = plan.windows[1]

    assert (interior_b.buffer_lo, interior_b.commit_lo) == (7, 7)
    assert (interior_b.commit_hi, interior_b.buffer_hi) == (15, 15)
    assert interior_b.closed_temporal_boundaries is True
    assert (0, 1) in plan.internal_dependencies
    assert (2, 1) in plan.internal_dependencies


def test_a_tail_of_at_most_d_rounds_is_absorbed_into_the_preceding_a():
    """Terminal branch one: no window of its own for a short remainder."""
    row = parallel_scheme.ParallelWindowScheme()

    plan = row.plan_operation(1, 20, commit_round_count=3, buffer_round_count=3)

    assert _geometries(plan) == [
        (1, 1, 6, 9),
        (7, 7, 15, 15),
        (13, 16, 20, 20),
    ]


def test_a_tail_of_at_most_3d_rounds_is_a_terminal_b_with_one_predecessor():
    """Terminal branch two: a closed B at the physical boundary."""
    row = parallel_scheme.ParallelWindowScheme()

    plan = row.plan_operation(1, 22, commit_round_count=3, buffer_round_count=3)
    terminal_b = plan.windows[3]

    assert _geometries(plan) == [
        (1, 1, 6, 9),
        (7, 7, 15, 15),
        (13, 16, 18, 21),
        (19, 19, 22, 22),
    ]
    assert terminal_b.closed_temporal_boundaries is True
    assert plan.internal_dependencies == ((0, 1), (2, 1), (2, 3))


def test_the_commit_graph_is_depth_two_and_not_a_chain():
    row = parallel_scheme.ParallelWindowScheme()

    plan = row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=3)

    assert row.commits_in_one_serial_chain is False
    assert plan.entry_window_indices == (0, 2, 4, 6)
    assert plan.exit_window_indices == (1, 3, 5)


def test_unequal_commit_and_buffer_widths_are_refused():
    """Skoric's construction fixes ncom = nbuf = d."""
    row = parallel_scheme.ParallelWindowScheme()

    with pytest.raises(ValueError) as refusal:
        row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=4)

    assert "ncom = nbuf = d" in str(refusal.value)


def test_this_row_measures_the_buffer_against_the_wider_of_the_two_floors():
    """A blocks read on both sides, so the leading floor binds here too."""
    row = parallel_scheme.ParallelWindowScheme()
    thin_leading_only = _geometry(
        buffer_round_count=3, leading_floor=5, trailing_floor=3
    )

    with pytest.raises(ValueError) as refusal:
        row.validate_buffer(thin_leading_only)

    assert "two-sided buffering floor 5" in str(refusal.value)


def _geometry(*, buffer_round_count, leading_floor, trailing_floor):
    """A d=3 surface geometry with the buffer width and floors given."""
    return program_records.ResolvedCodeGeometry(
        code_name="rotated surface code (d=3)",
        distance=3,
        commit_round_count=3,
        buffer_round_count=buffer_round_count,
        minimum_leading_buffer_round_count=leading_floor,
        minimum_trailing_buffer_round_count=trailing_floor,
        one_patch_spatial_node_count=9,
        window_floor_justification=None,
    )
