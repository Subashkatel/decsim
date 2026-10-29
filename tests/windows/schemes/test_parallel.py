"""The parallel row's block A/B schedule and its two terminal branches.

Skoric et al. 2209.08552 section I.C: A blocks decode at the same time
and B blocks reconcile the seams between them, so the commit graph is
depth two rather than a chain, and the published construction fixes
ncom = nbuf = d. The two ways a stream can end are the row's own rules:
a tail of at most d rounds is absorbed into the preceding A commit, and
a tail of at most 3d is a terminal B with only its left A behind it. A
growing stream lays the plan's blocks as their commits begin, Skoric's
online decoder (lines 1042-1055), and its end clips them as the plan's
tail does, but for a tail of at most d rounds past an A, where it keeps
the B it laid.
"""

import pytest

import decsim.records.windows as window_records
import decsim.windows.schemes.parallel as parallel_scheme


def _geometries(plan) -> list:
    return _spans(plan.windows)


def _spans(windows) -> list:
    """Each window as (buffer_lo, commit_lo, commit_hi, buffer_hi)."""
    laid_out = []
    for window in windows:
        laid_out.append(
            (
                window.buffer_lo,
                window.commit_lo,
                window.commit_hi,
                window.buffer_hi,
            )
        )
    return laid_out


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

    with pytest.raises(ValueError, match="parallel A/B decoding requires"):
        row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=4)


def _laid(stream, highest_known_round: int) -> list:
    """The stream's blocks laid through the round, as the planner lays them."""
    windows = []
    for geometry in stream.begun(highest_known_round, None):
        window = window_records.Window(
            stream_id=1,
            window_index=len(windows),
            commit_lo=geometry.commit_lo,
            commit_hi=geometry.commit_hi,
            buffer_hi=geometry.buffer_hi,
            round_count=geometry.round_count,
            buffer_lo=geometry.buffer_lo,
            closed_temporal_boundaries=geometry.closed_temporal_boundaries,
        )
        windows.append(window)
    return windows


def _stream():
    layouts = parallel_scheme.ParallelStreamLayouts()
    return layouts.open_stream(
        1, None, commit_round_count=3, buffer_round_count=3
    )


def test_a_growing_stream_lays_the_plans_blocks_as_their_commits_begin():
    """Skoric's online decoder hands each block its rounds as they come."""
    stream = _stream()
    row = parallel_scheme.ParallelWindowScheme()

    first = _laid(stream, 15)
    later = stream.begun(16, None)

    plan = row.plan_operation(1, 40, commit_round_count=3, buffer_round_count=3)
    planned = _geometries(plan)
    assert _spans(first) == planned[:2]
    assert later == [plan.windows[2]]


def test_a_streams_end_clips_its_blocks_as_the_plans_tail_does():
    """A tail of more than d rounds past an A is the plan's terminal B."""
    stream = _stream()
    windows = _laid(stream, 22)
    row = parallel_scheme.ParallelWindowScheme()

    change = stream.close_tail(windows, 22)

    plan = row.plan_operation(1, 22, commit_round_count=3, buffer_round_count=3)
    assert _spans(windows) == _geometries(plan)
    assert change.released == (windows[3],)


def test_a_tail_of_at_most_d_rounds_past_an_a_is_a_short_terminal_b():
    """The plan absorbs it into the A; a stream keeps the B it laid."""
    stream = _stream()
    windows = _laid(stream, 20)

    change = stream.close_tail(windows, 20)

    assert _spans(windows) == [
        (1, 1, 6, 9),
        (7, 7, 15, 15),
        (13, 16, 18, 20),
        (19, 19, 20, 20),
    ]
    assert change.relaid == (windows[2], windows[3])
    assert change.released == (windows[3],)


def test_a_stream_ending_in_an_a_commit_ends_that_a_there():
    stream = _stream()
    windows = _laid(stream, 17)

    change = stream.close_tail(windows, 17)

    spans = _spans(windows)
    assert spans[-1] == (13, 16, 17, 17)
    assert change.released == ()


def test_a_b_block_counts_the_a_after_it_and_that_a_links_itself():
    """The B waits for both A blocks; an A waits for none."""
    stream = _stream()
    first_a, b_block, second_a = _laid(stream, 16)

    assert stream.readers_of(b_block) == (None,)
    assert stream.reads_of(second_a) == ((1, 1),)
    assert stream.readers_of(first_a) == ()
    assert stream.reads_of(first_a) == ()
    assert stream.reads_of(b_block) == ()


def test_a_segment_edge_on_a_parallel_stream_is_refused():
    stream = _stream()
    windows = _laid(stream, 10)

    with pytest.raises(ValueError, match="not implemented yet: parallel"):
        stream.cut_after(windows, 8)
