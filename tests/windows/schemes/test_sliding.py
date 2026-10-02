"""The sliding row's two terminal policies, laid out by the row itself.

Skoric et al. 2209.08552 section I.B for the (W, F) construction; qLDPC's
SlidingWindowDecoder tail rule (qLDPC src/qldpc/decoders/sinter.py:
776-777) for the last window. The row's own docstring says
windows.terminal_policy is the one key it reads, so both branches are
pinned here on the row, not through the planner.
"""

import decsim.windows.schemes.sliding as sliding_scheme


def _geometries(plan) -> list:
    """Each window as (buffer_lo, commit_lo, commit_hi, buffer_hi)."""
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


def test_the_flush_policy_ends_the_last_window_at_the_streams_last_round():
    row = sliding_scheme.SlidingWindowScheme(terminal_policy="flush")

    plan = row.plan_operation(1, 20, commit_round_count=3, buffer_round_count=3)

    assert _geometries(plan) == [
        (1, 1, 3, 6),
        (4, 4, 6, 9),
        (7, 7, 9, 12),
        (10, 10, 12, 15),
        (13, 13, 20, 20),
    ]


def test_the_lookahead_policy_keeps_the_stride_and_reads_past_the_stream():
    """Every window strides F, so the last still reads past its commit."""
    row = sliding_scheme.SlidingWindowScheme(terminal_policy="lookahead")

    plan = row.plan_operation(1, 20, commit_round_count=3, buffer_round_count=3)

    assert _geometries(plan) == [
        (1, 1, 3, 6),
        (4, 4, 6, 9),
        (7, 7, 9, 12),
        (10, 10, 12, 15),
        (13, 13, 15, 18),
        (16, 16, 18, 21),
        (19, 19, 20, 23),
    ]


def test_only_the_lookahead_policy_declares_trailing_tail_context():
    """The fact the escalation policy reads, declared by the policy."""
    flush_row = sliding_scheme.SlidingWindowScheme(terminal_policy="flush")
    lookahead_row = sliding_scheme.SlidingWindowScheme(
        terminal_policy="lookahead"
    )

    assert flush_row.has_trailing_tail_context is False
    assert lookahead_row.has_trailing_tail_context is True


def test_every_window_chains_to_the_one_after_it():
    row = sliding_scheme.SlidingWindowScheme(terminal_policy="flush")

    plan = row.plan_operation(1, 20, commit_round_count=3, buffer_round_count=3)

    assert plan.internal_dependencies == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert plan.entry_window_indices == (0,)
    assert plan.exit_window_indices == (4,)
    assert row.commits_in_one_serial_chain is True
