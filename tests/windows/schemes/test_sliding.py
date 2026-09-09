"""The sliding row's two terminal policies, laid out by the row itself.

Skoric et al. 2209.08552 section I.B for the (W, F) construction; Tan et
al. 2209.09219 lines 1029-1030 for the QUITS flush, which is also
qLDPC's SlidingWindowDecoder tail rule. The row's own docstring says
windows.terminal_policy is the one key it reads, so both branches are
pinned here on the row, not through the planner.
"""

import pytest

import decsim.records.program as program_records
import decsim.records.windows as window_records
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
    card = window_records.WindowingSchemeCard(terminal_policy="flush")
    row = sliding_scheme.SlidingWindowScheme(card)

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
    card = window_records.WindowingSchemeCard(terminal_policy="lookahead")
    row = sliding_scheme.SlidingWindowScheme(card)

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
    flush = window_records.WindowingSchemeCard(terminal_policy="flush")
    lookahead = window_records.WindowingSchemeCard(terminal_policy="lookahead")

    flush_row = sliding_scheme.SlidingWindowScheme(flush)
    lookahead_row = sliding_scheme.SlidingWindowScheme(lookahead)

    assert flush_row.has_trailing_tail_context is False
    assert lookahead_row.has_trailing_tail_context is True


def test_every_window_chains_to_the_one_after_it():
    card = window_records.WindowingSchemeCard(terminal_policy="flush")
    row = sliding_scheme.SlidingWindowScheme(card)

    plan = row.plan_operation(1, 20, commit_round_count=3, buffer_round_count=3)

    assert plan.internal_dependencies == ((0, 1), (1, 2), (2, 3), (3, 4))
    assert plan.entry_window_indices == (0,)
    assert plan.exit_window_indices == (4,)
    assert row.commits_in_one_serial_chain is True


def test_this_row_measures_the_buffer_against_the_trailing_floor_alone():
    """The sliding row reads past its commit only, so only that floor binds."""
    card = window_records.WindowingSchemeCard(terminal_policy="flush")
    row = sliding_scheme.SlidingWindowScheme(card)
    thin_leading_only = _geometry(
        buffer_round_count=3,
        leading_floor=5,
        trailing_floor=3,
    )

    row.validate_buffer(thin_leading_only)


def test_a_buffer_below_the_trailing_floor_is_refused_by_the_row():
    card = window_records.WindowingSchemeCard(terminal_policy="flush")
    row = sliding_scheme.SlidingWindowScheme(card)
    thin = _geometry(buffer_round_count=1, leading_floor=3, trailing_floor=3)

    with pytest.raises(ValueError) as refusal:
        row.validate_buffer(thin)

    assert "trailing buffering floor 3" in str(refusal.value)


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
