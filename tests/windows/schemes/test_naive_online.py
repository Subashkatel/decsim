"""The naive_online row: one window over the whole operation.

No windowing at all: the operation's rounds are decoded as one batch
once they have arrived, which is the baseline every windowed scheme is
measured against. The row declares that it is not windowed and that idle
rounds ahead of the operation fold into its batch.
"""

import decsim.windows.schemes.naive_online as naive_online_scheme


def test_one_window_covers_the_whole_operation():
    row = naive_online_scheme.NaiveOnlineScheme()

    plan = row.plan_operation(1, 17, commit_round_count=3, buffer_round_count=3)
    only_window = plan.windows[0]

    assert len(plan.windows) == 1
    assert (only_window.buffer_lo, only_window.commit_lo) == (1, 1)
    assert (only_window.commit_hi, only_window.buffer_hi) == (17, 17)


def test_the_commit_and_buffer_sizes_are_ignored():
    """One batch has no stride, so neither size changes the layout."""
    row = naive_online_scheme.NaiveOnlineScheme()

    wide = row.plan_operation(
        1, 17, commit_round_count=11, buffer_round_count=7
    )
    narrow = row.plan_operation(
        1, 17, commit_round_count=1, buffer_round_count=1
    )

    assert wide.windows == narrow.windows


def test_the_row_declares_itself_unwindowed_and_batches_idle_rounds():
    row = naive_online_scheme.NaiveOnlineScheme()

    plan = row.plan_operation(1, 17, commit_round_count=3, buffer_round_count=3)

    assert plan.windowed is False
    assert plan.batch_preceding_idle_rounds is True
    assert plan.internal_dependencies == ()
    assert row.has_trailing_tail_context is False
    assert row.supports_dynamic_streams is False
