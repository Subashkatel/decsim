"""The default interaction's strong region, against the paper's rounds.

Toshio et al. 2510.25222 Sec. III C (text lines 1236-1246) and Fig. 12:
the strong region is r_strong = r_com + 2 r_buf rounds from the
escalated window's commit start, and the weak decoder resumes "after
r_com + r_buf rounds are subsequently stored", so the paper's restart
window begins at the round after the strong region and reads nothing
inside it. escalation.restart_reread_buffer_regions is that width in
buffer regions: 0 is the paper, 1 is decsim's forward window, which
reads one buffer region of the strong region as the restart window's
far boundary.
"""

import decsim.records.windows as window_records
import decsim.windows.window_interactions as window_interactions


def _weak_window_info(commit_lo: int) -> window_records.WindowInfo:
    """A d=3 sliding window: three committed rounds, three buffered."""
    commit_hi = commit_lo + 2
    buffer_hi = commit_lo + 5
    window = window_records.Window(
        operation_id=1,
        window_index=1,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
        round_count=6,
    )
    return window_records.WindowInfo.from_window(window)


def test_the_strong_region_is_commit_plus_two_buffers_from_the_commit_start():
    interaction = window_interactions.DefaultWindowInteraction(1)
    weak_window = _weak_window_info(1)

    plan = interaction.plan_strong_region(weak_window, [], 30)

    assert plan.commit_lo == 1
    assert plan.commit_hi == 9
    assert plan.context_lo == 1
    assert plan.context_hi == 12


def test_the_paper_restart_reads_no_round_of_the_strong_region():
    """Width 0: the restart begins at the round after the strong region.

    It shares no round with the strong region, so it owns the faults of
    the rounds it reads (Fig. 12).
    """
    interaction = window_interactions.DefaultWindowInteraction(0)
    weak_window = _weak_window_info(1)

    plan = interaction.plan_strong_region(weak_window, [], 30)

    assert plan.commit_hi == 9
    assert plan.restart_buffer_lo == 10
    assert (
        plan.restart_seam_fault_owner
        is window_records.SeamFaultOwner.RESTART_WINDOW
    )


def test_one_buffer_region_of_re_read_reaches_back_into_the_strong_region():
    """Width 1: the restart's buffer start is one buffer region back.

    The re-read rounds are read twice, and the strong region, which
    decoded them with both boundaries determined, keeps their faults.
    """
    interaction = window_interactions.DefaultWindowInteraction(1)
    weak_window = _weak_window_info(1)

    plan = interaction.plan_strong_region(weak_window, [], 30)

    assert plan.commit_hi == 9
    assert plan.restart_buffer_lo == 7
    assert (
        plan.restart_seam_fault_owner
        is window_records.SeamFaultOwner.STRONG_REGION
    )


def test_a_re_read_never_reaches_before_the_strong_regions_commit_start():
    """A strong region clipped by the operation's end keeps its own start."""
    interaction = window_interactions.DefaultWindowInteraction(1)
    weak_window = _weak_window_info(10)

    plan = interaction.plan_strong_region(weak_window, [], 30)

    assert plan.commit_lo == 10
    assert plan.commit_hi == 18
    assert plan.restart_buffer_lo == 16


def test_a_strong_region_at_the_operations_end_has_no_restart_window():
    interaction = window_interactions.DefaultWindowInteraction(0)
    weak_window = _weak_window_info(4)

    plan = interaction.plan_strong_region(weak_window, [], 12)

    assert plan.commit_hi == 12
    assert plan.restart_buffer_lo is None
    assert plan.restart_seam_fault_owner is None
