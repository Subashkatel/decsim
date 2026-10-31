"""A strong redo of a merge window reads the patches the merge couples.

A lattice-surgery merge joins its patches into one decoding problem, so
a window over it reads each coupled region too (SWIPER 2412.05115,
swiper/window_builder.py; windows/region_layout.py). A redo or a
revision decodes that window again, so it reads the same regions.
"""

import decsim.escalation.strong_regions as strong_regions
import decsim.records.windows as window_records

PARTNER = window_records.RoundSpan("b", 1, 6)


def merge_window(spatial_reads):
    """Window 0 of stream a: commits rounds 1-3 and reads to round 6."""
    spatial_round_count = sum(span.round_count for span in spatial_reads)
    round_count = 6 + spatial_round_count
    return window_records.Window(
        stream_id="a",
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=6,
        buffer_lo=1,
        round_count=round_count,
        spatial_reads=spatial_reads,
    )


class OneWindowPlan:
    def __init__(self, window):
        self.window = window

    def window_at(self, _key):
        return self.window


class OwnRoundsRetention:
    """Retention that reads a window's own stream and nothing before it."""

    def strong_rounds_before(self, _stream_id, _first_round, _last_round):
        return []

    def read_keys_for_bounds(self, stream_id, start_round, buffer_hi, _window):
        stop_round = buffer_hi + 1
        rounds = range(start_round, stop_round)
        return [(stream_id, round_index) for round_index in rounds]


def regions_over(window):
    regions = strong_regions.StrongRegions()
    regions.planner = OneWindowPlan(window)
    regions.retention = OwnRoundsRetention()
    return regions


def test_a_redo_of_a_merge_window_reads_the_coupled_region():
    weak_window = merge_window((PARTNER,))
    regions = regions_over(weak_window)

    redo = regions.redo_region(("a", 0))

    assert redo.window.spatial_reads == (PARTNER,)
    assert redo.window.round_count == 12
    assert set(PARTNER.round_keys) <= set(redo.context_read_keys)


def test_a_revision_of_a_merge_window_reads_the_coupled_region():
    weak_window = merge_window((PARTNER,))
    regions = regions_over(weak_window)
    redo = regions.redo_region(("a", 0))

    revision = regions.revision_region(redo.window)

    assert revision.window.spatial_reads == (PARTNER,)
    assert revision.context_read_keys == redo.context_read_keys


def test_a_redo_of_a_window_on_one_patch_reads_only_its_stream():
    weak_window = merge_window(())
    regions = regions_over(weak_window)

    redo = regions.redo_region(("a", 0))

    own_rounds = [("a", round_index) for round_index in range(1, 7)]
    assert redo.window.round_count == 6
    assert list(redo.context_read_keys) == own_rounds
