"""The region sliding row: its regions and its readiness.

Its windows match SWIPER's on SWIPER's own schedules (2412.05115;
swiper/window_builder.py build_windows: regions of d rounds;
swiper/window_manager.py SlidingWindowManager.process_round: the next
region as the buffer), checked in tests/windows/test_region_layout.py.
"""

import pytest

import decsim.records.windows as window_records
import decsim.windows.coupling_orders as coupling_orders
import decsim.windows.schemes.region_sliding as region_sliding


def readiness(local_rounds_arrived: int, spatial_reads_arrived: bool):
    return window_records.WindowReadiness(
        local_rounds_arrived=local_rounds_arrived,
        local_round_count=20,
        successors=(),
        memory_rounds_arrived=0,
        tail_closed=False,
        spatial_reads_arrived=spatial_reads_arrived,
    )


def window():
    return window_records.Window(
        stream_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=7,
        buffer_hi=14,
        round_count=14,
    )


def test_a_stream_is_laid_in_regions_each_reading_the_next():
    order = coupling_orders.CoordinateOrder()
    row = region_sliding.RegionSlidingScheme(order)

    plan = row.plan_operation(1, 20, commit_round_count=7, buffer_round_count=7)

    spans = [
        (each.commit_lo, each.commit_hi, each.buffer_hi)
        for each in plan.windows
    ]
    assert spans == [(1, 7, 14), (8, 14, 20), (15, 20, 20)]
    assert plan.internal_dependencies == ((0, 1), (1, 2))
    assert plan.exit_window_indices == (2,)


def test_a_window_waits_for_the_regions_it_reads_across():
    order = coupling_orders.CoordinateOrder()
    row = region_sliding.RegionSlidingScheme(order)
    merge = window()
    partner_missing = readiness(14, False)
    partner_in = readiness(14, True)

    waiting = row.data_complete(merge, readiness=partner_missing)
    complete = row.data_complete(merge, readiness=partner_in)

    assert waiting is False
    assert complete is True


def test_a_buffer_length_is_refused_since_the_buffer_is_a_region():
    with pytest.raises(ValueError, match="reads the whole next region"):
        region_sliding.RegionSlidingScheme.Settings(buffer_rounds=3)
