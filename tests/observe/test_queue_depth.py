"""The decode queue's depth over time, one sample per change.

A listener on the queue's depth_changed source; the samples are what the
gate pins as queue_log and what the switching study reads as the ready
queue's peak. Each pool's own peak is kept apart, the max backlog per
decoder instance DART-Q reports (2605.09142 lines 1101-1109).
"""

import decsim.observe.queue_depth as queue_depth


def test_a_queue_that_never_changed_has_no_sample_and_no_peak():
    log = queue_depth.QueueDepthLog()

    assert log.samples == []
    assert log.peak == 0


def test_one_sample_is_kept_per_change_with_its_tick():
    log = queue_depth.QueueDepthLog()

    log.depth_changed(1000, 1)
    log.depth_changed(2000, 2)
    log.depth_changed(3000, 0)

    assert log.samples == [(1000, 1), (2000, 2), (3000, 0)]


def test_the_peak_is_the_most_jobs_that_ever_waited_at_once():
    log = queue_depth.QueueDepthLog()

    log.depth_changed(1000, 1)
    log.depth_changed(2000, 5)
    log.depth_changed(3000, 2)

    assert log.peak == 5


def test_each_pools_peak_is_kept_on_its_own_and_is_no_sample():
    log = queue_depth.QueueDepthLog()

    log.pool_depth_changed(1000, "default", 2)
    log.pool_depth_changed(2000, "strong", 1)
    log.pool_depth_changed(3000, "default", 1)

    assert log.peak_by_pool == {"default": 2, "strong": 1}
    assert log.samples == []
