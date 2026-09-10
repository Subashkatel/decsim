"""Frozen views of a run's state: the backlog and the switching tables.

Each builder receives the owners it reads and writes nothing back, which
is what lets the backlog sampler take a view after every action without
changing what the run does.
"""

import decsim.observe.decode_records as decode_records
import decsim.observe.run_views as run_views
import decsim.observe.window_ledger as window_ledger_module


class _Queue:
    def __init__(self, waiting_by_pool: dict) -> None:
        self.waiting_by_pool = waiting_by_pool


class _DecoderManager:
    def __init__(self, waiting_by_pool: dict) -> None:
        self.queue = _Queue(waiting_by_pool)


class _WindowManager:
    """The one call the backlog view makes on the window side."""

    def __init__(self, backlog) -> None:
        self.backlog = backlog
        self.calls = 0

    def rounds_backlog(self):
        self.calls += 1
        return self.backlog


def test_an_idle_run_has_no_ready_job_and_no_waiting_round():
    decoders = _DecoderManager({"default": []})
    windows = _WindowManager(())

    view = run_views.backlog_view(windows, decoders)

    assert view.ready_jobs == 0
    assert view.per_lane == (("", 0),)
    assert view.total_rounds == 0


def test_the_default_pool_is_the_unnamed_lane_and_others_keep_their_name():
    decoders = _DecoderManager({"default": ["a"], "strong": ["b", "c"]})
    windows = _WindowManager(())

    view = run_views.backlog_view(windows, decoders)

    assert view.per_lane == (("", 1), ("strong", 2))
    assert view.ready_jobs == 3


def test_the_rounds_are_summed_per_operation_per_patch_and_over_the_run():
    decoders = _DecoderManager({"default": []})
    windows = _WindowManager(((1, "p0", 4), (2, "p0", 3), (3, "p1", 2)))

    view = run_views.backlog_view(windows, decoders)

    assert view.per_op_rounds == ((1, 4), (2, 3), (3, 2))
    assert view.per_patch_rounds == (("p0", 7), ("p1", 2))
    assert view.total_rounds == 9


def test_an_observer_that_only_wants_the_depths_skips_the_rounds_scan():
    decoders = _DecoderManager({"default": ["a"]})
    windows = _WindowManager(((1, "p0", 4),))

    view = run_views.backlog_view(windows, decoders, include_rounds=False)

    assert windows.calls == 0
    assert view.total_rounds == 0
    assert view.ready_jobs == 1


def test_the_switching_view_composes_the_three_tables_without_copying_timing():
    windows = window_ledger_module.WindowLedger()
    records = decode_records.DecodeRecordLedger()

    view = run_views.switching_records_view(windows, records)

    assert view.windows == ()
    assert view.requests == ()
    assert view.services == ()
