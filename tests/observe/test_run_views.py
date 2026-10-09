"""A frozen view of a run's decode backlog.

The builder receives the owners it reads and writes nothing back, which
is what lets the backlog sampler take a view after every action without
changing what the run does.
"""

import types

import decsim.observe.run_views as run_views


class _DecoderManager:
    def __init__(self, pool: str, waiting: list) -> None:
        self.pool = types.SimpleNamespace(name=pool)
        self.queue = types.SimpleNamespace(waiting=waiting)


class _WindowManager:
    """The one call the backlog view makes on the window side."""

    def __init__(self, backlog_rounds) -> None:
        self.backlog_rounds = backlog_rounds
        self.rounds_read_out = None

    def backlog_round_counts(self, rounds_read_out_by_operation):
        self.rounds_read_out = rounds_read_out_by_operation
        return self.backlog_rounds


def test_the_default_pool_is_the_unnamed_lane_and_the_hosts_keeps_its_name():
    chip = _DecoderManager("default", ["a"])
    host = _DecoderManager("strong", ["b", "c"])
    decoders = (chip, host)
    windows = _WindowManager(())

    view = run_views.backlog_view(windows, decoders, {})

    assert view.per_lane == (("", 1), ("strong", 2))
    assert view.ready_jobs == 3


def test_the_rounds_are_summed_per_operation_per_patch_and_over_the_run():
    """The window side counts each operation's rounds from its readouts."""
    decoders = (_DecoderManager("default", []),)
    windows = _WindowManager(((1, "p0", 4), (2, "p0", 3), (3, "p1", 2)))
    rounds_read_out = {1: 6, 2: 3, 3: 8}

    view = run_views.backlog_view(windows, decoders, rounds_read_out)

    assert windows.rounds_read_out == rounds_read_out
    assert view.per_op_rounds == ((1, 4), (2, 3), (3, 2))
    assert view.per_patch_rounds == (("p0", 7), ("p1", 2))
    assert view.total_rounds == 9
