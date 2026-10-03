"""The ticks of every operation's life, heard from the execution runtime.

The runtime keeps which operations have started, finished and been
released; the ticks live here, keyed by operation id. The one rule
beyond bookkeeping is that a QPU start replaces the issue stamp, because
the issue is when the operation was taken and the start is when its body
began.
"""

import decsim.observe.runtime_stamps as runtime_stamps


def test_the_qpu_start_replaces_the_issue_stamp():
    stamps = runtime_stamps.RuntimeStamps()

    stamps.operation_issued(1, 1000)
    stamps.operation_started(1, 4000)

    assert stamps.op_start[1] == 4000


def test_the_last_finish_is_the_latest_body_done_whatever_the_order():
    stamps = runtime_stamps.RuntimeStamps()

    stamps.body_finished(1, 9000)
    stamps.body_finished(2, 5000)

    assert stamps.last_finish == 9000
