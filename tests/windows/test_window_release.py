"""A stream lets its final windows go and decides as before (window_release.py).

The referent is the same stream run keeping every window: the result
each operation delivers, its scoring and the frame's correction must be
the same, while the machine keeps a bounded number of windows however
long the stream runs.
"""

import pytest

import decsim.machine as machine_module
import decsim.windows.window_release as window_release
import tests.stream_machines as stream_machines

SEED = 17
# about 300 windows: several lettings-go, with escalations among them
LONG_RUN = stream_machines.StreamRun(900, 0.5, 1, True)


def _built_and_run(folds: bool):
    graph = stream_machines.stream_graph(LONG_RUN, folds)
    machine = machine_module.Machine.build(graph, SEED)
    result = machine.run()
    return machine, result


def test_a_stream_that_lets_windows_go_delivers_the_same_results():
    _whole_machine, whole = _built_and_run(folds=False)

    _machine, lean = _built_and_run(folds=True)

    assert lean.operation_results == whole.operation_results
    assert lean.fully_done_ticks == whole.fully_done_ticks


def test_the_frame_folds_what_it_lets_go_into_the_same_correction():
    whole_machine, _whole = _built_and_run(folds=False)
    whole_frame = whole_machine.control.pauli_frame

    machine, _result = _built_and_run(folds=True)

    frame = machine.control.pauli_frame
    stream_id = stream_machines.STREAM_ID
    expected = whole_frame.frame_for_stream(stream_id)
    assert frame.frame_for_stream(stream_id) == expected


def test_a_long_stream_keeps_a_bounded_number_of_windows():
    machine, _result = _built_and_run(folds=True)

    planner = machine.windows.planner
    stream_id = stream_machines.STREAM_ID
    kept_bound = window_release.KEPT_FINAL_WINDOWS + window_release.LET_GO_EVERY
    assert planner.window_count_of(stream_id) > 4 * kept_bound
    assert len(planner.windows_by_key) <= kept_bound
    assert len(planner.models.model_by_window) <= kept_bound
    assert len(machine.windows.ledger.contributions) <= kept_bound


def test_reading_a_window_let_go_says_it_was_let_go():
    machine, _result = _built_and_run(folds=True)

    planner = machine.windows.planner
    first_key = (stream_machines.STREAM_ID, 0)
    with pytest.raises(KeyError, match="let it go"):
        planner.window_at(first_key)
