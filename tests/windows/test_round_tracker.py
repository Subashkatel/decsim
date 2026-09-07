"""The round tracker's laws: a window is complete when its rounds arrived.

The rule is the sliding window's (Skoric et al. 2209.08552, lines
194-203): a window reads through buffer_hi; a buffer past the
operation's end is satisfied by a successor's rounds, by memory rounds
or by a closed tail (the artificial defects at the commit edge carry
into the next window, lines 275-278).
"""

import types

import decsim.message as message
import decsim.records.windows as window_records
import decsim.windows.round_tracker as round_tracker
import decsim.windows.windowing_schemes as windowing_schemes


class _Planner:
    """The geometry the tracker reads: round counts and successors."""

    def __init__(self, round_counts: dict, successors: dict) -> None:
        self.round_counts = round_counts
        self.successors_by_operation = successors
        self.models = types.SimpleNamespace(
            check_stream_length=lambda _operation, _count: None
        )

    def round_count_of(self, operation_id) -> int:
        return self.round_counts[operation_id]


def _operation(operation_id, **fields) -> message.Operation:
    return message.Operation(operation_id, f"op{operation_id}", (0,), **fields)


def _window(
    operation_id, commit_lo, commit_hi, buffer_hi
) -> window_records.Window:
    round_count = buffer_hi - commit_lo + 1
    return window_records.Window(
        op_id=operation_id,
        k=0,
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
        n_rounds=round_count,
    )


def _tracker(
    round_counts: dict, successors: dict
) -> round_tracker.RoundTracker:
    planner = _Planner(round_counts, successors)
    scheme = windowing_schemes.SlidingWindowScheme()
    return round_tracker.RoundTracker(scheme, planner)


def test_a_window_is_complete_exactly_when_rounds_through_buffer_hi_arrived():
    tracker = _tracker({1: 9}, {1: []})
    operation_1 = _operation(1)
    tracker.register_operation(operation_1)
    window = _window(1, 1, 3, 5)
    tracker.note_arrival(1, 4)
    assert not tracker.is_data_complete(window)
    tracker.note_arrival(1, 5)
    assert tracker.is_data_complete(window)


def test_an_overflow_past_the_end_is_satisfied_by_a_successors_rounds():
    tracker = _tracker({1: 4, 2: 9}, {1: [2], 2: []})
    operation_1 = _operation(1)
    tracker.register_operation(operation_1)
    operation_2 = _operation(2)
    tracker.register_operation(operation_2)
    window = _window(1, 1, 3, 5)
    tracker.note_arrival(1, 4)
    assert not tracker.is_data_complete(window)
    tracker.note_arrival(2, 1)
    assert tracker.is_data_complete(window)


def test_an_overflow_past_the_end_is_satisfied_by_memory_rounds():
    tracker = _tracker({1: 4, 2: 9}, {1: [2], 2: []})
    operation_1 = _operation(1)
    tracker.register_operation(operation_1)
    operation_2 = _operation(2)
    tracker.register_operation(operation_2)
    window = _window(1, 1, 3, 5)
    tracker.note_arrival(1, 4)
    assert tracker.note_memory_round(1) == 1
    assert tracker.is_data_complete(window)


def test_a_closed_tail_completes_the_window_at_the_operations_end():
    blocked = _operation(2, blocked_by=1)
    source = _operation(1, feedback_boundary_mode="measurement_closed")
    tracker = _tracker({1: 4, 2: 9}, {1: [2], 2: []})
    tracker.register_operation(source)
    tracker.register_operation(blocked)
    window = _window(1, 1, 3, 5)
    tracker.note_arrival(1, 4)
    assert tracker.closed_boundary_round_for_window(window) == 4
    assert tracker.effective_round_count_for_window(1, window) == 4
    assert tracker.is_data_complete(window)


def test_a_stream_closed_boundary_in_the_buffer_is_the_earliest_one():
    tracker = _tracker({}, {"stream": []})
    operation_stream = _operation("stream")
    tracker.register_stream(operation_stream, None)
    for boundary in (2, 3, 4, 6, 7):
        tracker.close_boundary("stream", boundary)
    window = _window("stream", 1, 3, 7)
    assert tracker.closed_boundary_round_for_window(window) == 3


def test_a_finite_stream_refuses_a_boundary_inside_its_circuit():
    tracker = _tracker({}, {"stream": []})
    operation_stream = _operation("stream")
    tracker.register_stream(operation_stream, 5)
    try:
        tracker.close_boundary("stream", 4)
    except RuntimeError as error:
        assert "destructive boundary" in str(error)
    else:
        raise AssertionError("an internal boundary was accepted")
    tracker.close_boundary("stream", 6)


def test_the_round_limit_follows_the_streams_length_knowledge():
    tracker = _tracker({1: 11}, {})
    operation_1 = _operation(1)
    tracker.register_operation(operation_1)
    operation_open = _operation("open")
    tracker.register_stream(operation_open, None)
    operation_capped = _operation("capped")
    tracker.register_stream(operation_capped, 9)
    assert tracker.arrival_round_limit(1) == 11
    assert tracker.arrival_round_limit("open") is None
    assert tracker.arrival_round_limit("capped") == 9
    tracker.seal("open", 4)
    assert tracker.arrival_round_limit("open") == 4
    assert tracker.is_sealed("open")
    assert tracker.is_sealed(1)
    assert not tracker.is_sealed("capped")
    assert tracker.has_unsealed_streams()


def test_an_open_streams_round_count_is_the_windows_read_end_or_arrivals():
    tracker = _tracker({}, {"open": []})
    operation_open = _operation("open")
    tracker.register_stream(operation_open, None)
    tracker.note_arrival("open", 4)
    window = _window("open", 1, 3, 8)
    assert tracker.round_count_for_window("open", window) == 8
    assert tracker.round_count_for_window("open", None) == 4


def test_a_capped_stream_seals_when_its_limit_arrives():
    tracker = _tracker({}, {"capped": []})
    operation_capped = _operation("capped")
    tracker.register_stream(operation_capped, 5)
    tracker.note_arrival("capped", 4)
    assert not tracker.reaches_source_limit("capped")
    tracker.note_arrival("capped", 5)
    assert tracker.reaches_source_limit("capped")
    assert tracker.source_round_limit("capped") == 5


def test_registration_is_idempotent_for_a_known_operation():
    tracker = _tracker({1: 6}, {1: []})
    operation = _operation(1)
    assert tracker.register_operation(operation) is True
    tracker.note_arrival(1, 3)
    assert tracker.register_operation(operation) is False
    assert tracker.rounds_arrived(1) == 3
