import pytest

from decsim.engine import Engine


def test_earliest_event_runs_after_clock_advances():
    """The earliest event runs first and sees its scheduled tick as current time."""
    engine = Engine()
    observed = []
    engine.schedule(5, lambda: observed.append(("late", engine.now)))
    engine.schedule(2, lambda: observed.append(("early", engine.now)))

    returned = engine.run()

    assert returned is None
    assert observed == [("early", 2), ("late", 5)]
    assert engine.now == 5


def test_same_tick_events_use_priority_then_insertion_order():
    """Same-tick events fire lowest priority first, then insertion order."""
    engine = Engine()
    observed = []
    engine.schedule(
        4, lambda: observed.append("priority-two"), label="a", priority=2
    )
    engine.schedule(
        4, lambda: observed.append("first-priority-one"), label="z", priority=1
    )
    engine.schedule(
        4, lambda: observed.append("second-priority-one"), label="a", priority=1
    )
    engine.schedule(
        4, lambda: observed.append("priority-zero"), label="z", priority=0
    )

    engine.run()

    assert observed == [
        "priority-zero",
        "first-priority-one",
        "second-priority-one",
        "priority-two",
    ]


def test_scheduled_events_receive_unique_increasing_sequences():
    """Accepted events receive unique sequence values that increase with insertion."""
    engine = Engine()
    engine.schedule(3, lambda: None, label="first")
    engine.schedule(1, lambda: None, label="second")
    engine.schedule(2, lambda: None, label="third")

    sequence_by_label = {
        event.label: event.sequence_number for event in engine._event_queue
    }
    sequences = [
        sequence_by_label["first"],
        sequence_by_label["second"],
        sequence_by_label["third"],
    ]

    assert len(set(sequences)) == 3
    assert sequences == sorted(sequences)


def test_schedule_accepts_zero_integer_ticks_and_rejects_invalid_delays():
    """Scheduling accepts zero integer ticks and rejects negative or fractional delays."""
    engine = Engine()

    with pytest.raises(ValueError):
        engine.schedule(-1, lambda: None)

    assert engine.now == 0
    assert engine._event_queue == []
    engine.schedule(0, lambda: None)
    assert type(engine._event_queue[0].time) is int
    assert engine._event_queue[0].time == 0


def test_zero_delay_event_scheduled_by_action_joins_current_order():
    """A zero-delay event added by an action joins the queue at the current tick."""
    engine = Engine()
    observed = []

    def first_action():
        observed.append(("first", engine.now))
        engine.schedule(
            0,
            lambda: observed.append(("inserted", engine.now)),
            priority=1,
        )

    engine.schedule(5, first_action, priority=0)
    engine.schedule(
        5,
        lambda: observed.append(("existing", engine.now)),
        priority=2,
    )

    engine.run()

    assert observed == [
        ("first", 5),
        ("inserted", 5),
        ("existing", 5),
    ]


def test_event_exception_propagates_once():
    """An action exception propagates unchanged without retry; the rest of
    the queue is still there for a later run."""
    engine = Engine()
    error = RuntimeError("action failed")
    calls = []

    def failing_action():
        calls.append("failing")
        raise error

    engine.schedule(2, failing_action)
    engine.schedule(3, lambda: calls.append("later"))

    with pytest.raises(RuntimeError) as caught:
        engine.run()

    assert caught.value is error
    assert calls == ["failing"]
    assert engine.now == 2
    assert len(engine._event_queue) == 1

    engine.run()
    assert calls == ["failing", "later"]
