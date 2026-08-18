import pytest

from decsim.engine import Engine
from decsim.run_spec import RunSpec


class ProbeMetric:
    def __init__(self, name="metric", version=1):
        self.name = name
        self.result_schema_version = version
        self.observed_engines = []
        self.on_observe = None
        self.on_result = None

    def observe(self, engine):
        self.observed_engines.append(engine)
        if self.on_observe is not None:
            return self.on_observe(engine)
        return None

    def result(self):
        if self.on_result is not None:
            return self.on_result()
        return len(self.observed_engines)


def test_earliest_event_runs_after_clock_advances():
    """The earliest event runs first and sees its scheduled tick as current time."""
    engine = Engine(verbose=False)
    observed = []
    engine.schedule(5, lambda: observed.append(("late", engine.now)))
    engine.schedule(2, lambda: observed.append(("early", engine.now)))

    returned = engine.run()

    assert returned is None
    assert observed == [("early", 2), ("late", 5)]
    assert engine.now == 5


def test_same_tick_events_use_priority_then_insertion_order():
    """Same-tick events fire lowest priority first, then insertion order."""
    engine = Engine(verbose=False)
    observed = []
    engine.schedule(4, lambda: observed.append("priority-two"),
                    label="a", priority=2)
    engine.schedule(4, lambda: observed.append("first-priority-one"),
                    label="z", priority=1)
    engine.schedule(4, lambda: observed.append("second-priority-one"),
                    label="a", priority=1)
    engine.schedule(4, lambda: observed.append("priority-zero"),
                    label="z", priority=0)

    engine.run()

    assert observed == [
        "priority-zero",
        "first-priority-one",
        "second-priority-one",
        "priority-two",
    ]


def test_scheduled_events_receive_unique_increasing_sequences():
    """Accepted events receive unique sequence values that increase with insertion."""
    engine = Engine(verbose=False)
    engine.schedule(3, lambda: None, label="first")
    engine.schedule(1, lambda: None, label="second")
    engine.schedule(2, lambda: None, label="third")

    sequence_by_label = {
        event.label: event.seq for event in engine._event_queue
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
    engine = Engine(verbose=False)

    with pytest.raises(ValueError):
        engine.schedule(-1, lambda: None)

    assert engine.now == 0
    assert engine._event_queue == []
    engine.schedule(0, lambda: None)
    assert type(engine._event_queue[0].time) is int
    assert engine._event_queue[0].time == 0


def test_zero_delay_event_scheduled_by_action_joins_current_order():
    """A zero-delay event added by an action joins the queue at the current tick."""
    engine = Engine(verbose=False)
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


def test_run_observes_initial_and_successful_event_boundaries():
    """A drain observes metrics before events and after every successful action."""
    engine = Engine(verbose=False)
    metric = ProbeMetric(name="boundaries")
    boundaries = []
    metric.on_observe = lambda current: boundaries.append(
        (current.now, metric in current.metrics)
    )

    returned_metric = engine.add_metric(metric)
    engine.schedule(2, lambda: None)
    engine.schedule(3, lambda: None)

    returned = engine.run()

    assert returned_metric is metric
    assert returned is None
    assert boundaries == [
        (0, False),
        (0, True),
        (2, True),
        (3, True),
    ]
    assert engine._event_queue == []


def test_run_rejects_an_event_behind_the_current_clock():
    """A queued event behind the current clock is rejected without moving time backward."""
    engine = Engine(verbose=False)
    fired = []
    engine.schedule(1, lambda: fired.append(True))
    engine.now = 2

    with pytest.raises((RuntimeError, ValueError)):
        engine.run()

    assert fired == []
    assert engine.now == 2
    assert engine._event_queue == []


def test_event_exception_propagates_once():
    """An action exception propagates unchanged without retry; the rest of
    the queue is still there for a later run."""
    engine = Engine(verbose=False)
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


def test_metric_registration_rejects_duplicate_names():
    """Metric registration rejects a name that is already registered."""
    engine = Engine(verbose=False)
    registered = ProbeMetric(name="duplicate")
    duplicate = ProbeMetric(name="duplicate")
    engine.add_metric(registered)

    with pytest.raises(ValueError):
        engine.add_metric(duplicate)

    assert engine.metrics == [registered]
    assert duplicate.observed_engines == []


def test_metric_that_fails_its_first_observation_is_not_registered():
    """A metric exception propagates unchanged and the metric is not kept."""
    engine = Engine(verbose=False)
    error = RuntimeError("observation failed")
    metric = ProbeMetric(name="failing")

    def fail_observation(current):
        raise error

    metric.on_observe = fail_observation

    with pytest.raises(RuntimeError) as caught:
        engine.add_metric(metric)

    assert caught.value is error
    assert engine.metrics == []


def test_metric_observation_uses_registration_order_snapshot():
    """An observation pass uses one registration-order snapshot despite list mutation."""
    engine = Engine(verbose=False)
    first = ProbeMetric(name="first")
    second = ProbeMetric(name="second")
    added_during_observation = ProbeMetric(name="added")
    engine.add_metric(first)
    engine.add_metric(second)
    observed = []

    def observe_first(current):
        observed.append("first")
        current.metrics.append(added_during_observation)

    first.on_observe = observe_first
    second.on_observe = lambda current: observed.append("second")
    added_during_observation.on_observe = (
        lambda current: observed.append("added")
    )

    engine.run()

    assert observed == ["first", "second"]


def test_metric_results_preserve_names_and_registration_order():
    """Metric results form a name-keyed dictionary in registration order."""
    engine = Engine(verbose=False)
    first = ProbeMetric(name="first")
    second = ProbeMetric(name="second")
    result_calls = []

    def first_result():
        result_calls.append("first")
        return 11

    def second_result():
        result_calls.append("second")
        return 22

    first.on_result = first_result
    second.on_result = second_result
    engine.add_metric(first)
    engine.add_metric(second)

    results = engine.metric_results()

    assert results == {"first": 11, "second": 22}
    assert list(results) == ["first", "second"]
    assert result_calls == ["first", "second"]


def test_logging_stores_current_tick_and_prints_only_when_verbose(capsys):
    """Logging stores the current tick and caller text but prints only when verbose."""
    quiet_engine = Engine(verbose=False)
    quiet_engine.log("worker", "ready")
    first_line = quiet_engine.log_lines[-1]
    quiet_engine.now = 1_000_000
    quiet_engine.log("worker", "ready")
    second_line = quiet_engine.log_lines[-1]

    assert capsys.readouterr().out == ""
    assert first_line.endswith("] worker: ready")
    assert second_line.endswith("] worker: ready")
    assert first_line != second_line

    verbose_engine = Engine(verbose=True)
    verbose_engine.now = 1_000_000
    verbose_engine.log("worker", "ready")

    assert capsys.readouterr().out == verbose_engine.log_lines[-1] + "\n"


