import pytest

from decsim.engine import Engine, SimulationFailed
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
    engine._start_running()

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
    engine._start_running()

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

    with pytest.raises((TypeError, ValueError)):
        engine.schedule(-1, lambda: None)
    with pytest.raises((TypeError, ValueError)):
        engine.schedule(0.5, lambda: None)

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
    engine._start_running()

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
    engine._start_running()

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
    engine._start_running()
    engine.now = 2

    with pytest.raises((RuntimeError, ValueError)):
        engine.run()

    assert fired == []
    assert engine.now == 2
    assert engine._event_queue == []


def test_lifecycle_requires_quiescence_and_ordered_transitions():
    """Construction, running, finalization, and completion occur in order at quiescence."""
    engine = Engine(verbose=False)
    fired = []
    metric = ProbeMetric(name="setup")

    assert engine._phase == "construction"
    engine.schedule(0, lambda: fired.append(True))
    engine.add_metric(metric)
    with pytest.raises(RuntimeError):
        engine.run()

    engine._start_running()
    with pytest.raises(RuntimeError):
        engine._begin_finalization()
    with pytest.raises(RuntimeError):
        engine._complete()

    engine.run()
    engine._begin_finalization()
    assert engine._phase == "finalizing"
    engine._complete()

    assert fired == [True]
    assert engine._phase == "completed"


def test_terminal_phases_block_schedule_registration_and_run():
    """Finalizing and completed engines reject scheduling, metric registration, and runs."""
    engine = Engine(verbose=False)
    engine._start_running()
    engine.run()
    engine._begin_finalization()

    for phase in ("finalizing", "completed"):
        assert engine._phase == phase
        with pytest.raises(RuntimeError):
            engine.schedule(0, lambda: None)
        with pytest.raises(RuntimeError):
            engine.add_metric(ProbeMetric(name=f"metric-{phase}"))
        with pytest.raises(RuntimeError):
            engine.run()
        if phase == "finalizing":
            engine._complete()


def test_invalidation_retains_first_cause_and_blocks_future_operations():
    """Invalidation keeps its first cause and chains it from later operation failures."""
    engine = Engine(verbose=False)
    first_cause = ValueError("first failure")
    engine._invalidate(first_cause)
    engine._invalidate(RuntimeError("later failure"))

    operations = [
        lambda: engine.schedule(0, lambda: None),
        lambda: engine.add_metric(ProbeMetric()),
        engine.run,
    ]
    for operation in operations:
        with pytest.raises(SimulationFailed) as caught:
            operation()
        assert caught.value.__cause__ is first_cause

    assert engine._phase == "invalid"
    assert engine._failure_cause is first_cause


def test_completed_engine_ignores_invalidation():
    """Invalidation leaves a completed engine and its failure cause unchanged."""
    engine = Engine(verbose=False)
    engine._start_running()
    engine.run()
    engine._begin_finalization()
    engine._complete()

    engine._invalidate(ValueError("too late"))

    assert engine._phase == "completed"
    assert engine._failure_cause is None


def test_event_exception_propagates_once_and_restores_action_state():
    """An action exception propagates unchanged without retry and clears the active flag."""
    engine = Engine(verbose=False)
    error = RuntimeError("action failed")
    calls = []

    def failing_action():
        calls.append(("failing", engine._event_action_in_progress))
        raise error

    engine.schedule(2, failing_action)
    engine.schedule(
        3,
        lambda: calls.append(("later", engine._event_action_in_progress)),
    )
    engine._start_running()

    with pytest.raises(RuntimeError) as caught:
        engine.run()

    assert caught.value is error
    assert calls == [("failing", True)]
    assert engine.now == 2
    assert engine._event_action_in_progress is False
    assert engine._phase == "running"
    assert len(engine._event_queue) == 1

    engine.run()
    assert calls == [("failing", True), ("later", True)]


def test_primary_build_invalidates_and_reraises_failure():
    """The primary build path invalidates its engine and reraises the original failure."""
    error = RuntimeError("build failed")
    captured_engines = []

    class FailingRunSpec(RunSpec):
        def _build_once(self, engine, root_seed):
            captured_engines.append(engine)
            raise error

    with pytest.raises(RuntimeError) as caught:
        FailingRunSpec().build(verbose=False)

    assert caught.value is error
    assert len(captured_engines) == 1
    assert captured_engines[0]._phase == "invalid"
    assert captured_engines[0]._failure_cause is error


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("", 1),
        ("\ud800", 1),
        (object(), 1),
        ("metric", 0),
        ("metric", True),
        ("metric", 1.0),
    ],
    ids=[
        "empty-name",
        "surrogate-name",
        "non-string-name",
        "zero-version",
        "boolean-version",
        "fractional-version",
    ],
)
def test_metric_registration_rejects_invalid_identity(name, version):
    """Metric registration requires a stable nonempty name and positive built-in version."""
    engine = Engine(verbose=False)
    metric = ProbeMetric(name=name, version=version)

    with pytest.raises((TypeError, ValueError)):
        engine.add_metric(metric)

    assert metric.observed_engines == []
    assert engine.metrics == []


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


def test_metric_registration_rejects_active_callback_boundaries():
    """Metric registration is rejected during event actions and metric callbacks."""
    action_engine = Engine(verbose=False)
    action_candidate = ProbeMetric(name="during-action")

    def event_action():
        with pytest.raises(RuntimeError):
            action_engine.add_metric(action_candidate)
        assert action_engine._event_action_in_progress is True

    action_engine.schedule(0, event_action)
    action_engine._start_running()
    action_engine.run()
    stable_candidate = ProbeMetric(name="stable-running-boundary")
    assert action_engine.add_metric(stable_candidate) is stable_candidate

    callback_engine = Engine(verbose=False)
    anchor = ProbeMetric(name="anchor")
    callback_candidate = ProbeMetric(name="during-callback")
    callback_engine.add_metric(anchor)

    def observation(current):
        with pytest.raises(RuntimeError):
            current.add_metric(callback_candidate)
        assert current._metric_callback_in_progress is True

    anchor.on_observe = observation
    callback_engine._start_running()
    callback_engine.run()

    assert action_candidate.observed_engines == []
    assert stable_candidate.observed_engines == [action_engine]
    assert callback_candidate.observed_engines == []


@pytest.mark.parametrize(
    ("attribute", "new_value"),
    [("name", "changed"), ("result_schema_version", 2)],
)
def test_metric_registration_rejects_identity_changes(attribute, new_value):
    """A metric that changes its identity during initial observation is not registered."""
    engine = Engine(verbose=False)
    metric = ProbeMetric(name="stable", version=1)
    metric.on_observe = lambda current: setattr(metric, attribute, new_value)

    with pytest.raises(RuntimeError):
        engine.add_metric(metric)

    assert engine.metrics == []
    assert engine._metric_callback_in_progress is False


def test_metric_callback_exception_propagates_and_restores_state():
    """A metric exception propagates unchanged and clears the metric callback flag."""
    engine = Engine(verbose=False)
    error = RuntimeError("observation failed")
    metric = ProbeMetric(name="failing")

    def fail_observation(current):
        assert current._metric_callback_in_progress is True
        raise error

    metric.on_observe = fail_observation

    with pytest.raises(RuntimeError) as caught:
        engine.add_metric(metric)

    assert caught.value is error
    assert engine._metric_callback_in_progress is False
    assert engine.metrics == []


def test_metric_callbacks_reject_actions_and_nested_callbacks():
    """Metric callbacks are rejected during actions and other metric callbacks."""
    engine = Engine(verbose=False)
    inner_calls = []

    def outer_callback():
        assert engine._metric_callback_in_progress is True
        with pytest.raises(RuntimeError):
            engine._invoke_metric_callback(
                lambda: inner_calls.append("nested"),
                callback_kind="nested",
            )

    engine._invoke_metric_callback(outer_callback, callback_kind="outer")
    assert engine._metric_callback_in_progress is False

    def event_action():
        assert engine._event_action_in_progress is True
        with pytest.raises(RuntimeError):
            engine._invoke_metric_callback(
                lambda: inner_calls.append("action"),
                callback_kind="during action",
            )

    engine.schedule(0, event_action)
    engine._start_running()
    engine.run()

    assert inner_calls == []


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
    engine._start_running()

    engine.run()

    assert observed == ["first", "second"]


def test_result_callback_runs_at_quiescent_finalization_boundary():
    """A metric result callback runs guarded after the queue is empty and sealed."""
    engine = Engine(verbose=False)
    metric = ProbeMetric(name="result")
    engine.add_metric(metric)
    engine.schedule(1, lambda: None)
    engine._start_running()
    with pytest.raises(RuntimeError):
        engine._begin_finalization()

    engine.run()
    engine._begin_finalization()

    def produce_result():
        assert engine._phase == "finalizing"
        assert engine._event_queue == []
        assert engine._metric_callback_in_progress is True
        return "complete"

    metric.on_result = produce_result
    result = engine._invoke_metric_callback(
        metric.result,
        callback_kind="result",
    )

    assert result == "complete"
    assert engine._metric_callback_in_progress is False


def test_metric_results_preserve_names_and_registration_order():
    """Metric results form a name-keyed dictionary in registration order."""
    engine = Engine(verbose=False)
    first = ProbeMetric(name="first")
    second = ProbeMetric(name="second")
    result_calls = []

    def first_result():
        result_calls.append(("first", engine._metric_callback_in_progress))
        return 11

    def second_result():
        result_calls.append(("second", engine._metric_callback_in_progress))
        return 22

    first.on_result = first_result
    second.on_result = second_result
    engine.add_metric(first)
    engine.add_metric(second)

    results = engine.metric_results()

    assert results == {"first": 11, "second": 22}
    assert list(results) == ["first", "second"]
    assert result_calls == [("first", True), ("second", True)]
    assert engine._phase == "construction"


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


def test_recursive_run_is_rejected_without_losing_action_state():
    """An action cannot recursively drain the engine or lose ownership of its active flag."""
    engine = Engine(verbose=False)
    observed = []

    def outer_action():
        observed.append(("outer-start", engine._event_action_in_progress))
        with pytest.raises(RuntimeError):
            engine.run()
        observed.append(("outer-end", engine._event_action_in_progress))

    engine.schedule(0, outer_action, priority=0)
    engine.schedule(
        0,
        lambda: observed.append(("later", engine._event_action_in_progress)),
        priority=1,
    )
    engine._start_running()

    engine.run()

    assert observed == [
        ("outer-start", True),
        ("outer-end", True),
        ("later", True),
    ]
    assert engine._event_action_in_progress is False
