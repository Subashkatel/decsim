#==================================================================
# TESTS FOR ENGINE
#==================================================================
import pytest
from decsim.engine import Engine, SimulationFailed


class _RecordingMetric:
    name = "recording"
    result_schema_version = 1

    def __init__(self):
        self.observed_ticks = []

    def observe(self, engine):
        self.observed_ticks.append(engine.now)

    def result(self):
        return list(self.observed_ticks)


class _EqualMetricName(str):
    """Wrong identity type that compares equal to the frozen metric name."""


def test_engine():
    eng = Engine(verbose=False)
    order = []
    eng.schedule(10, lambda: order.append(10), label="This is 10")
    eng.schedule(5, lambda: order.append(5), label="This is 5")
    eng.schedule(15, lambda: order.append(15), label="This is 15")
    eng.schedule(5, lambda: order.append(5.5), label="This is 5.5", priority=-1)
    eng.run()
    assert order == [5.5, 5, 10, 15]

def test_engine_log():
    eng = Engine(verbose=False)
    eng.schedule(10, lambda: eng.log("Test", "This is 10"), label="This is 10")
    eng.schedule(5, lambda: eng.log("Test", "This is 5"), label="This is 5")
    eng.schedule(15, lambda: eng.log("Test", "This is 15"), label="This is 15")
    eng.schedule(20, lambda: eng.log("Test", "This is 20"), label="This is 20")
    eng.run()
    assert len(eng.log_lines) == 4
    assert "This is 20" in eng.log_lines[3]
    assert "This is 5" in eng.log_lines[0]
    assert "This is 10" in eng.log_lines[1]
    assert "This is 15" in eng.log_lines[2]

def test_engine_schedule_past():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError):
        eng.schedule(-10, lambda: None, label="This is in the past")

def test_event_within_event():
    """Tests one event that schedules another future event."""
    eng = Engine(verbose=False)
    log = []

    def tick(n):
        log.append((eng.now, n))
        if n < 4:
            eng.schedule(delay=10, action=lambda: tick(n+1), label=f"Tick {n+1}")

    eng.schedule(delay=0, action=lambda: tick(1), label="Tick 1")
    eng.run()
    assert log == [(0, 1), (10, 2), (20, 3), (30, 4)]

def test_event_schedules_multiple_events():
    """Tests one event that schedules multiple future events."""
    eng = Engine(verbose = False)
    log = []
    def child(name):
        log.append((eng.now, name))

    def parent():
        log.append((eng.now, "parent"))
        eng.schedule(delay=10, action=lambda: child("child1"), label="Child 1")
        eng.schedule(delay=20, action=lambda: child("child2"), label="Child 2")
    eng.schedule(delay=0, action=parent, label="Parent")
    eng.run()
    assert log == [(0, "parent"), (10, "child1"), (20, "child2")]


def test_same_tick_fifo_order():
    eng = Engine(verbose=False)
    fired = []
    eng.schedule(3, lambda: fired.append(1))
    eng.schedule(3, lambda: fired.append(2))
    eng.schedule(3, lambda: fired.append(3))
    eng.run()
    assert fired == [1, 2, 3]        # same tick fires in insertion order


def test_invalid_engine_preserves_and_chains_its_first_failure():
    engine = Engine(verbose=False)
    first_failure = ValueError("first failure")
    engine._invalidate(first_failure)
    engine._invalidate(RuntimeError("later failure"))

    assert engine._failure_cause is first_failure
    with pytest.raises(SimulationFailed) as run_failure:
        engine.run()
    assert run_failure.value.__cause__ is first_failure
    with pytest.raises(SimulationFailed) as schedule_failure:
        engine.schedule(0, lambda: None)
    assert schedule_failure.value.__cause__ is first_failure


def test_metric_observes_registration_run_entry_and_clock_only_boundary():
    engine = Engine(verbose=False)
    metric = _RecordingMetric()
    engine.add_metric(metric)
    engine.schedule(20, lambda: None)

    engine.run(until=10)

    assert metric.observed_ticks == [0, 0, 10]


def test_run_rejects_a_time_limit_before_the_current_clock():
    engine = Engine(verbose=False)
    metric = _RecordingMetric()
    engine.add_metric(metric)
    fired = []
    engine.schedule(20, lambda: fired.append(engine.now))
    engine.run(until=10)
    observations_before_rejection = list(metric.observed_ticks)

    with pytest.raises(ValueError, match="before current simulation time"):
        engine.run(until=5)

    assert engine.now == 10
    assert fired == []
    assert metric.observed_ticks == observations_before_rejection

    engine.run(until=10)
    assert engine.now == 10
    assert fired == []

    engine.run()
    assert fired == [20]


def test_metric_registration_is_rejected_mid_action_before_observation():
    engine = Engine(verbose=False)
    attempted = _RecordingMetric()
    states = []

    def action():
        states.append("before")
        with pytest.raises(RuntimeError, match="stable boundary"):
            engine.add_metric(attempted)
        states.append("after")

    engine.schedule(0, action)
    engine.run()

    assert states == ["before", "after"]
    assert attempted.observed_ticks == []
    assert attempted not in engine.metrics


def test_nested_initial_registration_is_atomic_and_guard_cleans_up():
    engine = Engine(verbose=False)
    nested = _RecordingMetric()

    class Outer(_RecordingMetric):
        name = "outer"

        def observe(self, observed_engine):
            with pytest.raises(RuntimeError, match="stable boundary"):
                observed_engine.add_metric(nested)
            super().observe(observed_engine)

    outer = Outer()
    engine.add_metric(outer)
    later = _RecordingMetric()
    later.name = "later"
    engine.add_metric(later)

    assert engine.metrics == [outer, later]
    assert nested.observed_ticks == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "changed"),
        ("name", _EqualMetricName("recording")),
        ("result_schema_version", 2),
        ("result_schema_version", True),
        ("result_schema_version", 1.0),
    ],
)
def test_initial_identity_mutation_never_appends(field, value):
    engine = Engine(verbose=False)

    class Mutating(_RecordingMetric):
        def observe(self, observed_engine):
            setattr(self, field, value)

    metric = Mutating()
    with pytest.raises(RuntimeError, match="identity changed"):
        engine.add_metric(metric)
    assert engine.metrics == []
