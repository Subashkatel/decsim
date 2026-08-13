"""ExecutionOrchestrator prediction retention and timing-route behavior."""

from dataclasses import fields

from decsim.engine import Engine
from decsim.message import Decision, DecodeResult, Operation
from decsim.orchestrators import ExecutionOrchestrator


def _orchestrator():
    return ExecutionOrchestrator(Engine(verbose=False))


def test_decision_contains_only_target_and_route():
    assert [field.name for field in fields(Decision)] == [
        "target_operation_id",
        "releases_operation",
    ]


def test_timing_only_result_return_preserves_route_without_effect_state():
    orchestrator = _orchestrator()
    operation = Operation(
        0,
        "T",
        (0,),
        clifford=False,
        requires_result_return_to_qpu=True,
    )

    decisions = orchestrator.on_result(
        operation,
        DecodeResult(0, -1, logical_observables=None),
    )

    assert decisions == [Decision(0, releases_operation=False)]
    assert orchestrator.history[-1]["logical_observables"] is None
    assert not hasattr(orchestrator, "frame")


def test_complete_functional_vector_returns_without_scalar_interpretation():
    orchestrator = _orchestrator()
    operation = Operation(
        0,
        "measurement",
        (0,),
        requires_result_return_to_qpu=True,
    )

    decisions = orchestrator.on_result(
        operation,
        DecodeResult(0, -1, logical_observables=(1, 0, 1)),
    )

    assert decisions == [Decision(0, releases_operation=False)]
    assert orchestrator.history[-1]["logical_observables"] == (1, 0, 1)


def test_blocking_result_releases_every_successor_and_preserves_vector():
    orchestrator = _orchestrator()
    operation = Operation(1, "measurement", (0,))
    orchestrator.register_blocked_operation(2, 1)
    orchestrator.register_blocked_operation(3, 1)

    decisions = orchestrator.on_result(
        operation,
        DecodeResult(1, 0, logical_observables=(1, 0, 1)),
    )

    assert decisions == [Decision(2), Decision(3)]
    assert orchestrator.blocked_by_index == {}
    assert orchestrator.history[-1]["logical_observables"] == (1, 0, 1)


def test_prediction_only_operation_records_complete_vector_without_instruction():
    orchestrator = _orchestrator()
    operation = Operation(1, "multi", (0,))

    decisions = orchestrator.on_result(
        operation,
        DecodeResult(1, 0, logical_observables=(1, 0, 1)),
    )

    assert decisions == []
    assert orchestrator.history[-1]["logical_observables"] == (1, 0, 1)
    assert orchestrator.stats == {
        "outcomes": 1,
        "decisions": 0,
        "result_returns": 0,
    }


def test_integrate_dispatches_route_decisions_through_controller():
    class RecordingController:
        def __init__(self):
            self.decisions = []

        def relay_instruction(self, decision, sink):
            self.decisions.append(decision)
            sink(decision)

    orchestrator = _orchestrator()
    controller = RecordingController()
    delivered = []
    orchestrator.connect(controller, delivered.append)
    operation = Operation(1, "measurement", (0,))
    orchestrator.register_blocked_operation(2, 1)

    orchestrator.integrate(
        operation,
        DecodeResult(1, 0, logical_observables=(1, 0)),
    )

    assert controller.decisions == delivered == [Decision(2)]
