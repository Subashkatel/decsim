"""Boundary tests for typed operation-body commands and the QPU clock."""

from dataclasses import FrozenInstanceError

import pytest

from decsim.engine import Engine
from decsim.message import Operation, QPUReadout, SyndromePayload, WINDOW_INPUT_ROUTE
from decsim.qpu import QPUDevice, RunOperationBody


class _RecordingEgress:
    def __init__(self):
        self.received = []

    def accept_qpu_readout(self, payload, route):
        self.received.append((payload, route))


class _RecordingModel:
    def __init__(self, payloads_per_round=1):
        self.payloads_per_round = payloads_per_round
        self.begun = []
        self.sampled_rounds = []

    def begin_operation(self, operation, segment_round_count, source_round_count):
        self.begun.append(
            (operation, segment_round_count, source_round_count)
        )

    def idle_round_payloads(self, operation, stream_id, global_round, patch):
        return [QPUReadout(stream_id, patch, global_round)]

    def round_payloads(self, operation, round_index):
        self.sampled_rounds.append(round_index)
        return [
            QPUReadout(operation.id, patch_index, round_index)
            for patch_index in range(self.payloads_per_round)
        ]


def _command(operation, *, round_ticks=7, round_count=2, source_round_count=5):
    return RunOperationBody(
        operation=operation,
        round_ticks=round_ticks,
        round_count=round_count,
        source_round_count=source_round_count,
        emits_detector_data=operation.emits_detector_data,
        finalizes_stream_round=operation.finalizes_stream_round,
    )


def test_run_operation_body_is_immutable():
    command = _command(Operation(3, "memory", (0,)))

    with pytest.raises(FrozenInstanceError):
        command.round_count = 9


def test_qpu_rejects_fake_command_at_its_boundary():
    qpu = QPUDevice(Engine(), _RecordingModel(), _RecordingEgress(), lambda op: None)

    with pytest.raises(TypeError, match="RunOperationBody"):
        qpu.issue(object())


def test_qpu_command_carries_cadence_without_a_plan_table():
    engine = Engine()
    model = _RecordingModel(payloads_per_round=2)
    egress = _RecordingEgress()
    completed = []
    operation = Operation(3, "memory", (0, 1))
    qpu = QPUDevice(engine, model, egress, lambda op: completed.append((op, engine.now)))

    qpu.issue(_command(operation, round_ticks=7, round_count=2, source_round_count=5))
    engine.run()

    assert model.begun == [(operation, 2, 5)]
    assert model.sampled_rounds == [1, 2]
    assert [payload.n_fragments for payload, _ in egress.received] == [2, 2, 2, 2]
    assert [payload.fragment_index for payload, _ in egress.received] == [0, 1, 0, 1]
    assert all(route is WINDOW_INPUT_ROUTE for _, route in egress.received)
    assert completed == [(operation, 14)]
    assert not hasattr(qpu, "_round_count_by_operation_id")


def test_qpu_frames_copies_without_mutating_model_payloads():
    engine = Engine()
    model = _RecordingModel(payloads_per_round=2)
    source_payloads = []

    def round_payloads(operation, round_index):
        payloads = [
            QPUReadout(operation.id, "north", round_index),
            QPUReadout(operation.id, "south", round_index),
        ]
        source_payloads.extend(payloads)
        return payloads

    model.round_payloads = round_payloads
    egress = _RecordingEgress()
    qpu = QPUDevice(engine, model, egress, lambda op: None)

    qpu.issue(_command(Operation(3, "memory", (0, 1)), round_count=1))
    engine.run()

    assert [payload.n_fragments for payload in source_payloads] == [1, 1]
    assert [payload.n_fragments for payload, _ in egress.received] == [2, 2]
    assert all(
        relayed is not source
        for (relayed, _), source in zip(egress.received, source_payloads)
    )


def test_qpu_owns_idle_stream_payload_production_and_framing():
    egress = _RecordingEgress()
    qpu = QPUDevice(Engine(), _RecordingModel(), egress, lambda op: None)
    qpu.emit_idle_stream_round(Operation(4, "memory", (2,)), 9, 3, 2)
    payload, route = egress.received[0]
    assert (payload.operation_id, payload.round_index,
            payload.n_fragments, payload.fragment_index) == (9, 3, 1, 0)
    assert route == WINDOW_INPUT_ROUTE


def test_qpu_owns_feedback_hold_round_production():
    egress = _RecordingEgress()
    qpu = QPUDevice(Engine(), _RecordingModel(), egress, lambda op: None)
    qpu.emit_feedback_memory_round(7, "north", 5)
    payload, route = egress.received[0]
    assert payload.operation_id == ("idle", 7, "north")
    assert route.source_operation_id == 7


def test_qpu_rejects_binary_ingress_payload_that_bypasses_controller():
    qpu = QPUDevice(Engine(), _RecordingModel(), _RecordingEgress(), lambda op: None)
    with pytest.raises(TypeError, match="QPUReadout"):
        qpu._emit([SyndromePayload(1, 0, 1)], Operation(1, "memory", (0,)))


def test_qpu_rejects_missing_and_undersized_readout_sets():
    qpu = QPUDevice(Engine(), _RecordingModel(), _RecordingEgress(), lambda op: None)
    with pytest.raises(ValueError, match="at least one"):
        qpu._emit([], Operation(1, "memory", (0,)))
    operation = Operation(
        2, "memory", (0,), syndrome_fragment_index=0,
        syndrome_fragment_count=2)
    with pytest.raises(ValueError, match="one payload"):
        qpu._emit([QPUReadout(2, 0, 1), QPUReadout(2, 1, 1)], operation)


def test_run_rejects_incomplete_declared_fragment_set():
    from decsim.decoders import PerRoundDecoder
    from decsim.planner import FixedRounds
    from decsim.run_spec import RunSpec

    operation = Operation(
        7, "partial", (0,), syndrome_fragment_index=0,
        syndrome_fragment_count=2)
    with pytest.raises(RuntimeError, match="incomplete syndrome ingress"):
        RunSpec(
            ops=[operation],
            rounds_policy=FixedRounds(1),
            decoder=PerRoundDecoder(0.0),
        ).build(False)
