"""ExecutionOrchestrator functional prediction and timing-progress behavior."""
import pytest

from decsim.engine import Engine
from decsim.orchestrators import ExecutionOrchestrator
from decsim.message import DecodeResult, IntrinsicMeasurement, Operation


def _fb():
    return ExecutionOrchestrator(Engine(verbose=False))


def test_timing_only_result_return_fabricates_no_effect():
    fb = _fb()
    op = Operation(0, "T", (0,), clifford=False, requires_result_return_to_chip=True)
    decisions = fb.on_result(
        op,
        DecodeResult(0, -1, logical_observables=None),
    )

    assert decisions[0].effect is None
    assert decisions[0].releases_operation is False
    assert fb.frame.snapshot() == {}


def test_clifford_updates_frame_no_instruction():
    fb = _fb()
    op = Operation(0, "CNOT", (2, 3), clifford=True)
    assert fb.on_result(
        op,
        DecodeResult(0, 0, logical_observables=(1,)),
    ) == []
    assert fb.frame.x_of(2) == 1                     # frame XOR on first qubit
    assert fb.stats["frame_updates"] == 1


def test_nonclifford_releases_blocked_ops_with_byproduct():
    fb = _fb()
    blocker = Operation(1, "T", (0,), clifford=False, byproduct_pauli="X",
                        requires_strong_commit=True,
                        intrinsic_measurement=IntrinsicMeasurement(
                            operation_id=1,
                            trajectory_id=1,
                            value=1,
                            source="controlled fixture",
                        ))
    fb.register_blocked_operation(blocked_op_id=2, blocking_op_id=1)
    decisions = fb.on_result(
        blocker,
        DecodeResult(1, 0, logical_observables=(1,)),
    )
    d = decisions[0]
    # measurement = decoded 1 XOR intrinsic 1 = 0 -> identity byproduct, basis Z
    assert d.target_operation_id == 2
    assert (
        d.effect.basis,
        d.effect.pauli,
        d.effect.apply_s,
        d.effect.correction_value,
    ) == ("Z", "I", False, 0)
    assert d.releases_operation and d.strong_committed is True
    assert fb.blocked_by_index == {}                 # popped


def test_measurement_one_gives_byproduct_and_s():
    fb = _fb()
    blocker = Operation(
        1,
        "T",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        byproduct_pauli="X",
    )
    fb.register_blocked_operation(2, 1)
    d = fb.on_result(
        blocker,
        DecodeResult(1, 0, logical_observables=(1,)),
    )[0]
    assert (
        d.effect.basis,
        d.effect.pauli,
        d.effect.apply_s,
        d.effect.correction_value,
    ) == ("X", "X", True, 1)


def test_timing_only_clifford_blocker_releases_without_frame_effect():
    fb = _fb()
    blocker = Operation(1, "MZ", (0,), clifford=True)
    fb.register_blocked_operation(2, 1)

    decisions = fb.on_result(
        blocker,
        DecodeResult(1, 0, logical_observables=None),
    )

    assert decisions == [
        decisions[0],
    ]
    assert decisions[0].target_operation_id == 2
    assert decisions[0].effect is None
    assert fb.frame.snapshot() == {}


def test_functional_clifford_blocker_releases_before_frame_update():
    fb = _fb()
    blocker = Operation(1, "MZ", (0,), clifford=True)
    fb.register_blocked_operation(2, 1)

    decision = fb.on_result(
        blocker,
        DecodeResult(1, 0, logical_observables=(1,)),
    )[0]

    assert decision.effect.decoded_value == 1
    assert fb.frame.snapshot() == {}
    assert fb.stats["frame_updates"] == 0


def test_prediction_only_operation_preserves_multi_observable_vector():
    fb = _fb()
    op = Operation(1, "multi", (0,), clifford=False,
                   consumes_magic_state=False)

    assert fb.on_result(
        op,
        DecodeResult(1, 0, logical_observables=(1, 0, 1)),
    ) == []

    assert fb.history[-1]["logical_observables"] == (1, 0, 1)
    assert fb.history[-1]["selected_observable_index"] is None


def test_scalar_consumer_requires_index_for_multi_observable_vector():
    fb = _fb()
    op = Operation(1, "multi", (0,), clifford=True)

    with pytest.raises(ValueError, match="logical_observable_index"):
        fb.on_result(
            op,
            DecodeResult(1, 0, logical_observables=(1, 0)),
        )


def test_configured_observable_index_drives_the_effect():
    fb = _fb()
    blocker = Operation(
        1,
        "multi",
        (0,),
        clifford=False,
        consumes_magic_state=False,
        logical_observable_index=1,
    )
    fb.register_blocked_operation(2, 1)

    decision = fb.on_result(
        blocker,
        DecodeResult(1, 0, logical_observables=(1, 0)),
    )[0]

    assert decision.effect.logical_observable_index == 1
    assert decision.effect.decoded_value == 0


def test_functional_magic_feedback_requires_intrinsic_provenance():
    fb = _fb()
    blocker = Operation(1, "T", (0,), clifford=False)
    fb.register_blocked_operation(2, 1)

    with pytest.raises(ValueError, match="intrinsic_measurement"):
        fb.on_result(
            blocker,
            DecodeResult(1, 0, logical_observables=(1,)),
        )
