"""ExecutionOrchestrator: verbatim orchestrator semantics (Contract 3 rule 3)."""
from decsim.engine import Engine
from decsim.orchestrators import ExecutionOrchestrator
from decsim.message import DecodeResult, Operation


def _fb():
    return ExecutionOrchestrator(Engine(verbose=False))


def test_none_logical_publishes_one_reviewed_constant():
    fb = _fb()
    op = Operation(0, "T", (0,), clifford=False, requires_result_return_to_chip=True)
    decisions = fb.on_result(op, DecodeResult(0, -1, logical_value=None))
    assert decisions[0].correction_value == 1        # None -> outcome 1, verbatim
    assert decisions[0].basis == "X"
    assert decisions[0].releases_operation is False


def test_clifford_updates_frame_no_instruction():
    fb = _fb()
    op = Operation(0, "CNOT", (2, 3), clifford=True)
    assert fb.on_result(op, DecodeResult(0, 0, logical_value=1)) == []
    assert fb.frame.x_of(2) == 1                     # frame XOR on first qubit
    assert fb.stats["frame_updates"] == 1


def test_nonclifford_releases_blocked_ops_with_byproduct():
    fb = _fb()
    blocker = Operation(1, "T", (0,), clifford=False, byproduct_pauli="X",
                        requires_strong_commit=True)
    fb.register_blocked_operation(blocked_op_id=2, blocking_op_id=1)
    fb.magic_measurements[1] = 1
    decisions = fb.on_result(blocker, DecodeResult(1, 0, logical_value=1))
    d = decisions[0]
    # measurement = decoded 1 XOR intrinsic 1 = 0 -> identity byproduct, basis Z
    assert (d.target_operation_id, d.basis, d.pauli, d.apply_s) == (2, "Z", "I", False)
    assert d.releases_operation and d.strong_committed is True
    assert fb.blocked_by_index == {}                 # popped


def test_measurement_one_gives_byproduct_and_s():
    fb = _fb()
    blocker = Operation(1, "T", (0,), clifford=False, byproduct_pauli="X")
    fb.register_blocked_operation(2, 1)
    d = fb.on_result(blocker, DecodeResult(1, 0, logical_value=1))[0]
    assert (d.basis, d.pauli, d.apply_s, d.correction_value) == ("X", "X", True, 1)
