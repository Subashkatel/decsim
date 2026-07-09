"""Kernel domain model: typed records, PauliFrame purity, no rollback fields."""
import pytest

from decsim.message import (Decision, DecodeJob, DecodeOutcome,
                            DecodeResult, OpKind, Operation, ResourceClaim,
                            SyndromePayload, Window, WindowGraph)
from decsim.pauli_frame import PauliFrame


def test_syndrome_round_matches_todays_payload_shape():
    r = SyndromePayload(operation_id=1, patch_id=0, round_index=3, bits=b"\x01")
    assert (r.operation_id, r.patch_id, r.round_index) == (1, 0, 3)
    assert r.n_fragments == 1 and r.size_bits is None
    with pytest.raises(ValueError):
        SyndromePayload(operation_id=1, patch_id=0, round_index=1, n_fragments=0)


def test_decode_job_has_no_rollback_fields():
    job = DecodeJob(op_id=1, window_id=0, n_rounds=11)
    for banned in ("epoch", "attempt_id", "causal_scope", "output_finality"):
        assert not hasattr(job, banned)          # rollback state is not core
    assert job.attempt == 0


def test_window_start_round_and_graph():
    w = Window(op_id=0, k=1, commit_lo=4, commit_hi=6, buffer_hi=9, n_rounds=9,
               buffer_lo=1)
    assert w.start_round == 1
    w2 = Window(op_id=0, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)
    assert w2.start_round == 1                   # no leading buffer -> commit_lo
    g = WindowGraph()
    g.add_window(w2)
    g.add_window(w)
    g.wire_dep((0, 0), (0, 1))
    assert g.windows[(0, 1)].deps_remaining == 1
    assert (0, 1) in g.windows[(0, 0)].dependents


def test_operation_kind_and_magic_state_rule():
    t = Operation(1, "T(q0)", (0,), clifford=False)
    assert t.kind is OpKind.GENERIC and t.needs_magic_state
    m = Operation(2, "MZ(q0)", (0,), clifford=True, kind=OpKind.MEASURE)
    assert m.kind is OpKind.MEASURE and not m.needs_magic_state
    assert Operation(3, "x", (0,), clifford=False,
                     consumes_magic_state=False).needs_magic_state is False
    assert Operation(4, "CNOT", (0, 1), clifford=True).needs_magic_state is False


def test_outcome_claim_decision_shapes():
    out = DecodeOutcome(job=DecodeJob(1, 0, 11),
                        result=DecodeResult(op_id=1, window_id=0, logical_value=1))
    assert out.result.logical_value == 1
    c = ResourceClaim("qubits", frozenset({0, 1}))
    assert c.kind == "qubits"
    d = Decision(4, "Z")
    assert d.releases_operation and d.pauli == "I"


def test_pauli_frame_behavior():
    f = PauliFrame()
    f.apply_pauli(0, "X")
    assert f.measurement_flip(0, "Z") == 1 and f.measurement_flip(0, "X") == 0
    f.apply_s(0)                                  # X -> Y: z ^= x
    assert f.z_of(0) == 1
    assert f.fold(0, "MZ", raw_bit=0) == 1
    f.apply_pauli(0, "Y")                         # cancels x and z
    assert f.snapshot() == {}
    with pytest.raises(ValueError):
        f.apply_pauli(0, "Q")
    with pytest.raises(ValueError):
        f.measurement_flip(0, "W")
