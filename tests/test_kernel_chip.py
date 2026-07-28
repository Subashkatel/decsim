"""Reaction gate kernel tests — each maps to Contract 3 rules."""
import pytest

from decsim.engine import Engine
from decsim.policies import Ignore, SeparateDecodeJobs, from_mode
from decsim.message import (
    Decision,
    FeedbackEffect,
    Operation,
    ResourceClaim,
)
from decsim.chip import Chip

ROUND = 1_100_000


def _effect(*, decoded=0, correction=0, basis="Z", pauli="I",
            apply_s=False):
    return FeedbackEffect(
        logical_observable_index=0,
        decoded_value=decoded,
        intrinsic_measurement=None,
        correction_value=correction,
        basis=basis,
        pauli=pauli,
        apply_s=apply_s,
    )


class _Code:
    round_us = None
    name = "fake"
    def commit_rounds(self): return 3
    def buffer_rounds(self): return 3
    def spatial_nodes(self, patches): return 9


class _Layout:
    def code_for_op(self, op): return _Code()
    def code_for_patch(self, patch): return _Code()
    def resources_for(self, op):
        return [ResourceClaim("qubits", frozenset(op.qubits))]


class _Cluster:
    """Minimal cluster facade recording calls."""
    layout = _Layout()
    def __init__(self):
        self.registered, self.prepended, self.memory, self.decodes = [], [], [], []
    def register_op(self, op): self.registered.append(op.id)
    def rounds_for(self, op): return 4
    def prepend_idle_rounds(self, op_id, n): self.prepended.append((op_id, n))
    def on_memory_round(self, op_id): self.memory.append(op_id)
    def on_syndrome_arrival(self, payload): pass
    def close_stream_boundary(self, stream_id, n): pass
    def seal_stream(self, stream_id, n): pass
    def has_dynamic_stream(self, stream_id): return False
    def submit_decode(self, rounds, on_done, code, spatial_nodes, label):
        self.decodes.append((rounds, label))


class _Source:
    """Emit rounds on the gate's clock and call body-done after rounds_for."""
    def __init__(self, engine, cluster):
        self.engine, self.cluster = engine, cluster
    def start(self, op, round_ticks, on_body_done):
        total = self.cluster.rounds_for(op)
        def tick(i):
            if i < total:
                self.engine.schedule(round_ticks, lambda: tick(i + 1))
            else:
                on_body_done(op)
        self.engine.schedule(round_ticks, lambda: tick(1))


class _Controller:
    def relay_syndrome(self, payload, deliver):
        deliver(payload)


class _Factory:
    def __init__(self, delay_ticks=0, engine=None):
        self.delay, self.engine = delay_ticks, engine
        self.requests = []
    def request(self, op_id, callback):
        self.requests.append(op_id)
        if self.delay and self.engine:
            self.engine.schedule(self.delay, callback)
        else:
            callback()


def _gate(ops, *, idle_policy=None, max_idle=None, boundaries=False,
          factory_delay=0):
    eng = Engine(verbose=False)
    cluster = _Cluster()
    factory = _Factory(factory_delay, eng)
    gate = Chip(eng, source=_Source(eng, cluster),
                        controller=_Controller(), cluster=cluster,
                        factory=factory, round_ticks=ROUND, code_distance=3,
                        idle_policy=idle_policy or Ignore(),
                        round_ticks_by_operation_id={
                            operation.id: ROUND
                            for operation in ops
                        },
                        round_ticks_by_patch={
                            patch: ROUND
                            for operation in ops
                            for patch in (
                                operation.patches
                                if operation.patches
                                else operation.qubits or (0,)
                            )
                        },
                        resource_claims_by_operation_id={
                            operation.id: tuple(
                                cluster.layout.resources_for(operation)
                            )
                            for operation in ops
                        },
                        max_idle_rounds=max_idle,
                        gates_start_on_round_boundaries=boundaries)
    gate._load(ops)
    return eng, gate, cluster, factory


def _blocked_pair(**succ_kw):
    a = Operation(0, "A:T(q0)", (0,), clifford=False, has_successor=True)
    b = Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                  predecessors=(0,), **succ_kw)
    return [a, b]


def test_release_unconditional_on_decision_contract_3_4():
    ops = _blocked_pair(requires_strong_commit=True)
    eng, gate, cluster, factory = _gate(ops)
    eng.run()                                    # A finishes; B blocked, idling
    assert 1 not in gate.started
    gate.on_decision(Decision(
        1,
        _effect(decoded=1, correction=1, basis="X", pauli="X"),
        strong_committed=False,
    ))
    eng.run()
    assert 1 in gate.decode_released and 1 in gate.started   # weak Decision released it
    assert gate.frame.x_of(0) == 1               # byproduct folded into the frame


def test_idle_cap_and_accounting_contract_3_5():
    eng, gate, cluster, _ = _gate(_blocked_pair(), max_idle=10)
    eng.run()
    assert gate.idle_rounds_emitted == 10        # capped
    assert len(gate.idle_cap_hits) == 1
    assert gate.idle_cap_hits[0]["max_idle_rounds"] == 10
    assert len(cluster.memory) == 10             # relayed as memory rounds
    assert gate.idle_rounds_by_patch == {0: 10}


def test_default_cap_is_100d():
    eng, gate, cluster, _ = _gate(_blocked_pair())
    assert gate.max_idle_rounds == 300


def _deliver_decision_late(eng, gate, at_ticks, hop_ticks=ROUND // 2):
    """Model the real transport: the Decision event is scheduled by an earlier
    event (oc->cq hop), so its engine seq is HIGHER than an idle tick already
    scheduled for the same time (Contract 3 rule 7's real ordering)."""
    eng.schedule(at_ticks - hop_ticks,
                 lambda: eng.schedule(hop_ticks,
                                      lambda: gate.on_decision(Decision(1))))


def test_idle_tie_beats_release_contract_3_7():
    """A release landing exactly on an idle tick: idle fires first (lower seq
    — it was scheduled a full round earlier), then the release starts the op
    on the same tick; the emitter stops at its next tick."""
    eng, gate, cluster, _ = _gate(_blocked_pair())
    # A's body: 4 rounds -> body_done at 4*ROUND; idle ticks at 5,6,7*ROUND...
    _deliver_decision_late(eng, gate, 6 * ROUND)
    eng.run()
    # idle ticks fired at 5*ROUND and 6*ROUND (tie -> idle first), then stopped
    assert gate.idle_rounds_emitted == 2
    assert gate.op_start_time[1] == 6 * ROUND    # started same tick as release


def test_idle_attachment_consumed_at_begin_contract_3_6():
    eng, gate, cluster, _ = _gate(_blocked_pair())
    _deliver_decision_late(eng, gate, 7 * ROUND)
    eng.run()
    assert cluster.prepended == [(1, 3)]         # ticks at 5,6,7*ROUND -> 3 idle rounds
    assert gate.idle_rounds_by_patch == {}       # popped at begin


def test_magic_wait_overlaps_feedback_wait():
    """Op waits max(state_ready, feedback_release), not the sum."""
    a = Operation(0, "A", (0,), clifford=True, has_successor=True)
    b = Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                  predecessors=(0,))
    eng, gate, cluster, factory = _gate([a, b], factory_delay=20 * ROUND)
    _deliver_decision_late(eng, gate, 6 * ROUND)
    eng.run()
    # A: no magic state, body done 4R -> B requests its state then (ready 24R);
    # feedback release lands at 6R and OVERLAPS the state wait.
    assert factory.requests == [1]
    assert gate.op_start_time[1] == 24 * ROUND   # max(24R, 6R), not 24R + 6R


def test_round_boundary_start_snaps():
    eng, gate, cluster, _ = _gate(_blocked_pair(), boundaries=True)
    # release mid-round: at 5.5 rounds
    eng.schedule(int(5.5 * ROUND), lambda: gate.on_decision(Decision(1)))
    eng.run()
    assert gate.op_start_time[1] == 6 * ROUND    # snapped to next idle boundary


def test_separate_decode_jobs_submits_every_commit():
    eng, gate, cluster, _ = _gate(_blocked_pair(),
                                  idle_policy=SeparateDecodeJobs(), max_idle=7)
    eng.run()
    # commit_rounds=3: external decodes due at idle rounds 3 and 6
    assert [r for r, _ in cluster.decodes] == [6, 6]


def test_resource_conflict_without_edge_raises():
    a = Operation(0, "A", (0,))
    b = Operation(1, "B", (0,))                  # same qubit, no edge
    with pytest.raises(RuntimeError, match="share qubit 0"):
        _gate([a, b])


def test_from_mode_validates():
    assert from_mode("ignore").mode == "ignore"
    with pytest.raises(ValueError):
        from_mode("bogus")


def test_timing_only_release_changes_no_functional_state():
    eng, gate, _, _ = _gate(_blocked_pair())
    eng.run()
    frame_before = gate.frame.snapshot()

    gate.on_decision(Decision(1, effect=None))

    assert 1 in gate.decode_released
    assert gate.frame.snapshot() == frame_before
    assert gate.applied_basis == {}
    assert gate.applied_pauli == {}
    assert gate.applied_s == {}
    assert gate.applied_frame_delta == {}


def test_nonreleasing_functional_return_is_transport_only():
    op = Operation(
        0,
        "A",
        (0,),
        requires_result_return_to_chip=True,
    )
    eng, gate, _, _ = _gate([op])
    eng.run()
    effect = _effect(
        decoded=1,
        correction=1,
        basis="X",
        pauli="X",
        apply_s=True,
    )
    frame_before = gate.frame.snapshot()

    gate.on_decision(Decision(0, effect, releases_operation=False))

    assert gate.result_return_time_by_operation[0] == eng.now
    assert gate.frame.snapshot() == frame_before
    assert gate.decode_release_time == {}
    assert gate.applied_basis == {}
    assert gate.applied_pauli == {}
    assert gate.applied_s == {}
