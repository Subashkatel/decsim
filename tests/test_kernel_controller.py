"""Reaction gate kernel tests — each maps to Contract 3 rules."""
import pytest

from decsim.engine import Engine
from decsim.policies import Ignore, SeparateDecodeJobs, from_mode
from decsim.message import (
    Decision,
    ExecutionProgram,
    Operation,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    ResolvedPatchPlanning,
    ResourceClaim,
    SyndromePayload,
)
from decsim.controller import Controller
from decsim.execution_runtime import ExecutionRuntime

ROUND = 1_100_000

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
    def accept_idle_decode_demand(self, *, rounds, code, spatial_nodes, label):
        self.submit_decode(rounds, lambda: None, code, spatial_nodes, label)


class _Source:
    """Emit rounds on the gate's clock and call body-done after rounds_for."""
    def __init__(self, engine, cluster):
        self.engine, self.cluster = engine, cluster
    def connect_completion_receiver(self, receiver):
        self.completion_receiver = receiver
    def issue(self, command):
        op, round_ticks, total = command.operation, command.round_ticks, command.round_count
        def tick(i):
            if i < total:
                self.engine.schedule(round_ticks, lambda: tick(i + 1))
            else:
                self.completion_receiver(op)
        self.engine.schedule(round_ticks, lambda: tick(1))
    def emit_feedback_memory_round(self, operation_id, patch, round_index):
        self.cluster.on_memory_round(operation_id)
    def emit_idle_stream_round(self, operation, stream_id, global_round, patch):
        self.cluster.on_syndrome_arrival(
            SyndromePayload(stream_id, patch, global_round))


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
    geometry = ResolvedCodeGeometry(
        code_name="fake",
        distance=3,
        commit_round_count=3,
        buffer_round_count=3,
        minimum_leading_buffer_round_count=3,
        minimum_trailing_buffer_round_count=3,
        one_patch_spatial_node_count=9,
        buffer_floor_override_active=False,
    )
    source = _Source(eng, cluster)
    gate = Controller(eng, qpu=source, window_manager=cluster,
                        round_ticks=ROUND,
                        code_geometry=geometry,
                        resolved_operations=tuple(
                            ResolvedOperationPlanning(
                                operation_id=operation.id,
                                code_geometry=geometry,
                                round_count=4,
                                round_ticks=ROUND,
                                spatial_node_count=9,
                            )
                            for operation in ops
                        ),
                        resolved_patches=tuple(
                            ResolvedPatchPlanning(
                                patch_identity=patch,
                                code_geometry=geometry,
                                round_ticks=ROUND,
                                spatial_node_count=9,
                            )
                            for patch in {
                                patch
                                for operation in ops
                                for patch in (
                                    operation.patches
                                    if operation.patches
                                    else operation.qubits or (0,)
                                )
                            }
                        ),
                        idle_policy=idle_policy or Ignore(),
                        max_idle_rounds=max_idle,
                        gates_start_on_round_boundaries=boundaries)
    runtime = ExecutionRuntime(
        eng, controller=gate, factory=factory,
        resource_claims_by_operation_id={
            operation.id: tuple(cluster.layout.resources_for(operation))
            for operation in ops})
    gate.connect_runtime(runtime)
    source.connect_completion_receiver(gate._body_done)
    gate.load_program(ExecutionProgram(tuple(ops)))
    return eng, runtime, cluster, factory


def _blocked_pair(**succ_kw):
    a = Operation(0, "A:T(q0)", (0,), clifford=False)
    b = Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                  predecessors=(0,), decoder_boundary_predecessors=(0,),
                  **succ_kw)
    return [a, b]


def test_release_unconditional_on_decision_contract_3_4():
    ops = _blocked_pair()
    eng, gate, cluster, factory = _gate(ops)
    eng.run()                                    # A finishes; B blocked, idling
    assert 1 not in gate.started
    gate.on_decision(Decision(1))
    eng.run()
    assert 1 in gate.decode_released and 1 in gate.started   # weak Decision released it


def test_idle_cap_and_accounting_contract_3_5():
    eng, gate, cluster, _ = _gate(_blocked_pair(), max_idle=10)
    eng.run()
    assert gate.controller.idle_rounds_emitted == 10        # capped
    assert len(gate.controller.idle_cap_hits) == 1
    assert gate.controller.idle_cap_hits[0]["max_idle_rounds"] == 10
    assert len(cluster.memory) == 10             # relayed as memory rounds
    assert gate.idle_rounds_by_patch == {0: 10}


def test_default_cap_is_100d():
    eng, gate, cluster, _ = _gate(_blocked_pair())
    assert gate.controller.max_idle_rounds == 300


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
    assert gate.controller.idle_rounds_emitted == 2
    assert gate.op_start_time[1] == 6 * ROUND    # started same tick as release


def test_idle_attachment_consumed_at_begin_contract_3_6():
    eng, gate, cluster, _ = _gate(_blocked_pair())
    _deliver_decision_late(eng, gate, 7 * ROUND)
    eng.run()
    assert cluster.prepended == [(1, 3)]         # ticks at 5,6,7*ROUND -> 3 idle rounds
    assert gate.idle_rounds_by_patch == {}       # popped at begin


def test_magic_wait_overlaps_feedback_wait():
    """Op waits max(state_ready, feedback_release), not the sum."""
    a = Operation(0, "A", (0,), clifford=True)
    b = Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0,
                  predecessors=(0,), decoder_boundary_predecessors=(0,))
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
    with pytest.raises(RuntimeError, match="share qubits resource 0"):
        _gate([a, b])


def test_from_mode_validates():
    assert from_mode("ignore").mode == "ignore"
    with pytest.raises(ValueError):
        from_mode("bogus")


def test_release_records_arrival_and_starts_waiting_operation():
    eng, gate, _, _ = _gate(_blocked_pair())
    eng.run()

    gate.on_decision(Decision(1))

    assert 1 in gate.decode_released
    assert gate.decode_release_time[1] == eng.now
    assert 1 in gate.started


def test_nonreleasing_result_return_records_arrival_without_starting():
    op = Operation(
        0,
        "A",
        (0,),
        requires_result_return_to_qpu=True,
    )
    eng, gate, _, _ = _gate([op])
    eng.run()

    gate.on_decision(Decision(0, releases_operation=False))

    assert gate.result_return_time_by_operation[0] == eng.now
    assert gate.decode_release_time == {}
