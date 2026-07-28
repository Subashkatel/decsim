"""Finality-gate tests (spec §8 test 4, plan Task 20 step 3).

Invariant #1 (spec §0.1): a non-Clifford feed-forward is steered only by a
finalized op result. Today's Eager parity behavior (Contract 3 rule 4) leaves
`requires_strong_commit` a marker — the release is unconditional — and the
NEW `gate_finalize(op_id, predicate)` seam is what enforces the fence when a
part opts in. All scenarios are real end-to-end simulations.
"""
import pytest

from decsim.policies import Held
from decsim.decoders import PerRoundDecoder, SampledConfidenceDecoder, SwitchingRouter
from decsim.frontends.circuit import CircuitFrontend
from decsim.message import Operation
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec
from decsim.switching import Switching


def _blocked_pair():
    """Non-Clifford A (marked requires_strong_commit) feeding blocked B."""
    return CircuitFrontend([
        Operation(0, "A:T(q0)", (0,), clifford=False,
                  requires_strong_commit=True),
        Operation(1, "B:T(q0)", (0,), clifford=False, blocked_by=0),
    ]).build()


def _decisions(completed_run):
    return [rec for rec in completed_run.orchestrator.history if rec["kind"] == "decision"]


def test_eager_parity_releases_on_weak_value():
    """Contract 3 rule 4: without a finalize gate, the release Decision fires
    from the weak value even for requires_strong_commit ops (marker only)."""
    completed_run = RunSpec(
        ops=_blocked_pair(),
        d=3,
        rounds_policy=FixedRounds(11),
        decoder=PerRoundDecoder(0.5),
        num_units=2,
    ).build()
    assert 1 in completed_run.chip.decode_release_time         # B released
    assert 0 not in completed_run.window_manager.op_strong_commit_time  # ... with no strong commit
    assert len(_decisions(completed_run)) == 1


def test_gate_finalize_holds_nonfinal_decision():
    """Held + gate_finalize: when nothing ever strong-commits A, A's result is
    never published and no non-Clifford Decision is emitted; the run winds
    down at the idle cap instead of steering B from a non-final frame."""
    captured = {}

    def configure(_engine, cluster, chip, _factory):
        runtime = cluster.window_manager
        captured["orchestrator"] = runtime.orchestrator
        captured["chip"] = chip
        runtime.gate_finalize(
            0,
            lambda op: op.id in runtime.op_strong_commit_time,
        )

    with pytest.raises(RuntimeError, match="workload completed"):
        RunSpec(
            ops=_blocked_pair(),
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PerRoundDecoder(0.5),
            num_units=2,
            boundary_policy=Held(),
            max_idle_rounds=20,
            make_metrics=lambda engine, cluster, chip, factory: (
                configure(engine, cluster, chip, factory) or []
            ),
        ).build()
    assert not captured["orchestrator"].history         # invariant #1 held
    assert 1 not in captured["chip"].decode_release_time
    assert captured["chip"].idle_cap_hits


def test_recheck_finalize_publishes_after_predicate_flips():
    """A part whose predicate flips outside commit/strong events re-drives the
    finish check via recheck_finalize and the held publication proceeds."""
    adopted = {"final": False}
    def configure(engine, cluster, _chip, _factory):
        runtime = cluster.window_manager
        runtime.gate_finalize(0, lambda op: adopted["final"])

        def make_final():
            adopted["final"] = True
            runtime.recheck_finalize(0)

        engine.schedule(20_000_000, make_final, label="test finality flip")

    completed_run = RunSpec(
        ops=_blocked_pair(),
        d=3,
        rounds_policy=FixedRounds(11),
        decoder=PerRoundDecoder(0.5),
        num_units=2,
        max_idle_rounds=20,
        make_metrics=lambda engine, cluster, chip, factory: (
            configure(engine, cluster, chip, factory) or []
        ),
    ).build()
    assert len(_decisions(completed_run)) == 1                 # published once, final
    assert 1 in completed_run.chip.decode_release_time         # B released afterwards


def test_gate_finalize_releases_after_strong_commit():
    """Escalating strategy: every weak decode escalates, the strong redo
    stamps op_strong_commit_time, the predicate turns true inside the normal
    strong-completion path, and the (now final) Decision releases B."""
    weak = SampledConfidenceDecoder(PerRoundDecoder(0.2), 1.0)
    def configure(_engine, cluster, _chip, _factory):
        runtime = cluster.window_manager
        runtime.gate_finalize(
            0,
            lambda op: op.id in runtime.op_strong_commit_time,
        )

    completed_run = RunSpec(
        ops=_blocked_pair(),
        d=3,
        rounds_policy=FixedRounds(11),
        strategy=Switching(confidence_threshold=0.5),
        router=SwitchingRouter(weak, PerRoundDecoder(3.0)),
        unit_pools={"default": 1, "strong": 1},
        boundary_policy=Held(),
        seed=7,
        make_metrics=lambda engine, cluster, chip, factory: (
            configure(engine, cluster, chip, factory) or []
        ),
    ).build()
    assert 0 in completed_run.window_manager.op_strong_commit_time
    assert 1 in completed_run.chip.decode_release_time
    assert (completed_run.chip.decode_release_time[1]
            > completed_run.window_manager.op_strong_commit_time[0])
    assert len(_decisions(completed_run)) == 1


def test_dynamic_window_created_during_held_boundary_still_depends_on_it():
    """Codex review finding (confirmed): a dynamic successor window created
    AFTER its predecessor weak-committed under a deferred (Held) boundary —
    strong redo still pending — must register the dependency and wait for
    the held handoff, not decode defect-free in the gap."""
    from types import MappingProxyType

    from decsim.message import (
        Operation as Op,
        OperationPlanningView,
        ResolvedOperationPlanning,
    )

    completed_run = RunSpec(ops=[Op(9, "M:mem(q9)", (9,), clifford=True)], d=3,
                    rounds_policy=FixedRounds(11),
                    num_units=1, decoder=PerRoundDecoder(0.2)).build()
    wm = completed_run.window_manager
    stream_op = Op(0, "S:stream(q0)", (0,), clifford=True)
    resolved = ResolvedOperationPlanning(
        operation_id=stream_op.id,
        code_geometry=wm._code_geometry,
        round_count=3,
        round_ticks=1,
        spatial_node_count=9,
    )
    wm._resolved_operations = MappingProxyType({
        **wm._resolved_operations,
        stream_op.id: resolved,
    })
    wm._planning_view_by_operation_id = MappingProxyType({
        **wm._planning_view_by_operation_id,
        stream_op.id: OperationPlanningView.from_operation(stream_op),
    })
    wm._register_dynamic_stream(stream_op, resolved)

    wm.create_dynamic_window(0, 0, 1, 3, 6, is_last=False)
    # W0 weak-commits while its strong redo is pending: committed, boundary HELD
    wm.committed_windows.add((0, 0))
    wm._held_boundary[(0, 0)] = (0, {4: [1]})

    wm.create_dynamic_window(0, 1, 4, 6, 9, is_last=False)
    w1 = wm.windows[(0, 1)]
    assert w1.deps_remaining == 1, \
        "successor treated the held boundary as already delivered"
    assert (0, 1) in wm.windows[(0, 0)].dependents
