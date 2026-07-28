"""Window runtime kernel tests — each maps to a numbered contract rule.

These drive a real Engine with fakes at the ports; end-to-end validation
against real decoders/streams happens via the frozen timing goldens (Task 19).
"""
from dataclasses import fields

import pytest

from decsim.engine import Engine
from decsim.message import (
    DecodeJob,
    DecodeResult,
    Operation,
    OperationPlanningView,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    ResolvedPatchPlanning,
    RetainedSyndromeFragment,
    SyndromePayload,
    SyndromeRoundPacket,
    Window,
    WindowPlan,
)
from decsim.protocols import Directive, OutcomeDirective, Submission
from decsim.window_manager import LogicalContribution, WindowManager
from decsim.window_interactions import DefaultWindowInteraction

T_DD = 500_000
T_DO = 1_000_000


class _Link:
    def __init__(self, ticks): self._t = ticks
    def cost(self): return self._t


class _Links:
    dd = _Link(T_DD)
    do = _Link(T_DO)


class _Scheme:
    def data_complete(self, w, *, rounds_arrived, successor_rounds,
                      memory_rounds, round_count, has_successor, operation):
        # sliding-window rule: all rounds up to buffer_hi present (or op ended)
        return rounds_arrived + memory_rounds >= min(w.buffer_hi, round_count)


class _Deadline:
    def deadline(self, op, window, now, on_reaction_path):
        return now + 1_000


class _Feedback:
    def __init__(self): self.integrated = []
    def integrate(self, op, result): self.integrated.append((op.id, result))


class _Eager:
    def on_commit(self, window, final): return True


class _Held:
    def on_commit(self, window, final): return final


class _RecordingStrategy:
    """Baseline-like: submit the weak job; FINALIZE unless told otherwise."""
    def __init__(self, directive=None):
        self.ready, self.outcomes = [], []
        self._directive = directive or OutcomeDirective(Directive.FINALIZE)
    def on_window_ready(self, window, weak_job, services):
        self.ready.append(weak_job)
        return [Submission(weak_job)]
    def on_decode_outcome(self, outcome, services):
        self.outcomes.append(outcome)
        return self._directive
    def metrics(self): return {}


def _planning_view(operation):
    return OperationPlanningView(**{
        item.name: getattr(operation, item.name)
        for item in fields(OperationPlanningView)
    })


def _runtime(boundary=None, strategy=None, ops=(0,), deps=(), blocking=()):
    eng = Engine(verbose=False)
    fb = _Feedback()
    runtime_operations = {
        op_id: Operation(
            op_id,
            f"op{op_id}",
            (op_id,),
            blocked_by=(op_id - 1 if op_id in blocking else None),
        )
        for op_id in ops
    }
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
    resolved_operations = tuple(
        ResolvedOperationPlanning(
            operation_id=op_id,
            code_geometry=geometry,
            round_count=6,
            round_ticks=1,
            spatial_node_count=9,
        )
        for op_id in ops
    )
    resolved_patches = tuple(
        ResolvedPatchPlanning(
            patch_identity=op_id,
            code_geometry=geometry,
            round_ticks=1,
            spatial_node_count=9,
        )
        for op_id in ops
    )
    rt = WindowManager(eng, scheme=_Scheme(), code_geometry=geometry,
                       resolved_operations=resolved_operations,
                       resolved_patches=resolved_patches,
                       deadline_policy=_Deadline(), links=_Links(),
                       orchestrator=fb, boundary_policy=boundary or _Eager(),
                       window_interaction=DefaultWindowInteraction(),
                       planning_view_by_operation_id={
                           op_id: _planning_view(operation)
                           for op_id, operation in runtime_operations.items()
                       })
    rt.strategy = strategy or _RecordingStrategy()
    rt.services = object()
    submitted = []
    rt.submit_fn = lambda job, delay: submitted.append((job, delay))
    windows, op_windows, count = {}, {}, {}
    for op_id, op in runtime_operations.items():
        rt.register_op(op)
        w = Window(op_id=op_id, k=0, commit_lo=1, commit_hi=3, buffer_hi=6,
                   n_rounds=6)
        windows[(op_id, 0)] = w
        op_windows[op_id] = [0]
        count[op_id] = 1
    for src, dst in deps:
        windows[dst].deps.append(src)
        windows[dst].deps_remaining += 1
        windows[src].dependents.append(dst)
    rt.load_execution_plan(WindowPlan(
        windows=windows, window_count=count, op_windows=op_windows,
        successors={op_id: [] for op_id in ops},
        spatial_nodes={op_id: 9 for op_id in ops},
        rounds_by_operation={op_id: 6 for op_id in ops},
        code_names={op_id: "fake" for op_id in ops},
        windowed_by_operation={op_id: True for op_id in ops},
        batch_preceding_idle_rounds_by_operation={
            op_id: False for op_id in ops
        },
        total_windows=len(windows)))
    return eng, rt, fb, submitted


def _feed_rounds(rt, op_id, n):
    for r in range(1, n + 1):
        rt.on_syndrome_arrival(SyndromeRoundPacket(
            operation_id=op_id,
            round_index=r,
            fragments=(RetainedSyndromeFragment.from_payload(
                SyndromePayload(op_id, 0, r)
            ),),
        ))


def test_decoder_assembly_uses_structural_patch_order_not_transport_order():
    def independent_bytes(identity):
        if type(identity) is int:
            payload = str(identity).encode("ascii")
            return b"I" + len(payload).to_bytes(8, "big") + payload
        if type(identity) is str:
            payload = identity.encode("utf-8")
            return b"S" + len(payload).to_bytes(8, "big") + payload
        items = tuple(independent_bytes(item) for item in identity)
        return (
            b"T"
            + len(items).to_bytes(8, "big")
            + b"".join(len(item).to_bytes(8, "big") + item for item in items)
        )

    patch_ids = (2, "2", (), (2, "north"), ("gross", (5, "north")))
    expected_order = tuple(sorted(patch_ids, key=independent_bytes))

    def assembled_order(order):
        engine, runtime, _, submitted = _runtime()
        for round_index in range(1, 7):
            fragments = tuple(
                RetainedSyndromeFragment(
                    operation_id=0,
                    patch_id=patch_id,
                    round_index=round_index,
                    bits=(position & 1,),
                    code=None,
                    size_bits=1,
                )
                for position, patch_id in enumerate(order)
            )
            engine.now = 7 if round_index == 1 else 100 + round_index
            runtime.on_syndrome_arrival(SyndromeRoundPacket(
                operation_id=0,
                round_index=round_index,
                fragments=fragments,
            ))
        job, _ = submitted[0]
        return (
            tuple(fragment.patch_id for fragment in job.payloads[:5]),
            job.window.t_first_round,
        )

    forward = assembled_order(patch_ids)
    reverse = assembled_order(tuple(reversed(patch_ids)))

    assert forward == (expected_order, 7)
    assert reverse == (expected_order, 7)


def test_not_ready_until_data_and_deps():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 1, 6)                       # dependent has data...
    assert not submitted                          # ...but dep outstanding
    _feed_rounds(rt, 0, 6)                        # predecessor ready & submits
    assert [j.op_id for j, _ in submitted] == [0]


def test_job_fields_match_contract_2a4():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, delay = submitted[0]
    assert delay == 0
    assert (job.op_id, job.window_id, job.n_rounds) == (0, 0, 6)
    assert job.deadline == eng.now + 1_000 and job.spatial_nodes == 9
    assert len(job.payloads) == 6 and job.strong_label == "strong(op0 W0)"


def test_eager_ships_weak_boundary_unconditionally_contract_1_2():
    strat = _RecordingStrategy()
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1),
                                      strategy=strat)
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True             # escalated (set by decode layer)
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(1,),
                                        boundary_defects={7: [1, 0, 1]}))
    dep = rt.windows[(1, 0)]
    assert dep.deps_remaining == 1                # not yet: travels t_dd
    eng.run(until=T_DD)
    assert dep.deps_remaining == 0                # Eager shipped despite pending strong
    assert dep.boundary_in == {1: [1, 0, 1]}      # src round 7 - rounds_for(6) -> dep round 1


def test_boundary_shift_rule_contract_1_3():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    # src op0 has rounds_for=6; defects at src rounds 7,8 -> dep rounds 1,2;
    # src round 3 -> dep round -3 dropped
    rt.on_decode_done(job, DecodeResult(0, 0, boundary_defects={
        7: [1], 8: [1, 1], 3: [1, 1, 1]}))
    eng.run(until=T_DD)
    dep = rt.windows[(1, 0)]
    assert set(dep.boundary_in) == {1, 2}


def test_default_boundary_revisions_replace_one_sources_contribution():
    eng, rt, _, _ = _runtime(
        deps=[((0, 0), (1, 0))],
        ops=(0, 1),
    )
    source = rt.windows[(0, 0)]
    operation = rt._ops[0]

    # The first scheduled message is stale before it arrives. Only the newer
    # revision releases the dependency and contributes a mask.
    rt._send_boundary(source, operation, {7: [1, 0]})
    rt._send_boundary(source, operation, {7: [0, 1]})
    eng.run(until=T_DD)
    destination = rt.windows[(1, 0)]
    assert destination.deps_remaining == 0
    assert destination.boundary_in == {1: [0, 1]}

    # A later accepted revision replaces this source rather than decrementing
    # the dependency twice or XORing old and new versions together.
    rt._send_boundary(source, operation, {7: [1, 1, 1]})
    eng.run(until=eng.now + T_DD)
    assert destination.deps_remaining == 0
    assert destination.boundary_in == {1: [1, 1, 1]}


def test_strong_revises_logical_only_contract_1_4():
    eng, rt, fb, submitted = _runtime(deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(1,),
                                        boundary_defects={7: [1]}))
    eng.run(until=T_DD)
    dep_boundary_before = dict(rt.windows[(1, 0)].boundary_in)
    rt.on_strong_decode_done((0, 0), DecodeResult(
        0, 0, logical_observables=(0,),
                                                  boundary_defects={7: [1, 1]}))
    assert rt.logical_contributions[(0, 0)].logical_observables == (0,)
    assert rt.windows[(1, 0)].boundary_in == dep_boundary_before  # untouched
    assert rt.op_strong_commit_time[0] == eng.now


def test_op_delivery_gated_on_pending_strong_contract_1_5():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(1,)))
    eng.run()
    assert fb.integrated == []                    # gated: pending strong
    rt.on_strong_decode_done(
        (0, 0),
        DecodeResult(0, 0, logical_observables=(1,)),
    )
    eng.run()
    assert [op_id for op_id, _ in fb.integrated] == [0]   # released after final


def test_op_delivery_immediate_when_not_awaiting():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(1,)))
    assert fb.integrated == []                    # travels t_do
    eng.run(until=T_DO)
    assert [op_id for op_id, _ in fb.integrated] == [0]
    assert fb.integrated[0][1].logical_observables == (1,)


def test_held_ships_only_when_final():
    eng, rt, fb, submitted = _runtime(boundary=_Held(),
                                      deps=[((0, 0), (1, 0))], ops=(0, 1))
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    job.awaiting_strong_result = True
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(1,),
                                        boundary_defects={7: [1]}))
    eng.run()
    assert rt.windows[(1, 0)].deps_remaining == 1   # held: nothing shipped
    rt.on_strong_decode_done((0, 0), DecodeResult(
        0, 0, logical_observables=(1,),
                                                  boundary_defects={7: [1]}))
    eng.run()
    assert rt.windows[(1, 0)].deps_remaining == 0   # shipped at final


def test_late_round_after_op_freed_raises():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    job, _ = submitted[0]
    rt.on_decode_done(job, DecodeResult(0, 0, logical_observables=(0,)))
    eng.run()
    with pytest.raises(RuntimeError, match="syndrome RAM was freed"):
        rt.on_syndrome_arrival(SyndromeRoundPacket(
            operation_id=0,
            round_index=7,
            fragments=(RetainedSyndromeFragment.from_payload(
                SyndromePayload(0, 0, 7)
            ),),
        ))


def test_packet_operation_and_round_limit_reject_before_storage_mutates():
    _, runtime, _, _ = _runtime()

    with pytest.raises(ValueError, match="unknown syndrome operation"):
        runtime.on_syndrome_arrival(SyndromeRoundPacket(
            operation_id="unknown",
            round_index=1,
            fragments=(RetainedSyndromeFragment.from_payload(
                SyndromePayload("unknown", 0, 1)
            ),),
        ))

    with pytest.raises(ValueError, match="round limit 6"):
        runtime.on_syndrome_arrival(SyndromeRoundPacket(
            operation_id=0,
            round_index=7,
            fragments=(RetainedSyndromeFragment.from_payload(
                SyndromePayload(0, 0, 7)
            ),),
        ))

    assert runtime.rounds_arrived[0] == 0
    assert runtime.store.fragments(0, 7) is None


def test_strong_job_two_sided_context_contract_2b6():
    eng, rt, fb, submitted = _runtime()
    _feed_rounds(rt, 0, 6)
    weak, _ = submitted[0]
    strong = rt.make_strong_decode_job(weak, round_count=9, label="strong")
    w = strong.window
    assert (w.buffer_lo, w.commit_lo, w.commit_hi, w.buffer_hi) == (1, 1, 3, 6)
    assert strong.hint == "strong" and strong.attempt == 1
    assert strong.strong_decode_for == (0, 0) and strong.deadline == eng.now


def test_logical_contributions_xor_vectors_over_exact_round_coverage():
    _, runtime, _, _ = _runtime()
    runtime.logical_contributions = {
        (0, 0): LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=3,
            ownership_kind="ordinary_window",
            logical_observables=(1, 0, 1),
        ),
        (0, 1): LogicalContribution(
            owner_key=(0, 1),
            commit_lo=4,
            commit_hi=6,
            ownership_kind="ordinary_window",
            logical_observables=(0, 1, 1),
        ),
    }

    prediction = runtime._logical_observables_for_interval(
        0,
        1,
        6,
        boundary_policy="strict",
    )

    assert prediction == (1, 1, 0)


def test_real_timing_only_contribution_keeps_interval_timing_only():
    _, runtime, _, _ = _runtime()
    runtime.logical_contributions = {
        (0, 0): LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=3,
            ownership_kind="ordinary_window",
            logical_observables=(1,),
        ),
        (0, 1): LogicalContribution(
            owner_key=(0, 1),
            commit_lo=4,
            commit_hi=6,
            ownership_kind="ordinary_window",
            logical_observables=None,
        ),
    }

    assert runtime._logical_observables_for_interval(
        0,
        1,
        6,
        boundary_policy="strict",
    ) is None


def test_functional_segment_cannot_split_a_contribution_extent():
    _, runtime, _, _ = _runtime()
    runtime.logical_contributions = {
        (0, 0): LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=6,
            ownership_kind="strong_slab",
            logical_observables=(1,),
        ),
    }

    with pytest.raises(RuntimeError, match="functional.*boundary"):
        runtime._logical_observables_for_interval(
            0,
            1,
            3,
            boundary_policy="stream_segment",
        )


def test_timing_only_segment_crossing_still_requires_exact_coverage():
    _, runtime, _, _ = _runtime()
    runtime.logical_contributions = {
        (0, 0): LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=2,
            ownership_kind="ordinary_window",
            logical_observables=None,
        ),
    }

    with pytest.raises(RuntimeError, match="contribution gap at round 3"):
        runtime._logical_observables_for_interval(
            0,
            2,
            4,
            boundary_policy="stream_segment",
        )


def test_timing_only_segment_crossing_with_exact_coverage_is_inert():
    _, runtime, _, _ = _runtime()
    runtime.logical_contributions = {
        (0, 0): LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=2,
            ownership_kind="ordinary_window",
            logical_observables=None,
        ),
        (0, 1): LogicalContribution(
            owner_key=(0, 1),
            commit_lo=3,
            commit_hi=4,
            ownership_kind="ordinary_window",
            logical_observables=None,
        ),
    }

    assert runtime._logical_observables_for_interval(
        0,
        2,
        4,
        boundary_policy="stream_segment",
    ) is None


def test_operation_observable_arity_cannot_change_between_windows():
    _, runtime, _, _ = _runtime()
    runtime._install_logical_contribution(
        LogicalContribution(
            owner_key=(0, 0),
            commit_lo=1,
            commit_hi=3,
            ownership_kind="ordinary_window",
            logical_observables=(1, 0),
        )
    )

    with pytest.raises(ValueError, match="length 1.*expected 2"):
        runtime._install_logical_contribution(
            LogicalContribution(
                owner_key=(0, 1),
                commit_lo=4,
                commit_hi=6,
                ownership_kind="ordinary_window",
                logical_observables=(1,),
            )
        )
