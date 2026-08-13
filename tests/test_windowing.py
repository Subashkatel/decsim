#==================================================================
# TESTS FOR WINDOWING (dependency seam + parallel A/B scheme)
#==================================================================
import math

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import PerRoundDecoder, PresetLatencyDecoder
from decsim.devices import TimingOnlyDevice
from decsim.layouts import UniformLayout
from decsim.message import (
    Operation,
    OperationPlanningView,
    OperationWindowPlan,
    ResolvedCodeGeometry,
    ResolvedOperationPlanning,
    WindowGeometry,
)
from decsim.planner import (
    PerOpRounds,
    _materialize_execution_plan,
    _validate_operation_graph,
)
from decsim.schemes import (
    NaiveOnlineScheme,
    ParallelWindowScheme,
    SlidingTerminalPolicy,
    SlidingWindowScheme,
)
from conftest import continuous_stream
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


def test_regular_stride_tail_is_available_only_by_explicit_legacy_policy():
    plan = SlidingWindowScheme(
        terminal_policy=SlidingTerminalPolicy.REGULAR_STRIDE_LOOKAHEAD,
    ).plan_operation(
        7,
        5,
        commit_round_count=2,
        buffer_round_count=1,
    )
    assert plan.windows == (
        WindowGeometry(1, 1, 2, 3),
        WindowGeometry(3, 3, 4, 5),
        WindowGeometry(5, 5, 5, 6),
    )


def test_default_sliding_matches_quits_tan_finite_forward_geometry():
    """Pin the finite QUITS/Tan tail independently of the implementation."""
    for commit_round_count, buffer_round_count in ((3, 3), (2, 3), (3, 0)):
        width = commit_round_count + buffer_round_count
        for round_count in range(1, 3 * width + commit_round_count + 1):
            regular_count = max(
                0,
                math.ceil((round_count - width) / commit_round_count),
            )
            expected = tuple(
                WindowGeometry(
                    i * commit_round_count + 1,
                    i * commit_round_count + 1,
                    (i + 1) * commit_round_count,
                    i * commit_round_count + width,
                )
                for i in range(regular_count)
            ) + (
                WindowGeometry(
                    regular_count * commit_round_count + 1,
                    regular_count * commit_round_count + 1,
                    round_count,
                    round_count,
                ),
            )
            plan = SlidingWindowScheme().plan_operation(
                7,
                round_count,
                commit_round_count=commit_round_count,
                buffer_round_count=buffer_round_count,
            )
            assert plan.windows == expected
            assert plan.internal_dependencies == tuple(
                (i, i + 1) for i in range(len(expected) - 1)
            )
            committed = tuple(
                r
                for window in plan.windows
                for r in range(window.commit_lo, window.commit_hi + 1)
            )
            assert committed == tuple(range(1, round_count + 1))
            assert max(window.buffer_hi for window in plan.windows) <= round_count


def _memory_op():
    """One single-patch Clifford op -- a quantum-memory stream."""
    op = Operation(0, "M(q0)", (0,), clifford=True)
    op.patches = (0,)
    return [op]


def test_typed_scheme_ledgers_pin_mode_idle_policy_and_parallel_boundaries():
    sliding = SlidingWindowScheme().plan_operation(
        7, 5, commit_round_count=2, buffer_round_count=1,
    )
    assert sliding == OperationWindowPlan(
        operation_id=7,
        windows=(
            WindowGeometry(1, 1, 2, 3),
            WindowGeometry(3, 3, 5, 5),
        ),
        internal_dependencies=((0, 1),),
        entry_window_indices=(0,),
        exit_window_indices=(1,),
        windowed=True,
        batch_preceding_idle_rounds=False,
    )

    naive = NaiveOnlineScheme().plan_operation(
        7, 5, commit_round_count=2, buffer_round_count=1,
    )
    assert naive.windows == (WindowGeometry(1, 1, 5, 5),)
    assert naive.internal_dependencies == ()
    assert naive.entry_window_indices == naive.exit_window_indices == (0,)
    assert naive.windowed is False
    assert naive.batch_preceding_idle_rounds is True

    expected = {
        1: ((), (0,), (0,)),
        2: ((), (0,), (0,)),
        3: ((), (0,), (0,)),
        4: (((0, 1),), (0,), (1,)),
        5: (((0, 1),), (0,), (1,)),
        6: (((0, 1), (2, 1)), (0, 2), (1,)),
    }
    for window_count, (edges, roots, sinks) in expected.items():
        plan = ParallelWindowScheme().plan_operation(
            7,
            window_count,
            commit_round_count=1,
            buffer_round_count=1,
        )
        assert plan.internal_dependencies == edges
        assert plan.entry_window_indices == roots
        assert plan.exit_window_indices == sinks
        assert plan.windowed is True
        assert plan.batch_preceding_idle_rounds is False


def _plan(scheme, ops, rounds_per_op, d=3):
    rounds_policy = (
        rounds_per_op
        if hasattr(rounds_per_op, "rounds_for")
        else FixedRounds(rounds_per_op)
    )
    _validate_operation_graph(ops)
    code = SurfaceCodeModel(d=d)
    layout = UniformLayout(code)
    leading_floor, trailing_floor = code.buffering_floor()
    geometry = ResolvedCodeGeometry(
        code_name=code.name,
        distance=code.distance,
        commit_round_count=code.commit_rounds(),
        buffer_round_count=code.buffer_rounds(),
        minimum_leading_buffer_round_count=leading_floor,
        minimum_trailing_buffer_round_count=trailing_floor,
        one_patch_spatial_node_count=code.spatial_nodes(1),
        buffer_floor_override_active=code.buffer_floor_override_active(),
    )
    resolved = []
    ledgers = []
    for operation in ops:
        round_count = rounds_policy.rounds_for(operation, code)
        patch_count = max(
            1,
            len(operation.patches)
            if operation.patches
            else len(operation.qubits),
        )
        resolved.append(
            ResolvedOperationPlanning(
                operation_id=operation.id,
                code_geometry=geometry,
                round_count=round_count,
                round_ticks=1,
                spatial_node_count=layout.spatial_nodes_for(
                    operation,
                    base_spatial_node_count=code.spatial_nodes(patch_count),
                ),
            )
        )
        ledgers.append(
            scheme.plan_operation(
                operation.id,
                round_count,
                commit_round_count=geometry.commit_round_count,
                buffer_round_count=geometry.buffer_round_count,
            )
        )
    return _materialize_execution_plan(
        tuple(
            OperationPlanningView.from_operation(operation)
            for operation in ops
        ),
        tuple(resolved),
        tuple(ledgers),
    )


def _max_window_depth(plan):
    memo = {}

    def depth(key):
        if key in memo:
            return memo[key]
        deps = plan.windows[key].deps
        memo[key] = 1 + max((depth(dep) for dep in deps), default=0)
        return memo[key]

    return max(depth(key) for key in plan.windows)


def _timing_stream_plan(segment_rounds, d=3):
    """Plan one timing-only decode stream split into scheduled operation segments."""
    segments, stream_op, rounds_map = continuous_stream(None, segment_rounds,
                                                        patch=0, base_id=0)
    plan = _plan(
        ParallelWindowScheme(),
        [stream_op],
        PerOpRounds(rounds_map),
        d=d,
    )
    return plan, segments, stream_op, rounds_map


# ---- structural: the finite QUITS/Tan forward plan remains a chain -------------------

def test_sequential_chain_deps_unchanged():
    plan = _plan(SlidingWindowScheme(), _memory_op(), rounds_per_op=11, d=3)
    assert plan.window_count[0] == 3
    assert plan.windows[(0, 0)].deps == []
    for k in range(1, 3):
        assert plan.windows[(0, k)].deps == [(0, k - 1)]
    # no leading buffers in the sequential scheme
    assert all(w.start_round == w.commit_lo for w in plan.windows.values())


def test_cross_op_deps_use_entry_and_exit_defaults():
    a = Operation(0, "A", (0,), clifford=True)
    b = Operation(1, "B", (0,), clifford=True)
    a.patches, b.patches = (0,), (0,)
    b.predecessors = (0,)
    b.decoder_boundary_predecessors = (0,)
    plan = _plan(SlidingWindowScheme(), [a, b], rounds_per_op=11, d=3)
    assert plan.windows[(1, 0)].deps == [(0, plan.window_count[0] - 1)]


# ---- structural: Skoric 2209.08552 sec. I.C block A/B layout ----------------------

def test_parallel_scheme_layout_and_deps():
    # Skoric block schedule, d=3: the first A commits 2d; the intervening
    # B commits the full 3d gap; the next A commits its middle d.
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=18, d=3)
    assert plan.window_count[0] == 3
    a0, b0, a1 = (plan.windows[(0, k)] for k in range(3))
    assert (a0.start_round, a0.commit_lo, a0.commit_hi, a0.buffer_hi) == (1, 1, 6, 9)
    assert (b0.start_round, b0.commit_lo, b0.commit_hi, b0.buffer_hi) == (7, 7, 15, 15)
    assert (a1.start_round, a1.commit_lo, a1.commit_hi, a1.buffer_hi) == (13, 16, 18, 18)
    assert a0.deps == [] and a1.deps == []
    assert not a0.closed_temporal_boundaries
    assert b0.closed_temporal_boundaries
    assert not a1.closed_temporal_boundaries
    assert sorted(b0.deps) == [(0, 0), (0, 2)]
    assert a0.n_rounds == b0.n_rounds == 9
    assert a1.n_rounds == 6


def test_parallel_scheme_tail_window():
    # R=23 ends in a reduced terminal B after the second A block.
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=23, d=3)
    assert plan.window_count[0] == 4
    tail = plan.windows[(0, 3)]
    assert (tail.start_round, tail.commit_lo, tail.commit_hi, tail.buffer_hi) == (19, 19, 23, 23)
    assert tail.deps == [(0, 2)]
    assert tail.closed_temporal_boundaries
    committed = []
    for window in plan.windows.values():
        committed += list(range(window.commit_lo, window.commit_hi + 1))
    assert sorted(committed) == list(range(1, 24))


def test_parallel_stream_bounds_depth_across_short_scheduled_ops():
    """DecLat/Skoric parallel windows are global over a decode stream, not reset at every
    short scheduled operation. The clean path is therefore to plan one stream op whose rounds are
    emitted by several segment ops; static materialization then gives O(1) A/B depth."""
    d = 3
    plan, _segments, stream_op, _rounds_map = _timing_stream_plan([d] * 32, d=d)

    assert _max_window_depth(plan) == 2
    stream_id = stream_op.id
    assert sorted(plan.windows[(stream_id, 1)].deps) == [(stream_id, 0), (stream_id, 2)]
    assert plan.windows[(stream_id, 0)].deps == []
    assert plan.windows[(stream_id, 2)].deps == []
    assert plan.windows[(stream_id, 1)].n_rounds == 3 * d
    assert plan.windows[(stream_id, 2)].n_rounds == 3 * d
    assert plan.windows[(stream_id, 0)].n_rounds == 3 * d


def test_timing_only_stream_runs_through_normal_parallel_scheme():
    """Timing-only and real-syndrome streams use the same runtime path. The device emits empty
    payloads, but they are tagged to the stream id/global round and decoded by the normal
    static-plan + ParallelWindowScheme path."""
    d = 3
    plan, segments, stream_op, rounds_map = _timing_stream_plan([6, 6, 6], d=d)
    res = simulate(RunSpec(
              ops=segments,
              decode_ops=[stream_op],
              device=TimingOnlyDevice(),
              num_units=4,
              rounds_policy=PerOpRounds(rounds_map),
              code=SurfaceCodeModel(d=d),
              scheme=ParallelWindowScheme(),
              decoder=PresetLatencyDecoder(0.1),
          ), verbose=False)
    cluster = res.window_manager
    assert cluster.window_count[stream_op.id] == plan.window_count[stream_op.id]
    assert all(seg.id not in cluster.window_count for seg in segments)
    assert len(cluster.committed_windows) == cluster.total_windows
    # A real stream window spans the scheduled op seam at round 6/7.
    assert any(w.start_round <= 6 and w.buffer_hi >= 7
               for (op_id, _k), w in cluster.windows.items()
               if op_id == stream_op.id)


def test_short_successor_closes_cross_operation_buffer():
    """A predecessor window can finish when a shorter successor is exhausted.

    This is the operation-boundary version of flushing a dangling stream: if a
    successor physically has fewer rounds than the predecessor's full buffer,
    the predecessor must not wait forever for rounds that cannot arrive.
    """
    first = Operation(0, "long", (0,), clifford=True, patches=(0,))
    second = Operation(1, "short", (0,), clifford=True, patches=(0,),
                       predecessors=(0,),
                       decoder_boundary_predecessors=(0,))
    rounds_map = {0: 3, 1: 1}
    result = simulate(RunSpec(
                 ops=[first, second],
                 num_units=2,
                 rounds_policy=PerOpRounds(rounds_map),
                 code=SurfaceCodeModel(d=3),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(0.1),
             ), verbose=False)
    cluster = result.window_manager

    assert len(cluster.committed_windows) == cluster.total_windows
    assert cluster.payloads_held == 0


# ---- ACCEPTANCE: reaction tail reproduces gamma_mem = 6d*tau_d(d^2) + hops (Eq. 13) --

def test_parallel_scheme_has_two_decode_layers_and_waits_for_both_A_results():
    result = simulate(RunSpec(
        ops=_memory_op(),
        num_units=4,
        d=3,
        rounds_policy=FixedRounds(18),
        round_us=1.1,
        decoder=PerRoundDecoder(tau_us=2.0),
        scheme=ParallelWindowScheme(),
    ), verbose=False)
    a0 = result.window_manager.windows[(0, 0)]
    b0 = result.window_manager.windows[(0, 1)]
    a1 = result.window_manager.windows[(0, 2)]

    assert a0.t_dispatch < a1.t_done
    assert a1.t_dispatch < a0.t_done
    assert b0.t_dispatch >= max(a0.t_done, a1.t_done)
    assert b0.deps_remaining == 0


# ---- backlog vs units sweep: parallelism helps A/B, cannot help the chain -----------

def test_backlog_sweep_parallel_vs_sequential():
    # service (10 us) exceeds both schemes' window inter-arrival (sequential: one window
    # per commit stride = 3.3 us; parallel: ~2 windows per 12-round period = ~6.6 us), so
    # ONE unit backlogs in both cases -- the question is whether extra units help.
    def run(scheme, units):
        r = simulate(RunSpec(
                ops=_memory_op(),
                num_units=units,
                d=3,
                rounds_policy=FixedRounds(63),
                round_us=1.1,
                decoder=PresetLatencyDecoder(10.0),
                scheme=scheme,
            ), verbose=False)
        peak_q = max((q for _, q in r.decoder_manager.queue_log), default=0)
        return r.result.fully_done_ticks, peak_q

    seq = {u: run(SlidingWindowScheme(), u) for u in (1, 2, 4)}
    par = {u: run(ParallelWindowScheme(), u) for u in (1, 2, 4)}
    # the sequential chain cannot use extra units: one op's windows decode one at a time
    assert seq[4][0] == seq[1][0]
    # the parallel scheme converts units into completion time and into backlog relief
    assert par[4][0] < par[1][0]
    assert par[4][1] <= par[1][1]
    # and with units available it beats the chain outright
    assert par[4][0] < seq[4][0]


# ---- the naive (batch) baseline of arXiv:2510.25222 Sec III.C ------------------------

def test_naive_scheme_is_one_batch_window_per_op():
    from decsim.schemes import NaiveOnlineScheme
    plan = _plan(NaiveOnlineScheme(), _memory_op(), rounds_per_op=11, d=3)
    assert plan.window_count[0] == 1
    w = plan.windows[(0, 0)]
    assert (w.commit_lo, w.commit_hi, w.buffer_hi) == (1, 11, 11)


def test_naive_scheme_decodes_only_after_the_last_round():
    """The defining cost of the baseline: nothing decodes until ALL rounds arrived."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from conftest import trace_time
    from decsim.schemes import NaiveOnlineScheme
    r = simulate(RunSpec(
            ops=_memory_op(),
            num_units=1,
            d=3,
            rounds_policy=FixedRounds(11),
            decoder=PresetLatencyDecoder(1.0),
            scheme=NaiveOnlineScheme(),
        ), verbose=False)
    lines = r.engine.log_lines
    # naive = one batch decode of the whole op (no "Wk"/commit vocabulary); match the
    # decode-start independent of that wording.
    assert trace_time(lines, "START DECODE M(q0)") >= \
           trace_time(lines, "round 11 of M(q0) arrived")
