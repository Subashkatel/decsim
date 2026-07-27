#==================================================================
# TESTS FOR WINDOWING (dependency seam + parallel A/B scheme)
#==================================================================
from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.decoders import PerRoundDecoder, PresetLatencyDecoder
from decsim.devices import TimingOnlyDevice
from decsim.layouts import UniformLayout
from decsim.message import Operation
from decsim.planner import WindowPlanner
from decsim.planner import PerOpRounds
from decsim.schemes import SlidingWindowScheme, ParallelWindowScheme
from conftest import continuous_stream
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds


def _memory_op(rounds_unused=None):
    """One single-patch Clifford op -- a quantum-memory stream."""
    op = Operation(0, "M(q0)", (0,), clifford=True)
    op.patches = (0,)
    return [op]


def _plan(scheme, ops, rounds_per_op, d=3):
    planner = WindowPlanner(scheme, UniformLayout(SurfaceCodeModel(d=d)), rounds_per_op)
    return planner.plan(ops)


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
    code = SurfaceCodeModel(d=d)
    segments, stream_op, rounds_map = continuous_stream(None, segment_rounds,
                                                        patch=0, base_id=0)
    planner = WindowPlanner(ParallelWindowScheme(), UniformLayout(code),
                            PerOpRounds(rounds_map))
    return planner.plan([stream_op]), segments, stream_op, rounds_map


# ---- structural: the default chain is unchanged by the seam refactor ----------------

def test_sequential_chain_deps_unchanged():
    plan = _plan(SlidingWindowScheme(), _memory_op(), rounds_per_op=11, d=3)
    assert plan.window_count[0] == 4                      # ceil(11/3)
    assert plan.windows[(0, 0)].deps == []
    for k in range(1, 4):
        assert plan.windows[(0, k)].deps == [(0, k - 1)]
    # no leading buffers in the sequential scheme
    assert all(w.start_round == w.commit_lo for w in plan.windows.values())


def test_cross_op_deps_use_entry_and_exit_defaults():
    a = Operation(0, "A", (0,), clifford=True)
    b = Operation(1, "B", (0,), clifford=True)
    a.patches, b.patches = (0,), (0,)
    b.predecessors, a.has_successor = (0,), True
    plan = _plan(SlidingWindowScheme(), [a, b], rounds_per_op=11, d=3)
    assert plan.windows[(1, 0)].deps == [(0, plan.window_count[0] - 1)]


# ---- structural: parallel A/B layout per Skoric 2209.08552 / Tan 2209.09219 ---------

def test_parallel_scheme_layout_and_deps():
    # d=3: commit and buffer are both 3 rounds. Commit regions tile the stream as
    # alternating A/B blocks, each of size d except a short final tail.
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=15, d=3)
    assert plan.window_count[0] == 5
    a0, b0, a1, b1, a2 = (plan.windows[(0, k)] for k in range(5))
    assert (a0.start_round, a0.commit_lo, a0.commit_hi, a0.buffer_hi) == (1, 1, 3, 6)
    assert (b0.start_round, b0.commit_lo, b0.commit_hi, b0.buffer_hi) == (1, 4, 6, 9)
    assert (a1.start_round, a1.commit_lo, a1.commit_hi, a1.buffer_hi) == (4, 7, 9, 12)
    assert (b1.start_round, b1.commit_lo, b1.commit_hi, b1.buffer_hi) == (7, 10, 12, 15)
    assert (a2.start_round, a2.commit_lo, a2.commit_hi, a2.buffer_hi) == (10, 13, 15, 18)
    # layer-A windows are independent; layer-B waits on its neighboring A windows.
    assert a0.deps == [] and a1.deps == [] and a2.deps == []
    assert sorted(b0.deps) == [(0, 0), (0, 2)]
    assert sorted(b1.deps) == [(0, 2), (0, 4)]
    assert b0.n_rounds == 9 and a1.n_rounds == 9 and b1.n_rounds == 9


def test_parallel_scheme_tail_window():
    # R=23 leaves a short layer-B tail after A_3.
    plan = _plan(ParallelWindowScheme(), _memory_op(), rounds_per_op=23, d=3)
    assert plan.window_count[0] == 8
    tail = plan.windows[(0, 7)]
    assert (tail.start_round, tail.commit_lo, tail.commit_hi, tail.buffer_hi) == (19, 22, 23, 26)
    assert tail.deps == [(0, 6)]
    # every round 1..R is committed by exactly one window
    committed = []
    for w in plan.windows.values():
        committed += list(range(w.commit_lo, w.commit_hi + 1))
    assert sorted(committed) == list(range(1, 24))


def test_parallel_stream_bounds_depth_across_short_scheduled_ops():
    """DecLat/Skoric parallel windows are global over a decode stream, not reset at every
    short scheduled operation. The clean path is therefore to plan one stream op whose rounds are
    emitted by several segment ops; the ordinary WindowPlanner then gives O(1) A/B depth."""
    d = 3
    plan, _segments, stream_op, _rounds_map = _timing_stream_plan([d] * 32, d=d)

    assert _max_window_depth(plan) == 2
    stream_id = stream_op.id
    assert sorted(plan.windows[(stream_id, 1)].deps) == [(stream_id, 0), (stream_id, 2)]
    assert plan.windows[(stream_id, 0)].deps == []
    assert plan.windows[(stream_id, 2)].deps == []
    assert plan.windows[(stream_id, 1)].n_rounds == 3 * d
    assert plan.windows[(stream_id, 2)].n_rounds == 3 * d
    assert plan.windows[(stream_id, 0)].n_rounds == 2 * d


def test_timing_only_stream_runs_through_normal_parallel_scheme():
    """Timing-only and real-syndrome streams use the same runtime path. The device emits empty
    payloads, but they are tagged to the stream id/global round and decoded by the normal
    WindowPlanner + ParallelWindowScheme path."""
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
    cluster = res["cluster"]
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
                       predecessors=(0,))
    first.has_successor = True
    rounds_map = {0: 3, 1: 1}
    result = simulate(RunSpec(
                 ops=[first, second],
                 num_units=2,
                 rounds_policy=PerOpRounds(rounds_map),
                 code=SurfaceCodeModel(d=3),
                 scheme=SlidingWindowScheme(),
                 decoder=PresetLatencyDecoder(0.1),
             ), verbose=False)
    cluster = result["cluster"]

    assert len(cluster.committed_windows) == cluster.total_windows
    assert cluster.payloads_held == 0


# ---- ACCEPTANCE: reaction tail reproduces gamma_mem = 6d*tau_d(d^2) + hops (Eq. 13) --

def test_parallel_scheme_reaction_matches_eq13():
    d = 3
    # tau_d = 1 us per round (Eq. 12 shape at this operating point)
    r = simulate(RunSpec(
            ops=_memory_op(),
            num_units=4,
            d=d,
            rounds_policy=FixedRounds(15),
            round_us=1.1,
            decoder=PerRoundDecoder(tau_us=1.0),
            scheme=ParallelWindowScheme(),
        ), verbose=False)
    tail = r["fully_done"] - r["chip_done"]
    # after the last round: chip->controller->decoders hops, the last layer-A window
    # (3d rounds), the t_dd boundary, the layer-B window (3d rounds), then t_do.
    # This is Eq. 13's two-window 6d*tau_d(d^2) plus the one-way hops (a Clifford memory
    # op pays no t_oc + t_cq return path -- Pauli-frame updates stay in the orchestrator).
    expected = (us(0.15) + us(2.0)            # t_qc + t_cd
                + us(3 * d * 1.0) + us(0.5)   # 3d rounds at 1 us/round + t_dd
                + us(3 * d * 1.0) + us(1.0))  # 3d rounds + t_do
    assert abs(tail - expected) <= 4          # integer-tick rounding only


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
        peak_q = max((q for _, q in r["cluster"].queue_log), default=0)
        return r["fully_done"], peak_q

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
    lines = r["engine"].log_lines
    # naive = one batch decode of the whole op (no "Wk"/commit vocabulary); match the
    # decode-start independent of that wording.
    assert trace_time(lines, "START DECODE M(q0)") >= \
           trace_time(lines, "round 11 of M(q0) arrived")
