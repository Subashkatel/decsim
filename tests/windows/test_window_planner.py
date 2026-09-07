"""The window planner's laws: the papers' schedules and a stream's growth.

Skoric et al. 2209.08552 (tmp/papers): window k of the sliding window
commits ncom rounds and reads nbuf past them (lines 194-203, 275-278),
and in block A/B decoding the first A commits the first 2d rounds while
a B window has smooth time boundaries and no buffers (lines 388-405,
688-700). Tan et al. 2209.09219: a type-1 sandwich window has width
s + 2b and a type-2 seam has both time boundaries closed (supplement
S8, lines 1116-1137). A stream grows one window when its commit region
begins and its tail is clipped at the seal (the runtime twin of the
static plan).
"""

import types

import pytest

import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.window_planner as window_planner
import decsim.windows.windowing_schemes as windowing_schemes

COMMIT_ROUNDS = 3
BUFFER_ROUNDS = 2


def _empty_plan() -> window_records.WindowPlan:
    return window_records.WindowPlan(
        windows={},
        window_count={},
        op_windows={},
        successors={},
        spatial_nodes={},
        rounds_by_operation={},
        code_names={},
        total_windows=0,
        windowed_by_operation={},
        batch_preceding_idle_rounds_by_operation={},
    )


def _resolved(operation_id, round_count: int):
    geometry = types.SimpleNamespace(
        commit_round_count=COMMIT_ROUNDS,
        buffer_round_count=BUFFER_ROUNDS,
        code_name="surface",
    )
    return types.SimpleNamespace(
        operation_id=operation_id,
        code_geometry=geometry,
        round_count=round_count,
        spatial_node_count=17,
    )


def _stream(operation_id="stream"):
    return program_records.Operation(operation_id, "stream", (0,))


class _FiniteSource:
    """A source that fixes the stream's length up front."""

    def __init__(self, round_limit: int) -> None:
        self.round_limit = round_limit

    def register_dynamic_stream(self, operation, round_count, **_arguments):
        del operation, round_count
        return self.round_limit

    def window_model_for_stream(self, stream_id, window):
        del stream_id, window
        return None


def _planner(source=None) -> window_planner.WindowPlanner:
    built = built_window_models.BuiltWindowModels()
    models = window_planner.WindowModels(source, lambda _code_name: None, built)
    scheme = windowing_schemes.SlidingWindowScheme()
    resolved = [_resolved("stream", 9)]
    plan = _empty_plan()
    return window_planner.WindowPlanner(
        scheme, resolved, plan, models, planned_operations=()
    )


def test_sliding_window_k_commits_ncom_rounds_and_reads_nbuf_past_them():
    scheme = windowing_schemes.SlidingWindowScheme()
    plan = scheme.plan_operation(
        1, 30, commit_round_count=5, buffer_round_count=5
    )
    for index, geometry in enumerate(plan.windows[:-1]):
        assert geometry.commit_lo == index * 5 + 1
        assert geometry.commit_hi == (index + 1) * 5
        assert geometry.buffer_hi == geometry.commit_hi + 5
    last = plan.windows[-1]
    assert last.commit_hi == 30
    assert last.buffer_hi == 30


def qldpc_sliding_windows(round_count, width, stride):
    """The tail loop of qLDPC's own SlidingWindowDecoder (sinter.py).

    `while start < end - (W + s - 1)` opens a regular window of width W
    every stride s and stops as soon as fewer than W + s rounds remain;
    the last window then commits everything left.
    """
    start = 0
    windows = []
    last_regular_start = round_count - (width + stride - 1)
    while start < last_regular_start:
        regular = (start + 1, start + stride, start + width)
        windows.append(regular)
        start += stride
    tail = (start + 1, round_count, round_count)
    windows.append(tail)
    return windows


def test_the_sliding_tail_is_qldpcs_tail_on_every_shape_it_ships():
    """The Tan flush terminal policy against qLDPC's loop, shape for shape."""
    scheme = windowing_schemes.SlidingWindowScheme()
    for round_count in (5, 13, 20, 30, 31, 32, 33):
        for commit_rounds, buffer_rounds in ((3, 6), (3, 3), (2, 4), (5, 5)):
            plan = scheme.plan_operation(
                1,
                round_count,
                commit_round_count=commit_rounds,
                buffer_round_count=buffer_rounds,
            )
            ours = [
                (window.commit_lo, window.commit_hi, window.buffer_hi)
                for window in plan.windows
            ]
            width = commit_rounds + buffer_rounds
            theirs = qldpc_sliding_windows(round_count, width, commit_rounds)
            assert ours == theirs


def test_the_sliding_tail_absorbs_a_short_remainder():
    """A tail shorter than a window is decoded by the window before it."""
    scheme = windowing_schemes.SlidingWindowScheme()
    plan = scheme.plan_operation(
        1, 31, commit_round_count=3, buffer_round_count=6
    )
    last = plan.windows[-1]
    assert (last.commit_lo, last.commit_hi) == (22, 31)


def test_first_block_a_commits_2d_and_a_block_b_has_no_buffers():
    scheme = windowing_schemes.ParallelWindowScheme()
    plan = scheme.plan_operation(
        1, 40, commit_round_count=3, buffer_round_count=3
    )
    first_a = plan.windows[0]
    assert first_a.commit_lo == 1
    assert first_a.commit_hi == 6
    first_b = plan.windows[1]
    assert first_b.closed_temporal_boundaries
    assert first_b.buffer_lo == first_b.commit_lo
    assert first_b.buffer_hi == first_b.commit_hi
    assert (0, 1) in plan.internal_dependencies
    assert (2, 1) in plan.internal_dependencies


def test_tan_type_1_width_is_s_plus_2b_and_the_seam_is_closed_both_sides():
    scheme = windowing_schemes.TanSandwichScheme()
    plan = scheme.plan_operation(
        1, 30, commit_round_count=4, buffer_round_count=2
    )
    interior_core = plan.windows[2]
    assert interior_core.round_count == 4 + 2 * 2
    seam = plan.windows[1]
    assert seam.closed_temporal_boundaries
    assert seam.round_count == 1
    assert (0, 1) in plan.internal_dependencies
    assert (2, 1) in plan.internal_dependencies


def test_a_stream_grows_one_window_when_its_commit_region_begins():
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream)
    assert planner.grow_stream("stream", 1, None) != []
    assert planner.grow_stream("stream", 3, None) == []
    created = planner.grow_stream("stream", 7, None)
    assert [window.k for window in created] == [1, 2]
    first, second, third = planner.windows_of("stream")
    assert (first.commit_lo, first.commit_hi, first.buffer_hi) == (1, 3, 5)
    assert (second.commit_lo, second.commit_hi, second.buffer_hi) == (4, 6, 8)
    assert (third.commit_lo, third.commit_hi, third.buffer_hi) == (7, 9, 11)
    assert planner.total_windows == 3
    assert planner.window_count_of("stream") == 3


def test_a_finite_source_fixes_the_stream_windows_in_its_own_order():
    source = _FiniteSource(5)
    planner = _planner(source)
    stream = _stream()
    limit = planner.register_stream(stream)
    assert limit == 5
    assert planner.is_finite_stream("stream")
    assert planner.grow_stream("stream", 0, None) == []
    created = planner.grow_stream("stream", 5, None)
    geometries = [(w.commit_lo, w.commit_hi, w.buffer_hi) for w in created]
    assert geometries == [(1, 5, 5)]
    assert planner.grow_stream("stream", 9, None) == []


def test_the_seal_clips_the_window_holding_the_last_round():
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream)
    planner.grow_stream("stream", 5, 5)
    first, second = planner.windows_of("stream")
    assert (second.commit_lo, second.commit_hi) == (4, 5)
    clipped = planner.trim_stream_tail("stream", 5)
    assert clipped is second
    assert (second.commit_hi, second.buffer_hi, second.n_rounds) == (5, 7, 4)
    assert (first.commit_hi, first.buffer_hi) == (3, 5)
    assert planner.trim_stream_tail("stream", 20) is None


def test_idle_rounds_fold_only_into_a_batch_style_operation():
    plan = _empty_plan()
    window = window_records.Window(
        operation_id=7, k=0, commit_lo=1, commit_hi=6, buffer_hi=6, n_rounds=6
    )
    plan.windows[(7, 0)] = window
    plan.batch_preceding_idle_rounds_by_operation[7] = True
    built = built_window_models.BuiltWindowModels()
    models = window_planner.WindowModels(None, lambda _code_name: None, built)
    scheme = windowing_schemes.NaiveOnlineScheme()
    resolved = [_resolved(7, 6)]
    planner = window_planner.WindowPlanner(
        scheme, resolved, plan, models, planned_operations=()
    )
    planner.prepend_idle_rounds(7, 4)
    planner.prepend_idle_rounds(7, 0)
    assert window.batched_preceding_idle_round_count == 4


def _geometry(*, buffer_round_count, justification):
    """A d=3 surface geometry with the buffer width and justification given."""
    return program_records.ResolvedCodeGeometry(
        code_name="rotated surface code (d=3)",
        distance=3,
        commit_round_count=3,
        buffer_round_count=buffer_round_count,
        minimum_leading_buffer_round_count=3,
        minimum_trailing_buffer_round_count=3,
        one_patch_spatial_node_count=9,
        window_floor_justification=justification,
    )


def test_a_buffer_below_the_papers_floor_is_refused_unless_the_card_says_why():
    """The (d, d) floor is Skoric's and Tan's; below it accuracy degrades.

    A card may still run below the floor, deliberately, by writing down
    why (Skoric 2209.08552, Tan PRX Quantum 4, 040344, Bombin
    2303.04846); the scheme refuses a silent one.
    """
    scheme = windowing_schemes.SlidingWindowScheme()
    bare = _geometry(buffer_round_count=0, justification=None)
    with pytest.raises(
        ValueError, match="is below the trailing buffering floor 3"
    ):
        scheme.validate_buffer(bare)
    justified = _geometry(
        buffer_round_count=0, justification="a zero buffer on purpose"
    )
    scheme.validate_buffer(justified)


def test_a_justification_on_a_card_at_the_floor_is_refused_as_stale():
    scheme = windowing_schemes.SlidingWindowScheme()
    geometry = _geometry(buffer_round_count=3, justification="stale")
    with pytest.raises(
        ValueError, match="is not below the trailing buffering floor 3"
    ):
        scheme.validate_buffer(geometry)


def test_the_parallel_scheme_refuses_unequal_commit_and_buffer_widths():
    """Skoric's block A/B construction fixes ncom = nbuf = d."""
    scheme = windowing_schemes.ParallelWindowScheme()
    with pytest.raises(
        ValueError, match="requires commit_round_count == buffer_round_count"
    ):
        scheme.plan_operation(1, 30, commit_round_count=3, buffer_round_count=4)
