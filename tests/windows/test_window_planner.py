"""The window planner's laws: the sliding tail and a stream's growth.

The sliding row's flush tail is qLDPC's own SlidingWindowDecoder loop,
shape for shape. A stream grows one window when its commit region
begins, its tail is clipped at the seal, and a cut restarts the stride
(the runtime twin of the static plan). Each scheme's own layout is
pinned in tests/windows/schemes.
"""

import types
from typing import Optional

import pytest

import decsim.records.program as program_records
import decsim.records.windows as window_records
import decsim.windows.built_window_models as built_window_models
import decsim.windows.schemes.naive_online as naive_online_scheme
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.window_planner as window_planner

COMMIT_ROUNDS = 3
BUFFER_ROUNDS = 2


def _empty_plan() -> window_records.WindowPlan:
    return window_records.WindowPlan(
        windows={},
        window_count={},
        op_windows={},
        successors={},
        rounds_by_operation={},
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
    """A source that fixes the stream's length up front.

    It fills the whole WindowModelSource port: a provider the planner
    accepts must answer every question the planner asks, and building
    no model is one of the answers.
    """

    operation_circuit_scope = "none"

    def __init__(self, round_limit: int) -> None:
        self.round_limit = round_limit

    def register_dynamic_stream(self, operation, round_count, **_arguments):
        del operation, round_count
        return self.round_limit

    def finalize_stream_models(
        self,
        operation: program_records.Operation,
        stream_round_count: int,
    ) -> bool:
        del operation
        del stream_round_count
        return False

    def window_models_for_operation(
        self, operation, windows, round_count, **_arguments
    ):
        del operation, round_count
        return [None] * len(windows)

    def window_model_for_stream(self, stream_id, window):
        del stream_id, window
        return None

    def strong_window_model_for_operation(
        self, operation, window, round_count, **_arguments
    ):
        del operation, window, round_count
        return None


class _Decoder:
    """The decoder, as the models ask it what a model must offer."""

    # this decoder asks a window model for nothing
    fault_model_requirement = None


def _planner(source=None) -> window_planner.WindowPlanner:
    built = built_window_models.BuiltWindowModels()
    models = window_planner.WindowModels(built)
    if source is not None:
        models.provider = source
    models.decoder = _Decoder()
    scheme = sliding_scheme.SlidingWindowScheme()
    resolved = [_resolved("stream", 9)]
    plan = _empty_plan()
    planner = window_planner.WindowPlanner(resolved, plan, ())
    planner.scheme = scheme
    planner.models = models
    planner.start()
    return planner


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


@pytest.mark.parametrize("round_count", [5, 13, 20, 30, 31, 32, 33])
@pytest.mark.parametrize(
    "commit_rounds, buffer_rounds", [(3, 6), (3, 3), (2, 4), (5, 5)]
)
def test_the_sliding_tail_is_qldpcs_tail_on_every_shape_it_ships(
    round_count, commit_rounds, buffer_rounds
):
    """The flush terminal policy against qLDPC's loop, shape for shape."""
    scheme = sliding_scheme.SlidingWindowScheme()
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


def test_a_stream_grows_one_window_when_its_commit_region_begins():
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream, None)
    assert planner.grow_stream("stream", 1, None) != []
    assert planner.grow_stream("stream", 3, None) == []
    created = planner.grow_stream("stream", 7, None)
    assert [window.window_index for window in created] == [1, 2]
    first, second, third = planner.windows_of("stream")
    assert (first.commit_lo, first.commit_hi, first.buffer_hi) == (1, 3, 5)
    assert (second.commit_lo, second.commit_hi, second.buffer_hi) == (4, 6, 8)
    assert (third.commit_lo, third.commit_hi, third.buffer_hi) == (7, 9, 11)
    assert planner.total_windows == 3
    assert planner.window_count_of("stream") == 3


@pytest.mark.parametrize("physical_round_limit", [None, 5])
def test_a_finite_source_fixes_the_stream_windows_in_its_own_order(
    physical_round_limit: Optional[int],
) -> None:
    source = _FiniteSource(5)
    planner = _planner(source)
    stream = _stream()
    limit = planner.register_stream(stream, physical_round_limit)
    assert limit == 5
    assert planner.is_finite_stream("stream")
    assert planner.grow_stream("stream", 0, None) == []
    created = planner.grow_stream("stream", 5, None)
    geometries = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in created
    ]
    assert geometries == [(1, 5, 5)]
    assert planner.grow_stream("stream", 9, None) == []


def test_the_seal_clips_the_window_holding_the_last_round():
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", 5, 5)
    first, second = planner.windows_of("stream")
    assert (second.commit_lo, second.commit_hi) == (4, 5)
    clipped = planner.trim_stream_tail("stream", 5)
    assert clipped is second
    assert (second.commit_hi, second.buffer_hi, second.round_count) == (5, 7, 4)
    assert (first.commit_hi, first.buffer_hi) == (3, 5)
    assert planner.trim_stream_tail("stream", 20) is None


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5, 6, 7])
def test_a_closed_boundary_ends_a_commit_region_and_restarts_the_stride(
    boundary: int,
) -> None:
    """No window commits across a closed boundary; the next starts after it.

    The window holding the boundary commits through it, and the rounds
    after it are windowed from the boundary on, each window committing
    the stride, until the seal clips the last one.
    """
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", boundary, None)
    clipped = planner.trim_stream_tail("stream", boundary)
    assert clipped.commit_hi == boundary
    stream_round_count = boundary + 7
    planner.grow_stream("stream", stream_round_count, stream_round_count)
    planner.trim_stream_tail("stream", stream_round_count)
    commits = _commit_spans(planner)
    after = _spans_after(commits, boundary)
    first_round_after = boundary + 1
    assert after[0][0] == first_round_after
    after_last_round = stream_round_count + 1
    every_round = range(1, after_last_round)
    assert _rounds_covered(commits) == list(every_round)
    stride_ends = [commit_hi for _, commit_hi in after[:-1]]
    first_stride_end = boundary + 3
    second_stride_end = boundary + 6
    assert stride_ends == [first_stride_end, second_stride_end]


@pytest.mark.parametrize("known_round", [4, 7])
def test_a_segment_cut_starts_a_window_on_the_segment_s_first_round(
    known_round: int,
) -> None:
    """A segment after round 7 starts a window at round 8.

    Laid or not yet laid, the window holding round 7 commits through
    it, and the stride restarts on round 8.
    """
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", known_round, None)
    planner.cut_stream_after("stream", 7)
    planner.grow_stream("stream", 14, None)
    commits = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in planner.windows_of("stream")
    ]
    assert commits == [
        (1, 3, 5),
        (4, 6, 8),
        (7, 7, 9),
        (8, 10, 12),
        (11, 13, 15),
        (14, 16, 18),
    ]


def test_a_cut_on_a_window_edge_leaves_the_stride():
    planner = _planner()
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", 4, None)
    assert planner.cut_stream_after("stream", 6) is None
    planner.grow_stream("stream", 7, None)
    commits = _commit_spans(planner)
    assert commits == [(1, 3), (4, 6), (7, 9)]


@pytest.mark.parametrize("known_round", [1, 4])
def test_a_finite_stream_cut_replans_the_rounds_after_it(
    known_round: int,
) -> None:
    """The scheme lays the rounds after the cut as their own stretch.

    Laid or not yet laid, the window holding round 4 commits through it.
    """
    source = _FiniteSource(12)
    planner = _planner(source)
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", known_round, None)
    planner.cut_stream_after("stream", 4)
    planner.grow_stream("stream", 12, None)
    commits = _commit_spans(planner)
    scheme = sliding_scheme.SlidingWindowScheme()
    rest = scheme.plan_operation(
        "stream", 8, commit_round_count=3, buffer_round_count=2
    )
    rest_commits = _shifted_spans(rest.windows, 4)
    assert commits[:2] == [(1, 3), (4, 4)]
    assert commits[2:] == rest_commits


def test_a_cut_in_a_finite_stream_s_laid_last_window_plans_the_rest():
    """The rounds after a cut inside the last window, laid, get windows."""
    source = _FiniteSource(12)
    planner = _planner(source)
    stream = _stream()
    planner.register_stream(stream, None)
    planner.grow_stream("stream", 10, None)
    planner.cut_stream_after("stream", 10)
    planner.grow_stream("stream", 12, None)
    commits = _commit_spans(planner)
    assert commits == [(1, 3), (4, 6), (7, 10), (11, 12)]


def test_a_finite_stream_replan_keeps_the_cuts_bound_before_it():
    """A cut at round 4 replans the rest and still ends a window on 8.

    The segment bound first, after round 8, keeps its first round on a
    window's first round: each stretch between cuts is its own plan.
    """
    source = _FiniteSource(12)
    planner = _planner(source)
    stream = _stream()
    planner.register_stream(stream, None)
    planner.cut_stream_after("stream", 8)
    planner.cut_stream_after("stream", 4)
    planner.grow_stream("stream", 12, None)
    commits = _commit_spans(planner)
    scheme = sliding_scheme.SlidingWindowScheme()
    stretch = scheme.plan_operation(
        "stream", 4, commit_round_count=3, buffer_round_count=2
    )
    first_stretch = _shifted_spans(stretch.windows, 4)
    second_stretch = _shifted_spans(stretch.windows, 8)
    assert commits[:2] == [(1, 3), (4, 4)]
    assert commits[2:] == first_stretch + second_stretch


def _commit_spans(planner: window_planner.WindowPlanner) -> list:
    """(commit_lo, commit_hi) of each window of the stream, in order."""
    return [
        (window.commit_lo, window.commit_hi)
        for window in planner.windows_of("stream")
    ]


def _spans_after(spans: list, last_round: int) -> list:
    """The spans that commit only rounds after the given one."""
    return [span for span in spans if span[0] > last_round]


def _rounds_covered(spans: list) -> list:
    """Every round the spans commit, in their order."""
    covered = []
    for commit_lo, commit_hi in spans:
        stop = commit_hi + 1
        covered.extend(range(commit_lo, stop))
    return covered


def _shifted_spans(geometries: tuple, round_shift: int) -> list:
    """The geometries' commit spans, later by the shift."""
    spans = []
    for geometry in geometries:
        commit_lo = geometry.commit_lo + round_shift
        commit_hi = geometry.commit_hi + round_shift
        spans.append((commit_lo, commit_hi))
    return spans


def test_idle_rounds_fold_only_into_a_batch_style_operation():
    plan = _empty_plan()
    window = window_records.Window(
        operation_id=7,
        window_index=0,
        commit_lo=1,
        commit_hi=6,
        buffer_hi=6,
        round_count=6,
    )
    plan.windows[(7, 0)] = window
    plan.batch_preceding_idle_rounds_by_operation[7] = True
    built = built_window_models.BuiltWindowModels()
    models = window_planner.WindowModels(built)
    models.decoder = _Decoder()
    scheme = naive_online_scheme.NaiveOnlineScheme()
    resolved = [_resolved(7, 6)]
    planner = window_planner.WindowPlanner(resolved, plan, ())
    planner.scheme = scheme
    planner.models = models
    planner.start()
    planner.prepend_idle_rounds(7, 4)
    planner.prepend_idle_rounds(7, 0)
    assert window.batched_preceding_idle_round_count == 4
