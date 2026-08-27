"""Behavior tests for dynamic timing-window lifecycle coordination."""

from types import SimpleNamespace

from decsim.message import DecoderTier

import pytest

from decsim.windows.dynamic_windows import DynamicWindows
from decsim.windows.window_boundaries import BoundaryCourier
from decsim.windows.window_manager import WindowManager


class RecordingWindowManager:
    def __init__(self):
        self.rounds_arrived = {}
        self.windows = {}
        self.committed_windows = set()
        self.events = []
        self.fail_creation = False
        self.lifecycle = None

    def create_dynamic_window(self, stream_id, window_index, commit_lo,
                              commit_hi, buffer_hi):
        self.events.append(("create", stream_id, window_index, commit_lo,
                            commit_hi, buffer_hi))
        if self.fail_creation:
            raise RuntimeError("creation failed")

    def validate_stream_length(self, stream_id, stream_round_count):
        self.events.append(("validate", stream_id, stream_round_count))

    def trim_dynamic_window_tail(self, stream_id, stream_round_count,
                                 buffer_rounds):
        self.events.append(("trim", stream_id, stream_round_count,
                            buffer_rounds))

    def refresh_unqueued_stream_windows(self, stream_id):
        self.events.append(("refresh", stream_id))

    def check_windows_for_operation(self, stream_id):
        sealed = None if self.lifecycle is None else self.lifecycle.sealed(stream_id)
        self.events.append(("check", stream_id, sealed))

    def finish_workload_if_ready(self):
        self.events.append(("finish",))

    def release_stream_segments_at_commit(self, stream_id, committed):
        self.events.append(("release", stream_id, committed))


class HoldRecorder:
    def __init__(self):
        self.replacements = []

    def replace_hold(self, key, reads):
        self.replacements.append((key, reads))


def operation(identity):
    return SimpleNamespace(id=identity)


def geometry(commit_lo, commit_hi, buffer_hi):
    return SimpleNamespace(
        commit_lo=commit_lo,
        commit_hi=commit_hi,
        buffer_hi=buffer_hi,
    )


def register(lifecycle, stream_id="stream", *, commit=3, buffer=2,
             source_limit=None, finite_geometries=None):
    lifecycle.register(
        operation(stream_id),
        commit_round_count=commit,
        buffer_round_count=buffer,
        source_round_limit=source_limit,
        finite_geometries=finite_geometries,
    )


def test_registration_keeps_aliases_and_applies_the_simplified_ownership():
    """Registration trusts inputs while obsolete shared-state surfaces stay removed."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    geometries = [geometry(1, 3, 5)]

    register(lifecycle, commit="three", buffer=-2,
             source_limit="cap", finite_geometries=geometries)
    lifecycle.closed_boundaries["stream"].add(7)
    lifecycle.committed_round_counts["stream"] = 4
    register(lifecycle, commit=5, buffer=1,
             source_limit=None, finite_geometries=geometries)

    assert lifecycle.window_manager is manager
    assert lifecycle.has("stream")
    assert lifecycle.closed_boundaries["stream"] == {7}
    assert lifecycle.committed_round_counts["stream"] == 4
    assert lifecycle._streams["stream"] == {
        "commit_rounds": 5,
        "buffer_rounds": 1,
        "next_window": 0,
        "sealed": False,
        "source_round_limit": None,
        "sealed_round_count": None,
        "finite_geometries": geometries,
    }
    assert lifecycle._streams["stream"]["finite_geometries"] is geometries
    assert not hasattr(lifecycle, "state")
    assert not hasattr(lifecycle, "unsealed")
    assert not hasattr(lifecycle, "segment_results_sent")


def test_real_manager_constructs_with_segment_delivery_state_in_its_owner():
    """The real manager, rather than its lifecycle helper, owns sent segments."""
    manager = WindowManager(
        SimpleNamespace(),
        scheme=SimpleNamespace(),
        code_geometry=SimpleNamespace(),
        resolved_operations=(),
        resolved_patches=(),
        links=SimpleNamespace(),
        conditional_release=SimpleNamespace(),
        boundary_policy=SimpleNamespace(),
        window_interaction=SimpleNamespace(),
        planning_view_by_operation_id={},
        fault_model_requirement_for=lambda _code_name: None,
        retain_strong_context=False,
        double_window=False,
        escalation_policy=SimpleNamespace(
            primary_tier=DecoderTier.WEAK),
        submit_fn=lambda job, reserve_transfer=None: None,
        check_strong_route=lambda weak_job, strong_job: None,
        on_workload_complete=lambda: None,
    )

    assert manager.segment_results_sent == set()
    assert not hasattr(manager.lifecycle, "segment_results_sent")
    assert not hasattr(manager.lifecycle, "unsealed")


def test_nonstream_queries_and_transitions_are_inert():
    """Static operation identities incur no dynamic-stream lifecycle obligations."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)

    assert lifecycle.has("static") is False
    assert lifecycle.sealed("static") is True
    assert lifecycle.arrival_round_limit("static", 11) == 11
    assert lifecycle.round_count_for_window("static", None, 12) == 12
    lifecycle.maybe_update("static")
    lifecycle.close_boundary("static", -10)
    assert manager.events == []
    with pytest.raises(TypeError):
        lifecycle.has([])


def test_round_count_views_follow_open_capped_unbounded_and_sealed_states():
    """Round-limit views select the appropriate static, open, capped, or sealed extent."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle, stream_id="open")
    register(lifecycle, stream_id="capped", source_limit=9,
             finite_geometries=[])
    manager.rounds_arrived.update(open=4, capped=3)

    assert lifecycle.arrival_round_limit("open", 99) is None
    assert lifecycle.arrival_round_limit("capped", 99) == 9
    assert lifecycle.round_count_for_window(
        "open", SimpleNamespace(buffer_hi=8), 99) == 8
    assert lifecycle.round_count_for_window("open", None, 99) == 4
    assert lifecycle.round_count_for_window("capped", None, 99) == 9
    register(lifecycle, stream_id="raw", source_limit="nine",
             finite_geometries=[])
    assert lifecycle.arrival_round_limit("raw", -1) == "nine"
    assert lifecycle.round_count_for_window("raw", None, -1) == "nine"
    lifecycle.seal("open", 4)
    assert lifecycle.arrival_round_limit("open", 99) == 4
    assert lifecycle.round_count_for_window("open", None, 99) == 4


def test_arithmetic_windows_grow_online_with_commit_and_buffer_extents():
    """Arriving high-water marks create each newly begun arithmetic timing window once."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    register(lifecycle)
    manager.rounds_arrived["stream"] = 1

    lifecycle.grow("stream")
    manager.rounds_arrived["stream"] = 7
    lifecycle.grow("stream")
    lifecycle.grow("stream")

    assert manager.events == [
        ("create", "stream", 0, 1, 3, 5),
        ("create", "stream", 1, 4, 6, 8),
        ("create", "stream", 2, 7, 9, 11),
    ]


def test_finite_geometry_growth_trusts_order_and_advances_after_creation():
    """Finite timing geometries remain ordered aliases and advance only after creation succeeds."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    geometries = [geometry(4, 6, 8), geometry(1, 3, 5)]
    register(lifecycle, finite_geometries=geometries)

    lifecycle.grow("stream", rounds_to_plan=1)
    assert manager.events == []
    manager.fail_creation = True
    with pytest.raises(RuntimeError, match="creation failed"):
        lifecycle.grow("stream", rounds_to_plan=4)
    manager.fail_creation = False
    lifecycle.grow("stream", rounds_to_plan=4)

    assert manager.events[-2:] == [
        ("create", "stream", 0, 4, 6, 8),
        ("create", "stream", 1, 1, 3, 5),
    ]


def test_growth_and_commit_end_deliberately_trust_malformed_geometry_values():
    """The coordinator forwards malformed timing geometry instead of adding local validation."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    malformed = [geometry(-2, -5, -9)]
    register(lifecycle, commit=-3, buffer=-4, source_limit=-7,
             finite_geometries=malformed)

    lifecycle.grow("stream", rounds_to_plan=0)
    assert manager.events == [
        ("create", "stream", 0, -2, -5, -9),
    ]
    state = {
        "commit_rounds": 3,
        "sealed_round_count": -4,
        "source_round_limit": 10,
    }
    assert lifecycle._commit_hi(state, 1) == -4

    looping_manager = RecordingWindowManager()
    original_create = looping_manager.create_dynamic_window

    def stop_unbounded_growth(*args, **kwargs):
        original_create(*args, **kwargs)
        if len(looping_manager.events) == 3:
            raise RuntimeError("bounded test stop")

    looping_manager.create_dynamic_window = stop_unbounded_growth
    looping_lifecycle = DynamicWindows(looping_manager)
    register(looping_lifecycle, commit=0, buffer=1)
    with pytest.raises(RuntimeError, match="bounded test stop"):
        looping_lifecycle.grow("stream", rounds_to_plan=1)
    assert [event[2] for event in looping_manager.events] == [0, 1, 2]


def test_arrival_update_grows_then_auto_seals_at_a_finite_source_limit():
    """A capped stream grows online and performs its finite seal transition at the cap."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    finite = [geometry(1, 3, 5), geometry(4, 5, 5)]
    register(lifecycle, source_limit=5, finite_geometries=finite)
    manager.rounds_arrived["stream"] = 5

    lifecycle.maybe_update("stream")

    assert manager.events == [
        ("create", "stream", 0, 1, 3, 5),
        ("create", "stream", 1, 4, 5, 5),
        ("validate", "stream", 5),
        ("check", "stream", True),
        ("finish",),
    ]
    assert lifecycle.sealed("stream") is True


def test_successful_seal_preserves_geometry_and_repeated_calls_are_noops():
    """A successful seal preserves event geometry and every later seal is a no-op."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle)

    lifecycle.seal("stream", 5)

    expected = [
        ("validate", "stream", 5),
        ("create", "stream", 0, 1, 3, 5),
        ("create", "stream", 1, 4, 5, 7),
        ("trim", "stream", 5, 2),
        ("check", "stream", True),
        ("finish",),
    ]
    assert manager.events == expected
    assert lifecycle._streams["stream"]["next_window"] == 2
    assert lifecycle._streams["stream"]["sealed_round_count"] == 5
    assert lifecycle.sealed("stream") is True

    lifecycle.grow("stream")
    lifecycle.seal("stream", 99)
    lifecycle.seal("stream", 5)
    assert manager.events == expected
    with pytest.raises(KeyError):
        lifecycle.grow("missing")
    with pytest.raises(KeyError):
        lifecycle.seal("missing", 1)


def test_preflight_failure_leaves_local_and_delegated_state_unchanged():
    """A pure length-validation failure leaves local and delegated state unchanged."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle)
    manager.rounds_arrived["stream"] = 4
    initial_stream_state = lifecycle._streams["stream"].copy()

    def reject_length(_stream_id, _stream_round_count):
        raise ValueError("invalid stream length")

    manager.validate_stream_length = reject_length
    with pytest.raises(ValueError, match="invalid stream length"):
        lifecycle.seal("stream", 5)

    assert lifecycle._streams["stream"] == initial_stream_state
    assert lifecycle.sealed("stream") is False
    assert lifecycle.arrival_round_limit("stream", 99) is None
    assert lifecycle.round_count_for_window("stream", None, 99) == 4
    assert manager.events == []
    assert manager.windows == {}


def test_real_manager_clips_only_the_first_containing_tail_window():
    """Tail clipping mutates one manager-owned window and refreshes its exact weak reads."""
    manager = object.__new__(WindowManager)
    first = SimpleNamespace(commit_lo=1, commit_hi=8, buffer_hi=10,
                            start_round=1, n_rounds=10)
    second = SimpleNamespace(commit_lo=4, commit_hi=9, buffer_hi=11,
                             start_round=4, n_rounds=8)
    manager.op_windows = {"stream": [0, 1]}
    manager.windows = {("stream", 0): first, ("stream", 1): second}
    manager.syndrome_buffer = HoldRecorder()

    manager.trim_dynamic_window_tail("stream", 5, 2)

    assert (first.commit_hi, first.buffer_hi, first.n_rounds) == (5, 7, 7)
    assert (second.commit_hi, second.buffer_hi, second.n_rounds) == (9, 11, 8)
    assert manager.syndrome_buffer.replacements == [
        (("stream", 0), [("stream", round_index) for round_index in range(1, 6)])
    ]
    manager.trim_dynamic_window_tail("stream", 20, 2)
    assert len(manager.syndrome_buffer.replacements) == 1


def test_boundary_closure_guards_finite_sources_and_rejects_duplicates():
    """Boundary closure rejects invalid bounds and duplicates before any mutation."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle, finite_geometries=[])
    with pytest.raises(ValueError, match="must be >= 1"):
        lifecycle.close_boundary("stream", 0)
    with pytest.raises(TypeError):
        lifecycle.close_boundary("stream", "four")

    lifecycle.close_boundary("stream", 4)
    initial_stream_state = lifecycle._streams["stream"].copy()
    initial_boundaries = lifecycle.closed_boundaries["stream"].copy()
    initial_events = list(manager.events)
    with pytest.raises(RuntimeError, match="already closed.*round 4"):
        lifecycle.close_boundary("stream", 4)
    assert lifecycle._streams["stream"] == initial_stream_state
    assert lifecycle.closed_boundaries["stream"] == initial_boundaries
    assert manager.events == initial_events == [
        ("refresh", "stream"), ("check", "stream", False),
    ]

    lifecycle.seal("stream", 0)
    manager.events.clear()
    lifecycle.close_boundary("stream", 5)
    assert manager.events == [
        ("refresh", "stream"), ("check", "stream", True),
    ]

    finite_manager = RecordingWindowManager()
    finite_lifecycle = DynamicWindows(finite_manager)
    register(finite_lifecycle, source_limit=5, finite_geometries=[])
    with pytest.raises(RuntimeError, match="destructive boundary"):
        finite_lifecycle.close_boundary("stream", 4)
    assert finite_lifecycle.closed_boundaries["stream"] == set()
    finite_lifecycle.close_boundary("stream", 6)
    assert finite_lifecycle.closed_boundaries["stream"] == {6}


def test_duplicate_boundaries_are_scoped_per_stream_and_round():
    """A duplicate blocks only the same stream and round while other keys remain valid."""
    manager = RecordingWindowManager()
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle, stream_id="first", finite_geometries=[])
    register(lifecycle, stream_id="second", finite_geometries=[])

    lifecycle.close_boundary("first", 4)
    with pytest.raises(RuntimeError, match="already closed.*round 4"):
        lifecycle.close_boundary("first", 4)
    lifecycle.close_boundary("first", 5)
    lifecycle.close_boundary("second", 4)

    assert lifecycle.closed_boundaries == {
        "first": {4, 5},
        "second": {4},
    }
    assert manager.events == [
        ("refresh", "first"), ("check", "first", False),
        ("refresh", "first"), ("check", "first", False),
        ("refresh", "second"), ("check", "second", False),
    ]


def test_closed_boundary_selection_uses_the_right_buffer_half_open_interval():
    """A window selects the earliest closed boundary in its right buffer only."""
    lifecycle = DynamicWindows(RecordingWindowManager())
    register(lifecycle)
    lifecycle.closed_boundaries["stream"].update({2, 3, 4, 6, 7})
    window = SimpleNamespace(op_id="stream", commit_hi=3, buffer_hi=7)

    assert lifecycle.closed_boundary_for_window(window) == 3
    lifecycle.closed_boundaries["stream"] = {2, 7}
    assert lifecycle.closed_boundary_for_window(window) is None


def test_real_manager_links_overlapping_online_windows_by_temporal_dependency():
    """Real manager creation links each later overlapping timing window to its predecessor."""
    manager = object.__new__(WindowManager)
    manager.window_interaction = SimpleNamespace(
        initial_boundary_state=lambda _window_info: "initial")
    manager.committed_windows = set()
    manager.courier = BoundaryCourier(manager)
    manager.windows = {}
    manager.op_windows = {"stream": []}
    manager.window_count = {"stream": 0}
    manager.total_windows = 0
    manager.error_model_provider = None
    manager._add_window_read_refs = lambda _key, _window: None
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle)

    lifecycle.grow("stream", rounds_to_plan=4)

    first = manager.windows[("stream", 0)]
    second = manager.windows[("stream", 1)]
    assert (first.commit_lo, first.commit_hi, first.buffer_hi) == (1, 3, 5)
    assert (second.commit_lo, second.commit_hi, second.buffer_hi) == (4, 6, 8)
    assert second.deps == [("stream", 0)]
    assert second.deps_remaining == 1
    assert first.dependents == [("stream", 1)]


def test_committed_prefix_stops_at_gaps_and_absorbs_adjacent_overlaps():
    """Committed timing windows extend a contiguous prefix from round one and stop at gaps."""
    manager = RecordingWindowManager()
    manager.windows = {
        ("stream", 0): SimpleNamespace(commit_lo=1, commit_hi=3),
        ("stream", 1): SimpleNamespace(commit_lo=3, commit_hi=6),
        ("stream", 2): SimpleNamespace(commit_lo=8, commit_hi=10),
        ("other", 0): SimpleNamespace(commit_lo=1, commit_hi=99),
    }
    manager.committed_windows = {
        ("stream", 2), ("other", 0), ("stream", 1), ("stream", 0)
    }
    lifecycle = DynamicWindows(manager)

    assert lifecycle._committed_prefix_round_count("stream") == 6
    manager.committed_windows.add(("stream", 99))
    with pytest.raises(KeyError):
        lifecycle._committed_prefix_round_count("stream")


def test_committed_round_count_reads_the_exact_cache_without_manager_events():
    """The committed-prefix query exposes exact cache state without manager side effects."""
    manager = RecordingWindowManager()
    manager.windows = {
        ("stream", 0): SimpleNamespace(commit_lo=1, commit_hi=3),
        ("stream", 1): SimpleNamespace(commit_lo=4, commit_hi=6),
    }
    manager.committed_windows = {("stream", 0)}
    lifecycle = DynamicWindows(manager)
    register(lifecycle)

    assert lifecycle.committed_round_count("missing") == 0
    assert lifecycle.committed_round_count("stream") == 0
    lifecycle.update_committed_round_count("stream")
    assert lifecycle.committed_round_count("stream") == 3
    manager.committed_windows.add(("stream", 1))
    lifecycle.update_committed_round_count("stream")
    manager.committed_windows.remove(("stream", 1))
    assert lifecycle.committed_round_count("stream") == 6
    assert lifecycle.recompute_committed_round_count("stream") == 3
    assert lifecycle.committed_round_count("stream") == 3
    manager.committed_windows.clear()
    assert lifecycle.recompute_committed_round_count("stream") == 0
    assert lifecycle.committed_round_count("stream") == 0
    expected_events = [
        ("release", "stream", 3),
        ("release", "stream", 6),
    ]
    assert manager.events == expected_events
    with pytest.raises(TypeError):
        lifecycle.committed_round_count([])
    assert manager.events == expected_events


def test_committed_prefix_updates_monotonically_and_recompute_can_regress():
    """Ordinary commits advance once while recovery recomputation may lower or clear the cache."""
    manager = RecordingWindowManager()
    manager.windows = {
        ("stream", 0): SimpleNamespace(commit_lo=1, commit_hi=3),
        ("stream", 1): SimpleNamespace(commit_lo=4, commit_hi=6),
    }
    manager.committed_windows = {("stream", 0)}
    lifecycle = DynamicWindows(manager)

    lifecycle.update_committed_round_count("stream")
    lifecycle.update_committed_round_count("stream")
    manager.committed_windows.add(("stream", 1))
    lifecycle.update_committed_round_count("stream")
    manager.committed_windows = {("stream", 0)}
    lifecycle.update_committed_round_count("stream")
    assert manager.events == [
        ("release", "stream", 3),
        ("release", "stream", 6),
    ]
    assert lifecycle.committed_round_counts["stream"] == 6

    assert lifecycle.recompute_committed_round_count("stream") == 3
    assert lifecycle.committed_round_counts["stream"] == 3
    manager.committed_windows.clear()
    assert lifecycle.recompute_committed_round_count("stream") == 0
    assert "stream" not in lifecycle.committed_round_counts
    assert manager.events == [
        ("release", "stream", 3),
        ("release", "stream", 6),
    ]


def test_unsealed_streams_block_real_manager_workload_finality_until_seal():
    """Real manager finality remains blocked until every registered timing stream seals."""
    manager = object.__new__(WindowManager)
    manager._workload_complete_sent = False
    manager.committed_windows = set()
    manager.total_windows = 0
    manager._pending_strong_windows = set()
    manager.speculative_recovery = SimpleNamespace(has_finality_blockers=False)
    completions = []
    manager.on_workload_complete = lambda: completions.append("complete")
    manager.error_model_provider = None
    manager.check_windows_for_operation = lambda _stream_id: None
    lifecycle = DynamicWindows(manager)
    manager.lifecycle = lifecycle
    register(lifecycle, finite_geometries=[])

    manager.finish_workload_if_ready()
    assert completions == []
    manager.seal_stream("stream", 0)
    assert completions == ["complete"]
    assert manager.has_dynamic_stream("stream") is True
    assert manager.committed_stream_round_count("stream") == 0
