"""The window records of decsim/records/windows.py.

Rounds are 1-based and ranges inclusive, so a window reads
[start_round, buffer_hi] and commits [commit_lo, commit_hi]; the
read-only view a policy sees copies the window's mutable topology.
"""

import decsim.records.windows as window_records


def test_window_start_round_and_key_reflect_runtime_geometry():
    """A window starts at its leading buffer when it has one."""
    window = window_records.Window(4, 2, 3, 5, 6, 4, buffer_lo=1)
    assert window.start_round == 1
    assert window.key == (4, 2)

    window.buffer_lo = None
    assert window.start_round == 3


def test_window_info_snapshots_topology_and_detector_positions():
    """The view copies the mutable topology and detector positions."""
    window = window_records.Window(4, 2, 3, 5, 6, 4, buffer_lo=1)
    window.deps.append((4, 1))
    window.dependents.append((4, 3))
    detector_positions = {7: (1, 2)}

    info = window_records.WindowInfo.from_window(
        window,
        detector_positions=detector_positions,
    )
    window.deps.append((4, 0))
    detector_positions[8] = (2, 3)

    assert info.start_round == 1
    assert info.deps == ((4, 1),)
    assert info.dependents == ((4, 3),)
    assert info.detector_positions == {7: (1, 2)}


def test_a_window_view_without_detector_positions_carries_none():
    """A policy that needs no positions gets None, not an empty mapping."""
    window = window_records.Window(4, 2, 3, 5, 6, 4)
    info = window_records.WindowInfo.from_window(window)
    assert info.detector_positions is None


def test_window_geometry_counts_the_rounds_between_both_buffers():
    """The interval's round count spans the leading and trailing buffers."""
    geometry = window_records.WindowGeometry(
        buffer_lo=2, commit_lo=3, commit_hi=5, buffer_hi=7
    )
    assert geometry.round_count == 6


def test_compiled_window_plan_carries_mutable_manager_mappings():
    """The compiled plan's mappings stay mutable for the window manager."""
    plan = window_records.WindowPlan(
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
    plan.window_count[4] = 2
    assert plan.window_count == {4: 2}


def test_boundary_delivery_is_current_only_at_both_latest_revisions():
    """Freshness requires the current source and the current delivery."""
    values = {
        "source_key": (1, 0),
        "destination_key": (1, 1),
        "source_revision": 2,
        "delivery_revision": 3,
        "latest_source_revision": 2,
        "latest_delivery_revision": 3,
        "source_operation_round_count": 5,
        "dependency_released": False,
        "payload": object(),
    }
    current = window_records.BoundaryDelivery(**values)
    stale_delivery_values = dict(values)
    stale_delivery_values["latest_delivery_revision"] = 4
    stale_delivery = window_records.BoundaryDelivery(**stale_delivery_values)
    stale_source_values = dict(values)
    stale_source_values["latest_source_revision"] = 3
    stale_source = window_records.BoundaryDelivery(**stale_source_values)
    assert current.is_current
    assert not stale_delivery.is_current
    assert not stale_source.is_current


def test_a_request_key_is_a_window_a_tier_and_a_run_ordinal():
    """The key's fields are the operation, the window, the tier, the ordinal."""
    request_key = window_records.DecoderRequestKey(
        (4, "op"), 2, window_records.DecoderTier.STRONG, 7
    )
    assert request_key.operation_id == (4, "op")
    assert request_key.window_id == 2
    assert request_key.tier is window_records.DecoderTier.STRONG
    assert request_key.run_sequence == 7


def test_two_requests_for_one_window_and_tier_differ_by_their_ordinal():
    """A retry of the same window and tier is a different request."""
    first = window_records.DecoderRequestKey(
        4, 2, window_records.DecoderTier.WEAK, 7
    )
    retry = window_records.DecoderRequestKey(
        4, 2, window_records.DecoderTier.WEAK, 8
    )
    assert first != retry
    assert len({first, retry}) == 2


def test_strong_context_is_one_buffer_on_each_side_of_the_commit():
    """A strong redo reads a buffer region past each end of its commit."""
    window = window_records.Window(
        operation_id=1,
        window_index=2,
        commit_lo=7,
        commit_hi=9,
        buffer_hi=11,
        round_count=5,
    )
    bounds = window_records.strong_context_bounds(window)
    assert bounds == (5, 7, 9, 11)


def test_a_strong_context_is_clipped_at_the_operations_first_round():
    """The context cannot start before the operation's first round."""
    window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
    )
    bounds = window_records.strong_context_bounds(window)
    assert bounds == (1, 1, 3, 5)
