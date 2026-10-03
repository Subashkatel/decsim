"""The window records of decsim/records/windows.py.

Rounds are 1-based and ranges inclusive, so a window reads
[start_round, buffer_hi] and commits [commit_lo, commit_hi]; the
read-only view a policy sees copies the window's mutable topology.
"""

import decsim.records.windows as window_records


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


def test_a_strong_context_starts_at_the_commit_and_ends_a_buffer_past_it():
    """A pinned past face reads no round behind the commit (Bombin 1456)."""
    window = window_records.Window(
        operation_id=1,
        window_index=2,
        commit_lo=7,
        commit_hi=9,
        buffer_hi=11,
        round_count=5,
    )
    bounds = window_records.strong_context_bounds(window)
    assert bounds == (7, 7, 9, 11)
