"""When a committed window ships its boundary to the windows after it.

The three rows answer one question: eager ships at every commit, held
ships only when the committing result is final, and checked ships at
every commit and again when the result is final. A provisional boundary
that ships and is never checked again is never revised, so a run whose
weak result the strong tier may replace needs the held or the checked
row (Toshio et al. 2510.25222 Sec. III A); the escalation policy's own
check_plan refuses the wrong pairing, and that refusal is pinned in
tests/escalation/test_policies.py.
"""

import decsim.windows.boundary_policies as boundary_policies


def test_eager_ships_a_provisional_boundary():
    eager = boundary_policies.Eager()
    window = object()
    assert eager.on_commit(window, final=False) is True


def test_held_holds_a_provisional_boundary():
    held = boundary_policies.Held()
    window = object()
    assert held.on_commit(window, final=False) is False


def test_checked_ships_a_provisional_boundary():
    checked = boundary_policies.Checked()
    window = object()
    assert checked.on_commit(window, final=False) is True


def test_only_checked_checks_a_shipped_boundary_again():
    rows = (
        boundary_policies.Eager(),
        boundary_policies.Held(),
        boundary_policies.Checked(),
    )
    rechecks = [row.rechecks_provisional_boundaries for row in rows]
    assert rechecks == [False, False, True]
