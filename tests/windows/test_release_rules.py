"""Which decodes a blocker's release waits for (windows/release_rules.py).

The operations form one chain into a blocker, 1 then 2 then the
blocker 3, which 4 waits on, and a second chain into the blocker 7,
which 8 waits on, through 6, an operation that decodes no rounds, as a
T injection's does not in SWIPER (window_builder.py build_windows).
"""

import decsim.windows.release_rules as release_rules
import tests.declared_run as declared_run


def _operations() -> tuple:
    def operation(operation_id, *predecessors, **changes):
        return declared_run.memory_operation(
            operation_id, predecessors=predecessors, **changes
        )

    return (
        operation(1),
        operation(2, 1),
        operation(3, 2),
        operation(4, 3, blocked_by=3),
        operation(5),
        operation(6, 5, emits_detector_data=False),
        operation(7, 6),
        operation(8, 7, blocked_by=7),
    )


OPERATIONS = _operations()


def test_own_streams_waits_for_the_blockers_segments_alone():
    settings = release_rules.OwnStreams.Settings()
    rule = settings.build(OPERATIONS)

    assert rule.awaited_by(3) == (3,)
    assert rule.awaiting(3) == (3,)


def test_a_blocker_waits_for_itself_and_each_decoded_ancestor():
    settings = release_rules.DecodedAncestors.Settings()
    rule = settings.build(OPERATIONS)

    awaited = rule.awaited_by(3)

    assert awaited[0] == 3
    assert sorted(awaited) == [1, 2, 3]


def test_an_ancestors_commit_may_release_the_blockers_it_precedes():
    rule = release_rules.DecodedAncestors(OPERATIONS)

    assert rule.awaiting(1) == (3,)
    assert rule.awaiting(4) == ()


def test_an_operation_that_decodes_nothing_ends_the_walk():
    rule = release_rules.DecodedAncestors(OPERATIONS)

    assert rule.awaited_by(7) == (7,)
    assert rule.awaiting(5) == ()
