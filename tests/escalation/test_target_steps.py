"""The outer loop stepping the share by the backlog (target_steps.py).

The referent is the rule worked by hand: checks every 2 windows, down a
level when the most windows waiting since the last check is above 3, up
after 2 checks in a row at or under 1, holding between; a bad setting
is refused by name.
"""

import pytest

import decsim.escalation.target_steps as target_steps

LEVELS = (0.1, 0.2, 0.3)


def _steps(start: float) -> target_steps.TargetSteps:
    settings = target_steps.TargetSteps.Settings(
        start=start,
        levels=LEVELS,
        check_every_windows=2,
        high_backlog_windows=3,
        low_backlog_windows=1,
        calm_checks=2,
    )
    return settings.build()


def _shares(steps, waiting_counts: list) -> list:
    """The share after each window."""
    shares = []
    for waiting in waiting_counts:
        steps.observe(waiting)
        shares.append(steps.target)
    return shares


def test_the_share_steps_down_on_a_backlog_and_up_after_calm_checks():
    steps = _steps(0.2)

    # check 1 peaks 5: down; check 2 peaks 2: hold; checks 3 and 4 calm: up
    shares = _shares(steps, [0, 5, 2, 0, 1, 0, 0, 1])

    assert shares == [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    assert steps.trajectory == [
        (2, 0.1, 5),
        (4, 0.1, 2),
        (6, 0.1, 1),
        (8, 0.2, 1),
    ]


def test_the_share_stays_inside_the_levels():
    low = _steps(0.1)
    high = _steps(0.3)

    _shares(low, [9, 9])
    _shares(high, [0, 0, 0, 0])

    assert (low.target, high.target) == (0.1, 0.3)


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"start": 0.15}, "not one of the levels"),
        ({"levels": (0.3, 0.1)}, "must rise"),
        ({"high_backlog_windows": 1}, "must be above"),
        ({"check_every_windows": 0}, "check_every_windows"),
    ],
)
def test_a_bad_setting_is_refused_by_name(changes, words):
    arguments = {"start": 0.1, "levels": LEVELS, **changes}

    with pytest.raises(ValueError, match=words):
        target_steps.TargetSteps.Settings(**arguments)
