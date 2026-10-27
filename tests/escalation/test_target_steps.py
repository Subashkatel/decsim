"""The outer loop stepping the share by the backlog's trend (target_steps.py).

The referent is the rule worked by hand: checks every 2 windows, down a
level when the mean windows waiting rose at each of the last 2 checks
and by more than 3 over them, up after 2 checks with no such rise when
the mean is no higher than 2 checks before, holding otherwise; a bad
setting is refused by name.
"""

import pytest

import decsim.escalation.target_steps as target_steps

LEVELS = (0.1, 0.2, 0.3)


def _steps(start: float) -> target_steps.TargetSteps:
    settings = target_steps.TargetSteps.Settings(
        start=start,
        levels=LEVELS,
        check_every_windows=2,
        rise_checks=2,
        rise_margin_windows=3.0,
        calm_checks=2,
    )
    return settings.build()


def _shares(steps, waiting_counts: list) -> list:
    """The share after each check."""
    shares = []
    for index, waiting in enumerate(waiting_counts, start=1):
        steps.observe(waiting)
        if index % 2 == 0:
            shares.append(steps.target)
    return shares


def test_a_rising_backlog_steps_down_and_a_flat_one_steps_back_up():
    steps = _steps(0.2)

    # check 2: calm twice, but no mean 2 checks before it: hold. check 3:
    # 1, 3, 6 rose twice by 5: down. check 5: calm twice, 6 no higher
    # than check 3's 6: up. check 7: calm twice, 2 under check 5's 6: up
    waiting = [1, 1, 3, 3, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2]
    shares = _shares(steps, waiting)

    assert shares == [0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3]
    means = [mean for _windows, _share, mean in steps.trajectory]
    assert means == [1, 3, 6, 6, 6, 2, 2]


def test_a_rise_under_the_margin_holds():
    steps = _steps(0.2)

    shares = _shares(steps, [1, 1, 2, 2, 3, 3])

    assert shares == [0.2, 0.2, 0.2]


def test_the_share_stays_inside_the_levels():
    low = _steps(0.1)
    high = _steps(0.3)

    _shares(low, [0, 0, 5, 5, 10, 10])
    _shares(high, [0, 0, 0, 0, 0, 0])

    assert (low.target, high.target) == (0.1, 0.3)


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"start": 0.15}, "not one of the levels"),
        ({"levels": (0.3, 0.1)}, "must rise"),
        ({"rise_margin_windows": -1.0}, "rise_margin_windows"),
        ({"check_every_windows": 0}, "check_every_windows"),
    ],
)
def test_a_bad_setting_is_refused_by_name(changes, words):
    arguments = {"start": 0.1, "levels": LEVELS, **changes}

    with pytest.raises(ValueError, match=words):
        target_steps.TargetSteps.Settings(**arguments)
