"""The windows section: the tables it resolves and the two keys it refuses.

STYLE.md rule 7: a pluggable component's section carries one kind key
naming a row of the root's table. This section carries four such keys,
and two of them, terminal_policy and boundaries, arrived in I7 Part 1
with a null default whose meaning is decided later, so the refusal each
one raises at the yaml boundary is what a user meets first.
"""

import pytest

import decsim.config as config
import decsim.records.windows as window_records
import decsim.windows.settings as window_settings

CLOCKS = config.ClockSettings({"decisions": 250.0})


def _section(**overrides) -> dict:
    """A complete windows section, with the given keys overridden."""
    section = {"kind": "sliding", "commit_rounds": None, "buffer_rounds": None}
    section.update(overrides)
    return section


def test_every_windowing_scheme_row_has_one_class():
    """The table is the plug point: one name, one row (STYLE.md rule 7)."""
    rows = window_settings.WINDOWING_SCHEMES
    row_classes = rows.values()
    distinct_classes = set(row_classes)

    assert sorted(rows) == ["naive_online", "parallel", "sandwich", "sliding"]
    assert len(distinct_classes) == len(rows)


def test_a_terminal_policy_that_is_not_one_of_the_two_words_is_refused():
    section = _section(terminal_policy="drain")

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    sentence = str(refusal.value)
    assert "windows.terminal_policy is one of" in sentence
    assert "'drain'" in sentence


def test_both_terminal_policies_of_the_record_are_accepted():
    for policy in window_records.TERMINAL_POLICIES:
        section = _section(terminal_policy=policy)
        settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

        assert settings.terminal_policy == policy


def test_a_boundaries_key_that_names_no_row_is_refused():
    section = _section(boundaries="lazy")

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    sentence = str(refusal.value)
    assert "windows.boundaries" in sentence
    assert "lazy" in sentence


def test_both_boundary_policy_rows_are_reachable_by_name():
    for name in window_settings.BOUNDARY_POLICIES:
        section = _section(boundaries=name)
        settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

        assert settings.boundaries == name


def test_both_keys_default_to_null_so_the_plan_decides_them():
    """Null is not a policy: the escalation row's declared fact picks one."""
    section = _section()
    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert settings.terminal_policy is None
    assert settings.boundaries is None


@pytest.mark.parametrize("cycles", [True, 0.5, float("nan"), float("inf")])
def test_decision_cycles_refuse_noninteger_values_by_name(cycles):
    section = _section(clock="decisions", decision_cycles=cycles)
    sentence = "windows.decision_cycles must be a nonnegative integer"
    with pytest.raises(ValueError, match=sentence):
        window_settings.WindowSettings.from_yaml(section, CLOCKS)


def test_an_unnamed_window_clock_uses_the_controller_clock():
    section = _section(decision_cycles=3)
    controller_clock = config.Clock(123)
    settings = window_settings.WindowSettings.from_yaml(
        section, CLOCKS, controller_clock
    )
    assert settings.clock is controller_clock
    assert settings.decision_cycles == 3


def test_a_window_clock_must_name_a_declared_domain():
    section = _section(clock="missing")
    with pytest.raises(
        ValueError, match="clock 'missing' is not a clocks entry"
    ):
        window_settings.WindowSettings.from_yaml(section, CLOCKS)


def test_a_charged_window_decision_needs_its_clock():
    with pytest.raises(
        ValueError, match="windows.decision_cycles needs a clock"
    ):
        window_settings.WindowSettings(decision_cycles=1)
