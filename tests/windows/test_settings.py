"""The windows section: the tables it resolves and the two keys it refuses.

STYLE.md rule 7: a pluggable component's section carries one kind key
naming a row of the root's table. This section carries four such keys,
and two of them, terminal_policy and boundaries, arrived in I7 Part 1
with a null default whose meaning is decided later, so the refusal each
one raises at the yaml boundary is what a user meets first.
"""

import pytest

import decsim.records.windows as window_records
import decsim.windows.settings as window_settings


def _section(**overrides) -> dict:
    """A complete windows section, with the given keys overridden."""
    section = {"kind": "sliding", "commit_rounds": None, "buffer_rounds": None}
    section.update(overrides)
    return section


def test_every_windowing_scheme_row_has_one_class():
    """The table is the plug point: one name, one row (STYLE.md rule 7)."""
    rows = window_settings.WINDOWING_SCHEMES

    assert sorted(rows) == ["naive_online", "parallel", "sandwich", "sliding"]
    assert len(set(rows.values())) == len(rows)


def test_a_terminal_policy_that_is_not_one_of_the_two_words_is_refused():
    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(
            _section(terminal_policy="drain")
        )

    sentence = str(refusal.value)
    assert "windows.terminal_policy is one of" in sentence
    assert "'drain'" in sentence


def test_both_terminal_policies_of_the_record_are_accepted():
    for policy in window_records.TERMINAL_POLICIES:
        settings = window_settings.WindowSettings.from_yaml(
            _section(terminal_policy=policy)
        )

        assert settings.terminal_policy == policy


def test_a_boundaries_key_that_names_no_row_is_refused():
    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(_section(boundaries="lazy"))

    sentence = str(refusal.value)
    assert "windows.boundaries" in sentence
    assert "lazy" in sentence


def test_both_boundary_policy_rows_are_reachable_by_name():
    for name in window_settings.BOUNDARY_POLICIES:
        settings = window_settings.WindowSettings.from_yaml(
            _section(boundaries=name)
        )

        assert settings.boundaries == name


def test_both_keys_default_to_null_so_the_plan_decides_them():
    """Null is not a policy: the escalation row's declared fact picks one."""
    settings = window_settings.WindowSettings.from_yaml(_section())

    assert settings.terminal_policy is None
    assert settings.boundaries is None
