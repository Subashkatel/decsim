"""The windows section: the tables it resolves and the keys it refuses.

STYLE.md rule 7: a pluggable component's section carries one kind key
naming a row of the root's table. This section carries four such keys,
and two of them, terminal_policy and boundaries, carry
a null default whose meaning is decided later, so the refusal each
one raises at the yaml boundary is what a user meets first.
"""

import dataclasses

import pytest

import decsim.config as config
import decsim.records.windows as window_records
import decsim.windows.schemes.sliding as sliding_scheme
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


def test_a_boundary_payload_that_names_no_row_is_refused_at_load():
    section = _section(boundary_payload="bitmap")

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    sentence = str(refusal.value)
    assert "windows.boundary_payload" in sentence
    assert "bitmap" in sentence


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


def test_a_window_size_is_a_whole_count_of_rounds():
    """F >= 1 and B >= 0, refused at load, never as a TypeError at build."""
    refused = (
        ("commit_rounds", 0),
        ("commit_rounds", True),
        ("commit_rounds", "3"),
        ("commit_rounds", 2.5),
        ("buffer_rounds", -1),
        ("buffer_rounds", False),
    )
    for key, rounds in refused:
        section = _section(**{key: rounds})
        with pytest.raises(ValueError, match=f"windows.{key} is a whole"):
            window_settings.WindowSettings.from_yaml(section, CLOCKS)
    section = _section(commit_rounds=1, buffer_rounds=0)
    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert (settings.commit_rounds, settings.buffer_rounds) == (1, 0)


def test_a_charged_window_decision_needs_its_clock():
    with pytest.raises(
        ValueError, match="windows.decision_cycles needs a clock"
    ):
        window_settings.WindowSettings(decision_cycles=1)


class _SteppedScheme(sliding_scheme.SlidingWindowScheme):
    """A sliding scheme with one key of its own, for the section's split."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        stride_rounds: int = 1

        @classmethod
        def from_yaml(cls, section):
            return cls(**section)


def test_a_scheme_rows_own_key_reaches_its_settings(monkeypatch):
    """gem5's shape: the row declares its key, the section hands it over."""
    monkeypatch.setitem(
        window_settings.WINDOWING_SCHEMES, "stepped", _SteppedScheme
    )
    section = _section(kind="stepped", stride_rounds=2)

    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert settings.row_settings == _SteppedScheme.Settings(stride_rounds=2)


def test_a_key_no_row_declares_is_refused_naming_the_sections_keys():
    section = _section(stride_rounds=2)

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert str(refusal.value) == (
        "windows does not know ['stride_rounds']; its keys are ['kind', "
        "'clock', 'decision_cycles', 'commit_rounds', 'buffer_rounds', "
        "'boundary_payload', 'terminal_policy', 'boundaries']"
    )
