"""The windows section: the tables it resolves and the keys it refuses.

STYLE.md rule 7: a pluggable component's section carries one kind key
naming a row of the root's table. This section carries four such keys,
and two of them, terminal_policy and boundaries, carry a null default
the reader resolves from the run's switching slot, so the refusal each
one raises at the yaml boundary is what a user meets first.
"""

import dataclasses
from typing import Optional

import pytest

import decsim.config as config
import decsim.records.windows as window_records
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings

CLOCKS = config.ClockSettings({"decisions": 250.0})
SCHEME_ROWS = window_settings.WINDOWING_SCHEMES.values()


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


@pytest.mark.parametrize("policy", window_records.TERMINAL_POLICIES)
def test_both_terminal_policies_of_the_record_are_accepted(policy):
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


@pytest.mark.parametrize("name", sorted(window_settings.BOUNDARY_POLICIES))
def test_both_boundary_policy_rows_are_reachable_by_name(name):
    section = _section(boundaries=name)
    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)
    row = window_settings.BOUNDARY_POLICIES[name]

    assert isinstance(settings.boundary_policy, row.Settings)


def test_a_run_with_no_switching_reads_null_keys_as_flush_and_eager():
    """Null is not a policy: the run's shape picks one as the yaml is read."""
    section = _section()
    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert settings.terminal_policy == "flush"
    assert settings.boundary_policy == boundary_policies.Eager.Settings()


def test_a_python_record_defaults_to_flush_and_eager():
    settings = window_settings.WindowSettings()

    assert settings.scheme.name == "sliding"
    assert settings.terminal_policy == "flush"
    assert settings.boundary_policy == boundary_policies.Eager.Settings()


def test_a_section_without_its_sizes_is_refused_by_name():
    section = {"kind": "sliding"}

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert str(refusal.value) == (
        "windows needs the keys ['buffer_rounds', 'commit_rounds']; "
        "configs/reference.yaml holds every key with its meaning"
    )


@pytest.mark.parametrize(
    "key, rounds",
    [
        ("commit_rounds", 0),
        ("commit_rounds", True),
        ("commit_rounds", "3"),
        ("commit_rounds", 2.5),
        ("buffer_rounds", -1),
        ("buffer_rounds", False),
    ],
)
def test_a_window_size_that_is_not_a_whole_count_is_refused(key, rounds):
    """F >= 1 and B >= 0, refused at load, never as a TypeError at build."""
    section = _section(**{key: rounds})
    with pytest.raises(ValueError, match=f"windows.{key} is a whole"):
        window_settings.WindowSettings.from_yaml(section, CLOCKS)


@pytest.mark.parametrize("row", SCHEME_ROWS)
def test_every_scheme_record_refuses_a_window_that_commits_nothing(row):
    with pytest.raises(ValueError, match="commit_rounds is a whole number"):
        row.Settings(commit_rounds=0)


def test_the_smallest_whole_window_size_is_accepted():
    section = _section(commit_rounds=1, buffer_rounds=0)
    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    sizes = (settings.scheme.commit_rounds, settings.scheme.buffer_rounds)
    assert sizes == (1, 0)


class _SteppedScheme(sliding_scheme.SlidingWindowScheme):
    """A sliding scheme with one key of its own, for the section's split."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        commit_rounds: Optional[int] = None
        buffer_rounds: Optional[int] = None
        stride_rounds: int = 1
        name = "stepped"


def test_a_scheme_rows_own_key_reaches_its_settings(monkeypatch):
    """gem5's shape: the row declares its key, the section hands it over."""
    monkeypatch.setitem(
        window_settings.WINDOWING_SCHEMES, "stepped", _SteppedScheme
    )
    section = _section(kind="stepped", stride_rounds=2)

    settings = window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert settings.scheme == _SteppedScheme.Settings(stride_rounds=2)


def test_a_key_no_row_declares_is_refused_naming_the_sections_keys():
    section = _section(stride_rounds=2)

    with pytest.raises(ValueError) as refusal:
        window_settings.WindowSettings.from_yaml(section, CLOCKS)

    assert str(refusal.value) == (
        "windows does not know ['stride_rounds']; its keys are ['kind', "
        "'clock', 'decision_cycles', 'commit_rounds', 'buffer_rounds', "
        "'boundary_payload', 'terminal_policy', 'boundaries']"
    )
