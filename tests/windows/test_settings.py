"""The windows record: its defaults, the switching preset, its sizes.

A run with no switching drains with a flush tail and ships boundaries
eagerly; switching_windows gives a switching run the lookahead tail
and the boundary row its strong window declares.
"""

import dataclasses

import pytest

import decsim.escalation.strong_window_shapes as strong_window_shapes
import decsim.windows.boundary_policies as boundary_policies
import decsim.windows.schemes.naive_online as naive_online_scheme
import decsim.windows.schemes.parallel as parallel_scheme
import decsim.windows.schemes.sandwich as sandwich_scheme
import decsim.windows.schemes.sliding as sliding_scheme
import decsim.windows.settings as window_settings

# every windowing scheme decsim ships
SCHEME_ROWS = (
    sliding_scheme.SlidingWindowScheme,
    parallel_scheme.ParallelWindowScheme,
    sandwich_scheme.TanSandwichScheme,
    naive_online_scheme.NaiveOnlineScheme,
)
# each strong window record and the boundary row it declares
PRESET_ROWS = [
    (strong_window_shapes.RedoWindow.Settings(), boundary_policies.Held),
    (strong_window_shapes.DoubleWindow.Settings(), boundary_policies.Eager),
]


def test_a_python_record_defaults_to_flush_and_eager():
    settings = window_settings.WindowSettings()

    assert settings.scheme.name == "sliding"
    assert settings.terminal_policy == "flush"
    assert settings.boundary_policy == boundary_policies.Eager.Settings()


def test_a_terminal_policy_off_its_two_words_is_refused():
    """A misspelt word lays out a last window that is neither policy's."""
    with pytest.raises(ValueError, match="terminal_policy is one of"):
        window_settings.WindowSettings(terminal_policy="lookahaed")


@pytest.mark.parametrize("strong_window, boundary_row", PRESET_ROWS)
def test_the_switching_preset_takes_lookahead_and_the_strong_windows_row(
    strong_window, boundary_row
):
    plain_windows = window_settings.WindowSettings()

    windows = window_settings.switching_windows(plain_windows, strong_window)

    assert windows.terminal_policy == "lookahead"
    assert windows.boundary_policy == boundary_row.Settings()
    assert windows.scheme == plain_windows.scheme


class _OutsideHeld:
    """A boundary row written outside decsim, in no table."""

    ships_provisional_boundaries = False

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The outside row's record."""

        def build(self) -> "_OutsideHeld":
            return _OutsideHeld()

    def on_commit(self, window, *, final: bool) -> bool:
        """Ship when final."""
        del window
        return final


@dataclasses.dataclass(frozen=True)
class _OutsideStrongWindow:
    """A strong window record from outside, giving its own boundary row."""

    boundary_policy = _OutsideHeld.Settings()


def test_the_switching_preset_takes_the_record_its_strong_window_gives():
    """The record arrives whole: no table is read and no word is named."""
    plain_windows = window_settings.WindowSettings()
    strong_window = _OutsideStrongWindow()

    windows = window_settings.switching_windows(plain_windows, strong_window)

    assert windows.boundary_policy is strong_window.boundary_policy
    assert windows.terminal_policy == "lookahead"


@pytest.mark.parametrize("row", SCHEME_ROWS)
def test_every_scheme_record_refuses_a_window_that_commits_nothing(row):
    with pytest.raises(ValueError, match="commit_rounds is a whole number"):
        row.Settings(commit_rounds=0)
