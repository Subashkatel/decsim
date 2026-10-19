"""The outer loop: step the escalated share between levels by the backlog.

Slice 06's loop (specs/adaptive-switching/slices/06-adaptive-capacity.md),
throughput first: every check_every_windows windows it reads the most
windows that waited for the weak tier since the last check. Above
high_backlog_windows the chain is falling behind and the share steps one
level down, fewer windows sent to the strong tier; at or under
low_backlog_windows for calm_checks checks in a row it steps one level
up; between the two it holds. The levels are a table of shares, as the
calibration table holds a threshold per share (experiments/basic_case/
calibration.py); what a share costs in accuracy is read off that table,
not decided here. Lowering the share cannot undo a window already sent.
"""

import dataclasses

import decsim.config as config


class TargetSteps:
    """The share the inner loop holds, stepped by the backlog it reads."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The levels, the start, the cadence and the two watermarks."""

        start: float
        levels: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
        check_every_windows: int = 100
        high_backlog_windows: int = 4
        low_backlog_windows: int = 1
        calm_checks: int = 5

        def __post_init__(self) -> None:
            _check_levels(self.levels, self.start)
            _check_counts(self)

        def build(self) -> "TargetSteps":
            """A loop at its start level."""
            return TargetSteps(self)

    def __init__(self, settings: "TargetSteps.Settings") -> None:
        self.settings = settings
        self.level_index = settings.levels.index(settings.start)
        self.window_count = 0
        self.peak_waiting = 0
        self.calm_count = 0
        # (window count, share, most windows waiting) at each check
        self.trajectory: list = []

    @property
    def target(self) -> float:
        """The share the inner loop holds now."""
        return self.settings.levels[self.level_index]

    def observe(self, waiting_windows: int) -> None:
        """One window decided, waiting_windows behind it; check on cadence."""
        self.window_count += 1
        self.peak_waiting = max(self.peak_waiting, waiting_windows)
        if self.window_count % self.settings.check_every_windows:
            return
        self._check(self.peak_waiting)
        self.trajectory.append(
            (self.window_count, self.target, self.peak_waiting)
        )
        self.peak_waiting = 0

    def _check(self, peak: int) -> None:
        """Down above the high mark, up after calm checks, else hold."""
        settings = self.settings
        if peak > settings.high_backlog_windows:
            self.calm_count = 0
            lower = self.level_index - 1
            self.level_index = max(lower, 0)
            return
        if peak > settings.low_backlog_windows:
            self.calm_count = 0
            return
        self.calm_count += 1
        if self.calm_count < settings.calm_checks:
            return
        self.calm_count = 0
        top = len(settings.levels) - 1
        higher = self.level_index + 1
        self.level_index = min(higher, top)


def _check_levels(levels: tuple, start: float) -> None:
    """The levels rise within [0, 1], and the start is one of them."""
    ordered = sorted(levels)
    in_range = all(0.0 <= level <= 1.0 for level in levels)
    if list(levels) != ordered or not in_range:
        raise ValueError(
            f"target_steps.levels must rise within [0, 1], got {levels}"
        )
    if start not in levels:
        raise ValueError(
            f"target_steps.start {start} is not one of the levels {levels}"
        )


def _check_counts(settings: TargetSteps.Settings) -> None:
    """Whole counts, and the high watermark above the low one."""
    counts = (
        ("check_every_windows", settings.check_every_windows, "windows", 1),
        ("calm_checks", settings.calm_checks, "checks", 1),
        ("low_backlog_windows", settings.low_backlog_windows, "windows", 0),
    )
    for name, value, unit, least in counts:
        path = f"target_steps.{name}"
        config.check_whole_count(path, value, unit, least)
    high = settings.high_backlog_windows
    low = settings.low_backlog_windows
    if high <= low:
        raise ValueError(
            "target_steps.high_backlog_windows must be above "
            f"low_backlog_windows, got {high} and {low}"
        )
