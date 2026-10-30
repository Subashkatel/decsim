"""The outer loop: step the escalated share down while the backlog grows.

Throughput comes first. A chain that keeps up holds a backlog that comes and
goes: one escalated window holds the next windows for its service, so
bursts are the queue's ordinary state. A chain that falls behind holds
one that grows. So every check_every_windows windows the loop reads the
mean windows waiting for its watched pool since the last check, and:
  - steps one level down, fewer windows to the strong tier, when that
    mean rose at each of the last rise_checks checks and by more than
    rise_margin_windows over them;
  - steps one level up when calm_checks checks have passed with no such
    rise and the mean is no higher than it was calm_checks checks ago,
    which it cannot be before that many checks;
  - holds otherwise.
The watched pool is the weak tier by default: with the sliding window a
strong decode holds the next windows, so the weak tier's line grows. With
the double window the next windows do not wait, so the weak tier's line
stays empty and the strong tier's grows; such a chain watches "target".
The levels are a table of shares, as the calibration table holds a
threshold per share (experiments/five_results/calibration.py); what a
share costs in accuracy is read off that table, not decided here.
Lowering the share cannot undo a window already sent.
"""

import collections
import dataclasses

import decsim.config as config


class TargetSteps:
    """The share the inner loop holds, stepped by the backlog's trend."""

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The levels, the start, the cadence, the trend's tests and the
        pool whose line the loop watches: "source" (the weak tier) or
        "target" (the strong tier)."""

        start: float
        levels: tuple = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
        check_every_windows: int = 100
        rise_checks: int = 2
        rise_margin_windows: float = 5.0
        calm_checks: int = 5
        watched_pool: str = "source"

        def __post_init__(self) -> None:
            _check_levels(self.levels, self.start)
            _check_counts(self)
            if self.watched_pool not in ("source", "target"):
                raise ValueError(
                    "target_steps.watched_pool must be 'source' or 'target', "
                    f"got {self.watched_pool!r}"
                )

        def build(self) -> "TargetSteps":
            """A loop at its start level."""
            return TargetSteps(self)

    def __init__(self, settings: "TargetSteps.Settings") -> None:
        self.settings = settings
        self.level_index = settings.levels.index(settings.start)
        # the windows waiting behind each window decided since the last check
        self.waiting_since_check: list = []
        self.calm_count = 0
        reach = max(settings.rise_checks, settings.calm_checks) + 1
        # the last checks' mean windows waiting, the newest last
        self.means: collections.deque = collections.deque(maxlen=reach)
        # (window count, share, mean windows waiting) at each check
        self.trajectory: list = []

    @property
    def target(self) -> float:
        """The share the inner loop holds now."""
        return self.settings.levels[self.level_index]

    def observe(self, waiting_windows: int) -> None:
        """One window decided, waiting_windows behind it; check on cadence."""
        self.waiting_since_check.append(waiting_windows)
        every = self.settings.check_every_windows
        if len(self.waiting_since_check) < every:
            return
        mean = sum(self.waiting_since_check) / every
        self.waiting_since_check.clear()
        self.means.append(mean)
        self._check()
        check_count = len(self.trajectory) + 1
        window_count = check_count * every
        self.trajectory.append((window_count, self.target, mean))

    def _check(self) -> None:
        """Down on a sustained rise, up after calm checks, else hold."""
        if self._is_rising():
            self.calm_count = 0
            lower = self.level_index - 1
            self.level_index = max(lower, 0)
            return
        self.calm_count += 1
        if self.calm_count < self.settings.calm_checks:
            return
        self.calm_count = 0
        if not self._is_no_higher_than(self.settings.calm_checks):
            return
        top = len(self.settings.levels) - 1
        higher = self.level_index + 1
        self.level_index = min(higher, top)

    def _is_rising(self) -> bool:
        """The mean rose at each of the last checks, and by the margin."""
        checks = self.settings.rise_checks
        if len(self.means) <= checks:
            return False
        means = list(self.means)
        recent = means[-checks - 1 :]
        steps = zip(recent, recent[1:], strict=False)
        each_rose = all(later > earlier for earlier, later in steps)
        total = recent[-1] - recent[0]
        return each_rose and total > self.settings.rise_margin_windows

    def _is_no_higher_than(self, checks: int) -> bool:
        """The mean is no higher than checks ago; False before that check."""
        if len(self.means) <= checks:
            return False
        earlier_index = -checks - 1
        return self.means[-1] <= self.means[earlier_index]


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
    """Whole counts, and a margin that is a number of at least 0."""
    counts = (
        ("check_every_windows", settings.check_every_windows, "windows"),
        ("rise_checks", settings.rise_checks, "checks"),
        ("calm_checks", settings.calm_checks, "checks"),
    )
    for name, value, unit in counts:
        path = f"target_steps.{name}"
        config.check_whole_count(path, value, unit)
    margin = settings.rise_margin_windows
    if not config.is_number(margin) or margin < 0:
        raise ValueError(
            "target_steps.rise_margin_windows must be a number of at least "
            f"0, got {margin!r}"
        )
