"""One operation's verdicts, when each was published, and their episodes.

A flag runs from its estimated onset to the last round the detector
still fires on. A verdict is read only once it is published, after the
detector's own cost, so a detector's latency delays every answer.
"""

import bisect
import dataclasses
from typing import Optional

import decsim.config as config
import decsim.records.windows as window_records


@dataclasses.dataclass(frozen=True)
class Flag:
    """A firing round's estimated onset, with the region that fired.

    region is the CUSUM's leading chart's region; a count flag has none
    of its own and takes its region when the priors ask.
    """

    onset_round: int
    region: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Episode:
    """One run of firing rounds, from the estimated onset.

    is_open says the newest published round still fires, so the flag
    covers every later round too; region is the flag's region at the
    run's newest firing round.
    """

    first_round: int
    last_firing_round: int
    is_open: bool
    region: Optional[int]

    def meets(self, first_round: int, last_round: int) -> bool:
        """Whether the flag covers any round of [first_round, last_round]."""
        if self.first_round > last_round:
            return False
        return self.is_open or self.last_firing_round >= first_round


class FlagLog:
    """One operation's verdicts, each with the tick it was published at."""

    def __init__(self, first_round: int) -> None:
        self.first_round = first_round
        self.flags: list = []
        self.published_ticks: list = []

    def record(self, flag: Optional[Flag], published_tick: int) -> None:
        """Keep a round's flag, None when it did not fire, and its tick."""
        self.flags.append(flag)
        self.published_ticks.append(published_tick)

    def publication_tick(
        self, now: int, clock: Optional[config.Clock], cycles: int
    ) -> int:
        """One unit serves the rounds in order, each for cycles of its clock.

        A round starts when it is formed or when the round before it is
        published, whichever is later, and is published cycles later,
        charged from the edge at or after that tick (gem5's Clocked,
        decsim/config.py Clock). A unit with no clock or no cycles
        publishes at once.
        """
        newest_tick = self._newest_published_tick()
        start = max(now, newest_tick)
        if clock is None or cycles == 0:
            return start
        return clock.edge(cycles, start)

    def episode_meeting(
        self, window: window_records.Window, now: int
    ) -> Optional[Episode]:
        """The newest published flag that meets the window, or None."""
        published_count = bisect.bisect_right(self.published_ticks, now)
        published = self.flags[:published_count]
        episodes = _episodes(published, self.first_round)
        for episode in reversed(episodes):
            if episode.meets(window.start_round, window.buffer_hi):
                return episode
        return None

    def _newest_published_tick(self) -> int:
        """The tick the previous round was published at; 0 before any."""
        if not self.published_ticks:
            return 0
        return self.published_ticks[-1]


def _episodes(flags: list, first_round: int) -> list:
    """The runs of firing rounds, each from its first round's onset.

    flags[i] is round first_round + i's flag, None when it did not fire.
    """
    runs = []
    for index, flag in enumerate(flags):
        if flag is None:
            continue
        round_index = first_round + index
        _extend_runs(runs, flag, round_index)
    newest_round = first_round + len(flags) - 1
    # a run still firing on the newest round covers every later round
    if runs and runs[-1].last_firing_round == newest_round:
        runs[-1] = dataclasses.replace(runs[-1], is_open=True)
    return runs


def _extend_runs(runs: list, flag: Flag, round_index: int) -> None:
    """A firing round joins the run it follows, or starts a run."""
    if runs and runs[-1].last_firing_round == round_index - 1:
        runs[-1] = dataclasses.replace(
            runs[-1], last_firing_round=round_index, region=flag.region
        )
        return
    run = Episode(flag.onset_round, round_index, False, flag.region)
    runs.append(run)
