"""A stream's measures over time: one row every so many rounds.

A stream folds its windows into totals (stream_measures.py), and a
total does not say when a waiting line grew or drained. Every
observation.series_every_rounds rounds, as the QPU reads the round out,
this writes one row: the windows the frame committed so far and the
escalated among them, the reaction times of the windows committed since
the row before, and each pool's waiting line at that moment. gem5 dumps
its statistics at a fixed period the same way
(m5.stats.periodicStatDump, src/python/m5/stats/__init__.py). The rows
give a burst's region, the time a line takes to drain after it, and the
share escalated before and after a drift.

A stream's patches read out the same rounds; a round's row is written
once, at the first patch to read it out. The rows span the whole
stream, the warm-up too, so the warm-up's own drain shows.
"""

import dataclasses
from collections.abc import Callable
from typing import Optional

import decsim.config as config_module
import decsim.observe.metrics as metrics


@dataclasses.dataclass(frozen=True)
class SeriesRow:
    """The stream at one round.

    committed_windows and escalated_windows count from the stream's
    start. reaction_mean_us and reaction_max_us are over the windows
    committed since the row before, None when none was. queued_jobs is
    each pool's waiting jobs now, by its word; backlog_rounds the rounds
    read out and not finally corrected as the backlog sampler last saw
    them, at the end of the tick before, None when the run samples no
    backlog (observation.backlog_trace).
    """

    round_index: int
    time_us: float
    committed_windows: int
    escalated_windows: int
    reaction_mean_us: Optional[float]
    reaction_max_us: Optional[float]
    queued_jobs: dict
    backlog_rounds: Optional[int]


class StreamSeries:
    """Writes a row each time the stream reaches a multiple of the period.

    queue_levels is a callable that gives each pool's waiting jobs now,
    by its word; backlog is the run's backlog sampler, or None.
    """

    def __init__(
        self,
        every_rounds: int,
        queue_levels: Callable[[], dict],
        backlog: Optional[metrics.DecodeBacklog],
    ) -> None:
        self.every_rounds = every_rounds
        self.queue_levels = queue_levels
        self.backlog = backlog
        self.rows: list = []
        self.commits = _Commits()
        self.reactions = _Reactions()

    def window_committed(self, reaction_us: float, is_escalated: bool) -> None:
        """One window the frame committed, with its reaction time."""
        self.commits.add(is_escalated)
        self.reactions.add(reaction_us)

    def round_read_out(self, round_index: int, tick: int) -> None:
        """A round left the QPU: a row when it is the period's next."""
        if round_index % self.every_rounds:
            return
        if self.rows and round_index <= self.rows[-1].round_index:
            return
        row = self._row(round_index, tick)
        self.rows.append(row)
        self.reactions = _Reactions()

    def row_dicts(self) -> list:
        """The rows as plain data, in round order."""
        return [dataclasses.asdict(row) for row in self.rows]

    def _row(self, round_index: int, tick: int) -> SeriesRow:
        time_us = config_module.ticks_to_microseconds(tick)
        queued = self.queue_levels()
        backlog_rounds = None
        if self.backlog is not None:
            backlog_rounds = self.backlog.rounds
        reaction_mean_us = self.reactions.mean()
        return SeriesRow(
            round_index=round_index,
            time_us=time_us,
            committed_windows=self.commits.committed,
            escalated_windows=self.commits.escalated,
            reaction_mean_us=reaction_mean_us,
            reaction_max_us=self.reactions.maximum,
            queued_jobs=queued,
            backlog_rounds=backlog_rounds,
        )


class _Commits:
    """The windows the frame committed so far, and the escalated ones."""

    def __init__(self) -> None:
        self.committed = 0
        self.escalated = 0

    def add(self, is_escalated: bool) -> None:
        self.committed += 1
        self.escalated += int(is_escalated)


class _Reactions:
    """The reaction times since the last row: count, sum and maximum."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum: Optional[float] = None

    def add(self, reaction_us: float) -> None:
        self.count += 1
        self.total += reaction_us
        if self.maximum is None or reaction_us > self.maximum:
            self.maximum = reaction_us

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count
