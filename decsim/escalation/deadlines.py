"""An edge's deadline: a source job late past it is rescued by the target.

DART-Q (Bhave, Choudhury and Basu 2605.09142, Sec. III) gives each
decode job a deadline d = a + Delta, its arrival plus a slack budget,
and a job that completes after d is a miss: "a late decode output is
treated as unavailable at the required control boundary" (Sec. III-C).
A rescue is extra work a policy condition injects: "a slack threshold,
a queue-occupancy threshold, or a bounded budget on additional service
time" (Sec. IV-C). Its evaluation (Sec. VII, Table I) shows rescue helps
only while its trigger stays selective.

decsim reads this on an escalate edge. The source job's arrival is its
admission to the source pool, its completion is its result reaching
the verdict, and the rescue is the edge's target decode started before
the verdict, the speculative path (Toshio et al. 2510.25222 Step 1). A
late result is unavailable by default, so the verdict escalates and the
target answers; late_source_result "revision" lets it meet the verdict.

    slack    rescue when the slack left reaches threshold_microseconds
    backlog  rescue at arrival when the source queue holds queued_jobs

DART-Q's third trigger, cutoff-hit, is a decoder's iteration cap with
the status verdict (verdicts.py). Not built: EDF scheduling, a cutoff
in microseconds, and a rescue into the source pool's own queue.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Optional, Protocol

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.log_sources as log_sources
import decsim.statistics.kinds as statistic_kinds
import decsim.statistics.registry as statistics_registry

# what a source result decided after its deadline is: DART-Q's
# "unavailable", or a revision that meets the verdict as on time
LATE_SOURCE_RESULTS = ("unavailable", "revision")


class TriggerSettings(Protocol):
    """A rescue trigger row's settings record (this module)."""

    name: str

    def rescue_delay(self, slack_ticks: int, queued_jobs: int) -> Optional[int]:
        """The ticks from arrival to the rescue; None for no rescue."""


class SlackTrigger:
    """Rescue when the slack left before the deadline reaches a threshold.

    DART-Q names a slack threshold (Sec. IV-C) and runs 5 us (Sec.
    VI-C 2). It does not say when slack is read; this reads it at every
    instant, so the rescue starts at d - threshold_microseconds, or at
    arrival when the threshold is the whole budget or more.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The slack row: the slack left at which the rescue starts."""

        threshold_microseconds: float = 0.0
        name = "slack"

        def __post_init__(self) -> None:
            config.check_duration(
                "slack.threshold_microseconds", self.threshold_microseconds
            )

        def rescue_delay(
            self, slack_ticks: int, queued_jobs: int
        ) -> Optional[int]:
            """The ticks until the slack left reaches the threshold."""
            del queued_jobs
            threshold_ticks = config.microseconds_to_ticks(
                self.threshold_microseconds
            )
            ticks_to_threshold = slack_ticks - threshold_ticks
            return max(0, ticks_to_threshold)


class BacklogTrigger:
    """Rescue at arrival when the source queue holds queued_jobs or more.

    DART-Q's queue-occupancy threshold "is measured in jobs" (Sec.
    IV-C), 2 jobs in its runs (Sec. VI-C 2), and was its most selective
    trigger (Table I). The queue is read as the job arrives, before it
    joins, as the paper's admission rule reads Q(a-) (Sec. IV-C).
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """The backlog row: the waiting jobs at which the rescue starts."""

        queued_jobs: int = 1
        name = "backlog"

        def __post_init__(self) -> None:
            if self.queued_jobs < 1:
                raise ValueError(
                    "backlog.queued_jobs must be a positive count, got "
                    f"{self.queued_jobs!r}"
                )

        def rescue_delay(
            self, slack_ticks: int, queued_jobs: int
        ) -> Optional[int]:
            """At arrival when the queue is long enough; else none."""
            del slack_ticks
            if queued_jobs < self.queued_jobs:
                return None
            return 0


class Deadline:
    """One edge's deadlines: armed at each source arrival, met or missed.

    source_queue is the source pool's manager, whose waiting jobs the
    backlog trigger reads. Statistics: rescues, the target decodes a
    trigger started (DART-Q's trigger rate), and misses, the source
    results that reached the verdict after their deadline (its MissRate
    numerator), raw counts.
    """

    @dataclasses.dataclass(frozen=True)
    class Settings:
        """An edge's deadline: the slack budget, the trigger, a late result.

        slack_microseconds is DART-Q's Delta, which it requires positive
        (Sec. III-A, Eq. 1). late_source_result names one of
        LATE_SOURCE_RESULTS.
        """

        slack_microseconds: float
        trigger: TriggerSettings = SlackTrigger.Settings()
        late_source_result: str = "unavailable"
        name = "deadline"

        def __post_init__(self) -> None:
            if not self.slack_microseconds > 0:
                raise ValueError(
                    "deadline.slack_microseconds must be positive, got "
                    f"{self.slack_microseconds!r}"
                )
            config.check_duration(
                "deadline.slack_microseconds", self.slack_microseconds
            )
            if self.late_source_result not in LATE_SOURCE_RESULTS:
                raise ValueError(
                    f"deadline.late_source_result must be one of "
                    f"{LATE_SOURCE_RESULTS}, got "
                    f"{self.late_source_result!r}"
                )

        def build(self, engine: engine_module.Engine) -> "Deadline":
            """A fresh deadline on the run's engine."""
            return Deadline(engine, self)

    source_queue = ports.Port(ports.DecodeQueue)

    STATISTICS = (
        statistics_registry.Statistic(
            "rescues",
            statistic_kinds.Count,
            "target decodes a deadline trigger started",
            "windows",
        ),
        statistics_registry.Statistic(
            "misses",
            statistic_kinds.Count,
            "source results that reached the verdict after the deadline",
            "windows",
        ),
    )
    statistics = statistics_registry.unrecorded(STATISTICS)

    def __init__(
        self, engine: engine_module.Engine, settings: "Deadline.Settings"
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.slack_ticks = config.microseconds_to_ticks(
            settings.slack_microseconds
        )
        self.deadline_tick_by_window: dict = {}
        self.rescue_by_window: dict = {}
        self.unavailable_windows: set = set()

    def arm(
        self,
        source_job: decoding_records.DecodeJob,
        rescue: Callable[[], None],
    ) -> None:
        """The source job arrives: its deadline, and its rescue if due."""
        window_key = _window_key_of(source_job)
        self.disarm(window_key)
        deadline_tick = self.engine.now + self.slack_ticks
        self.deadline_tick_by_window[window_key] = deadline_tick
        queued_jobs = self.source_queue.waiting_jobs()
        trigger = self.settings.trigger
        delay = trigger.rescue_delay(self.slack_ticks, queued_jobs)
        if delay is None:
            return
        fire = functools.partial(self._rescue, window_key, rescue)
        event = self.engine.schedule(delay, fire, label="deadline rescue")
        self.rescue_by_window[window_key] = event

    def note_result(self, source_job: decoding_records.DecodeJob) -> None:
        """The source result reached the verdict: no rescue; late or not.

        DART-Q's miss is a completion after the deadline, L = c - d > 0
        (Sec. III-A, Eq. 2).
        """
        window_key = _window_key_of(source_job)
        deadline_tick = self.deadline_tick_by_window.pop(window_key, None)
        self._cancel_rescue(window_key)
        if deadline_tick is None or self.engine.now <= deadline_tick:
            return
        self.statistics.misses.add()
        late_ticks = self.engine.now - deadline_tick
        late_by = config.format_ticks(late_ticks)
        self.engine.log(
            log_sources.DECODER_MANAGER,
            f"{source_job.label}: deadline missed by {late_by}",
        )
        if self.settings.late_source_result == "unavailable":
            self.unavailable_windows.add(window_key)

    def is_unavailable(self, source_job: decoding_records.DecodeJob) -> bool:
        """Whether the source result came late and counts as no answer."""
        window_key = _window_key_of(source_job)
        if window_key not in self.unavailable_windows:
            return False
        self.unavailable_windows.discard(window_key)
        return True

    def disarm(self, window_key: tuple) -> None:
        """The window's request ended: its deadline and rescue go."""
        self.deadline_tick_by_window.pop(window_key, None)
        self.unavailable_windows.discard(window_key)
        self._cancel_rescue(window_key)

    def _rescue(self, window_key: tuple, rescue: Callable[[], None]) -> None:
        del self.rescue_by_window[window_key]
        self.statistics.rescues.add()
        rescue()

    def _cancel_rescue(self, window_key: tuple) -> None:
        event = self.rescue_by_window.pop(window_key, None)
        if event is not None:
            self.engine.deschedule(event)


def _window_key_of(job: decoding_records.DecodeJob) -> tuple:
    request_key = job.request_key
    return (request_key.operation_id, request_key.window_id)
