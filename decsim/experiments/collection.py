"""The collection: how a task's shots are cut and stopped.

sinter's CollectionOptions (sinter/_data/_collection_options.py:34-38):
an experiment's default and a task's own. None of its keys enters a
task's id, as sinter's strong id leaves the collection options out
(sinter/_data/_task.py:167-204): collecting longer is the same task
run longer.

A task stops on its contiguous prefix of seeds, sinter's rule
(sinter/_collection/_collection_manager.py:66-67) with a minimum: at the
first shot where its scored failures reach max_failures and its scored
shots reach min_shots, or where its shots reach max_shots or its core
seconds reach max_core_seconds. The target and the minimum count scored
shots, since an unscored shot says nothing about failure; the caps count
every shot, since every shot costs its time, as sinter's shots_left
counts its discards (_collection_manager.py:330).
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Optional

import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.fold as fold

# QEC rounds per piece, every patch's rounds added up: a shot's cost
# grows with its rounds, so a piece sized in rounds takes about as long
# at any history length.
DEFAULT_PIECE_ROUNDS = 20000
# the state of a prefix its time cap stopped, whose limits assume a
# shot's time is independent of its failure
TIME_CAP_STATE = "time cap"


@dataclasses.dataclass(frozen=True)
class CollectionSettings:
    """The settings of one task's collection.

    max_failures is the target, None for none; min_shots the scored shots
    run whatever the failures; max_shots and max_core_seconds the caps, at
    least one set; piece_rounds the rounds of one piece. Python input
    enters here, so it checks its own values.

    core_seconds_per_shot is an estimate of one shot's core seconds, read
    off an earlier run's sim_wall_seconds; `decsim run --slurm` packs a
    fixed-shot task's pieces into jobs by it, and nothing else reads it.
    It sits here because it is about running the task, not about the
    machine: like the caps it is no part of the task's id, and the
    folder records it with the collection. sinter sizes a batch from a
    shot's time the same way (sinter/_data/_collection_options.py:27-31,
    max_batch_seconds), from shots it measured.
    """

    max_shots: Optional[int] = None
    max_core_seconds: Optional[float] = None
    max_failures: Optional[int] = None
    min_shots: int = 0
    piece_rounds: int = DEFAULT_PIECE_ROUNDS
    core_seconds_per_shot: Optional[float] = None

    def __post_init__(self) -> None:
        _check_caps(self)
        _check_optional_count(self.max_shots, "max_shots")
        _check_optional_count(self.max_failures, "max_failures")
        _check_core_seconds(self.max_core_seconds, "max_core_seconds")
        _check_core_seconds(self.core_seconds_per_shot, "core_seconds_per_shot")
        _check_minimum(self.min_shots)
        _check_count(self.piece_rounds, "piece_rounds")

    def text(self) -> str:
        """Every key the collection sets and its value, as one phrase."""
        pairs = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            pairs.append(f"{field.name} {value}")
        return ", ".join(pairs)

    def piece_shots(self, rounds_per_shot: int) -> int:
        """The shots one piece holds: its rounds over a shot's, at least one."""
        shots = self.piece_rounds // rounds_per_shot
        return max(shots, 1)

    def stop_kind(
        self, counts: "PrefixCounts"
    ) -> Optional[failure_statistics.StopKind]:
        """Why a contiguous prefix with these counts has stopped, or None.

        The counts only grow, so the first prefix given a kind is the stop. The
        target reached on the minimum's shot is a minimum stop, and the target
        comes before a cap on the same shot.
        """
        if self._has_reached_the_target(counts):
            return self._target_or_minimum(counts)
        if self.max_shots is not None and counts.shots >= self.max_shots:
            return failure_statistics.StopKind.CAP
        if self.max_core_seconds is None:
            return None
        if counts.core_seconds >= self.max_core_seconds:
            return failure_statistics.StopKind.CAP
        return None

    def is_time_cap(self, counts: "PrefixCounts") -> bool:
        """Whether a cap stop at these counts was the time cap alone.

        A shot cap reached on the same shot fixed the count in advance,
        which needs no assumption about time, so it names the stop.
        """
        if self.max_core_seconds is None:
            return False
        if self.max_shots is not None and counts.shots >= self.max_shots:
            return False
        return counts.core_seconds >= self.max_core_seconds

    def _has_reached_the_target(self, counts: "PrefixCounts") -> bool:
        """Whether the scored failures and scored shots are both far enough."""
        if self.max_failures is None:
            return False
        if counts.failures < self.max_failures:
            return False
        return counts.scored_shots >= self.min_shots

    def _target_or_minimum(
        self, counts: "PrefixCounts"
    ) -> failure_statistics.StopKind:
        """The target past the minimum, or the minimum that held it back."""
        if counts.scored_shots == self.min_shots:
            return failure_statistics.StopKind.MINIMUM
        return failure_statistics.StopKind.TARGET


@dataclasses.dataclass
class PrefixCounts:
    """A task's counts over its contiguous prefix of seeds."""

    shots: int = 0
    scored_shots: int = 0
    failures: int = 0
    core_seconds: float = 0.0

    def add(self, other: "PrefixCounts") -> None:
        """Counts of the seeds right after these: a prefix grows at its end."""
        self.shots += other.shots
        self.scored_shots += other.scored_shots
        self.failures += other.failures
        self.core_seconds += other.core_seconds


@dataclasses.dataclass(frozen=True)
class TaskRule:
    """What a task's summary reads its prefix by.

    settings is its collection, None for shots a caller fixed in advance,
    whose run is a cap at the shots it ran. is_adaptive says its
    threshold learns online, so its shots are not independent draws
    and have no interval here.
    """

    settings: Optional[CollectionSettings] = None
    is_adaptive: bool = False

    @classmethod
    def from_record(cls, record: Mapping) -> "TaskRule":
        """The rule a task's machine.json says its prefix is read by.

        The record's experiment facts are what collect_command wrote when
        it recorded the task, so a fold reads a task by the collection
        it was run under, whatever its run file says now.
        """
        facts = record["experiment"]
        settings = CollectionSettings(**facts["collection"])
        return cls(settings, facts["adaptive"])


class PrefixTracker:
    """One task's contiguous prefix of seeds as its shot rows arrive.

    The rows come in seed order, as a fold merges them. The prefix ends
    at the shot its rule stops on, or at the first missing seed, which
    holds the stop until the gap is filled; later rows count nowhere.
    """

    def __init__(self, rule: TaskRule) -> None:
        self.rule = rule
        self.counts = PrefixCounts()
        self.stop_kind = None
        self.is_open = True
        # each distinct (outputs, rounds per output) the prefix's shots ran
        self.round_shapes = set()

    def add(self, row: Mapping) -> None:
        """One shot row: onto the prefix while the prefix is open."""
        if not self.is_open:
            return
        seed = fold.number_of(row["seed"])
        if seed != self.counts.shots:
            self.is_open = False
            return
        shot_counts = _shot_counts_of(row)
        self.counts.add(shot_counts)
        round_shape = _round_shape_of(row)
        self.round_shapes.add(round_shape)
        if self.rule.settings is None:
            return
        self.stop_kind = self.rule.settings.stop_kind(self.counts)
        if self.stop_kind is not None:
            self.is_open = False

    def state(self) -> str:
        """The row's state: the stop's kind, or why there is none.

        A time cap has a cap's limits, exact only when a shot's time
        does not depend on whether it failed (design section 7), so its
        state says it was the time cap.
        """
        if self.counts.shots == 0:
            return "no data"
        if self.rule.is_adaptive:
            return "adaptive"
        if self._is_a_time_cap_stop():
            return TIME_CAP_STATE
        if self.stop_kind is not None:
            return self.stop_kind.value
        if self.rule.settings is None:
            return failure_statistics.StopKind.CAP.value
        return "running"

    def _is_a_time_cap_stop(self) -> bool:
        if self.stop_kind is not failure_statistics.StopKind.CAP:
            return False
        return self.rule.settings.is_time_cap(self.counts)

    def round_shape(self) -> Optional[tuple]:
        """The (outputs, rounds per output) every prefix shot ran, or None.

        A live stream's shots can run different rounds, and outputs that ran
        apart have no one length, so no one shape converts such a prefix.
        """
        if len(self.round_shapes) != 1:
            return None
        (round_shape,) = self.round_shapes
        _outputs, rounds = round_shape
        if rounds == 0:
            return None
        return round_shape

    def stop_kind_for_limits(self) -> failure_statistics.StopKind:
        """The kind whose limits hold for the prefix as it stands.

        A prefix its rule has not stopped, a gap or a stopped collect,
        has a shot count fixed by where it ended, so its limits are the
        cap's, and its state says it is incomplete.
        """
        if self.stop_kind is None:
            return failure_statistics.StopKind.CAP
        return self.stop_kind


def _shot_counts_of(row: Mapping) -> PrefixCounts:
    """One shot row's counts; a csv row's text is read as its number."""
    is_scored = fold.number_of(row["is_scored"])
    failed = fold.number_of(row["logical_failure"])
    core_seconds = fold.number_of(row["sim_wall_seconds"])
    scored_shots = int(bool(is_scored))
    failures = int(bool(failed))
    return PrefixCounts(1, scored_shots, failures, core_seconds)


def _round_shape_of(row: Mapping) -> tuple:
    """One shot row's outputs and rounds per output, as numbers."""
    outputs = fold.number_of(row["scored_outputs"])
    rounds = fold.number_of(row["rounds_per_output"])
    return (outputs, rounds)


def _check_caps(settings: CollectionSettings) -> None:
    """A task stops only at a cap it is given, so it needs one."""
    if settings.max_shots is not None:
        return
    if settings.max_core_seconds is not None:
        return
    raise ValueError(
        "collection has no cap; it sets max_shots or max_core_seconds, or a "
        "task may never stop"
    )


def _check_optional_count(value, key: str) -> None:
    """A count that may be None is None or a whole number of at least one."""
    if value is None:
        return
    _check_count(value, key)


def _check_count(value, key: str) -> None:
    """A count the collection reads is a whole number of at least one."""
    is_whole_number = isinstance(value, int) and not isinstance(value, bool)
    if is_whole_number and value >= 1:
        return
    raise ValueError(
        f"collection {key} must be a whole number of at least 1, got {value!r}"
    )


def _check_core_seconds(value, key: str) -> None:
    """Core seconds, when given, are a finite number above zero.

    An infinite time cap is no cap: a task with no other would never
    stop; an infinite shot estimate packs no job.
    """
    if value is None:
        return
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    is_finite_number = is_number and math.isfinite(value)
    if is_finite_number and value > 0:
        return
    raise ValueError(
        f"collection {key} must be a finite number of seconds above 0, got "
        f"{value!r}"
    )


def _check_minimum(value) -> None:
    """min_shots is a whole number, zero for no minimum."""
    is_whole_number = isinstance(value, int) and not isinstance(value, bool)
    if is_whole_number and value >= 0:
        return
    raise ValueError(
        f"collection min_shots must be a whole number of at least 0, got "
        f"{value!r}"
    )
