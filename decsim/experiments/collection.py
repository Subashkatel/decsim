"""The collection section: how a sweep point's shots are cut and stopped.

sinter's CollectionOptions as yaml (sinter/_data/_collection_options.py:
34-38), at the top of a file or in one sweep block, whose keys override
the top's one by one. None of its keys enters a point's id or a
configuration's id, as sinter's strong id leaves the collection options
out (sinter/_data/_task.py:167-204): collecting longer is the same point
run longer.

A point stops on its contiguous prefix of seeds, sinter's rule
(sinter/_collection/_collection_manager.py:66-67) with a minimum: at the
first shot where its scored failures reach max_failures and its scored
shots reach min_shots, or where its shots reach max_shots or its core
seconds reach max_core_seconds. The target and the minimum count scored
shots, since an unscored shot says nothing about failure; the caps count
every shot, since every shot costs its time, as sinter's shots_left
counts its discards (_collection_manager.py:330).
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.experiments.failure_statistics as failure_statistics
import decsim.experiments.refusal as refusal

# QEC rounds per piece: a shot's cost grows with its rounds, so a piece
# sized in rounds takes about as long at any history length.
DEFAULT_PIECE_ROUNDS = 20000
KEYS = (
    "max_failures",
    "max_shots",
    "max_core_seconds",
    "min_shots",
    "piece_rounds",
)
# The keys that stop a point; sinter refuses a collection without
# max_shots for the same reason (sinter/_collection/_collection_manager.py:
# 230-231): a point with no cap may never stop.
CAP_KEYS = ("max_shots", "max_core_seconds")


@dataclasses.dataclass(frozen=True)
class CollectionSettings:
    """One point's collection: when it stops and the rounds of a piece.

    max_failures is the target, None for none; min_shots the scored
    shots a point runs whatever its failures; max_shots and
    max_core_seconds the caps, at least one of them set.
    """

    max_shots: Optional[int] = None
    max_core_seconds: Optional[float] = None
    max_failures: Optional[int] = None
    min_shots: int = 0
    piece_rounds: int = DEFAULT_PIECE_ROUNDS

    @classmethod
    def from_yaml(
        cls, top: Optional[Mapping], block: Optional[Mapping], where: str
    ) -> "CollectionSettings":
        """The top section's keys, the block's written over them, checked.

        where names the block in a refusal, as the other block checks do.
        """
        merged = {}
        _add_section(merged, top, "the collection section")
        _add_section(merged, block, f"{where} collection")
        _check_caps(merged, where)
        _check_optional_count(merged, "max_shots", where)
        _check_optional_count(merged, "max_failures", where)
        _check_core_seconds(merged, where)
        _check_minimum(merged, where)
        piece_rounds = merged.get("piece_rounds", DEFAULT_PIECE_ROUNDS)
        _check_count(piece_rounds, "piece_rounds", where)
        return cls(**merged)

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

        The rule is checked after every shot and its counts only grow, so
        the first prefix it returns a kind for is the stop. The target
        reached on the minimum's own shot is a minimum stop, and the
        target comes before a cap reached on the same shot
        (failure_statistics.StopKind).
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
    """A point's counts over its contiguous prefix of seeds."""

    shots: int = 0
    scored_shots: int = 0
    failures: int = 0
    core_seconds: float = 0.0

    def add(self, other: "PrefixCounts") -> None:
        """The next seeds' counts, added on."""
        self.shots += other.shots
        self.scored_shots += other.scored_shots
        self.failures += other.failures
        self.core_seconds += other.core_seconds


def _add_section(merged: dict, section: Optional[Mapping], name: str) -> None:
    """One collection mapping's keys into the merged ones, each one known."""
    if section is None:
        return
    keys_text = ", ".join(KEYS)
    if not isinstance(section, Mapping):
        raise refusal.RefusalError(
            f"{name} is {section!r}; it is a mapping of {keys_text}"
        )
    written = set(section)
    unknown = written - set(KEYS)
    if unknown:
        listed = sorted(unknown)
        raise refusal.RefusalError(
            f"{name} names {listed}, which are no collection keys; the "
            f"keys are {keys_text} (configs/reference.yaml)"
        )
    merged.update(section)


def _check_caps(merged: dict, where: str) -> None:
    """A point stops only at a cap it is given, so it needs one."""
    for key in CAP_KEYS:
        if merged.get(key) is not None:
            return
    raise refusal.RefusalError(
        f"{where} has no cap; its collection sets max_shots or "
        "max_core_seconds, or a point may never stop"
    )


def _check_optional_count(merged: dict, key: str, where: str) -> None:
    """A count that may be null is null or a whole number of at least one."""
    value = merged.get(key)
    if value is None:
        return
    _check_count(value, key, where)


def _check_count(value, key: str, where: str) -> None:
    """A count the collection reads is a whole number of at least one."""
    is_whole_number = isinstance(value, int) and not isinstance(value, bool)
    if is_whole_number and value >= 1:
        return
    raise refusal.RefusalError(
        f"{where} collection {key} must be a whole number of at least 1, "
        f"got {value!r}"
    )


def _check_core_seconds(merged: dict, where: str) -> None:
    """The time cap, when given, is a number of seconds above zero."""
    value = merged.get("max_core_seconds")
    if value is None:
        return
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_number and value > 0:
        return
    raise refusal.RefusalError(
        f"{where} collection max_core_seconds must be a number of seconds "
        f"above 0, got {value!r}"
    )


def _check_minimum(merged: dict, where: str) -> None:
    """min_shots is a whole number, zero for no minimum."""
    value = merged.get("min_shots", 0)
    is_whole_number = isinstance(value, int) and not isinstance(value, bool)
    if is_whole_number and value >= 0:
        return
    raise refusal.RefusalError(
        f"{where} collection min_shots must be a whole number of at least "
        f"0, got {value!r}"
    )
