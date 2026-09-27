"""The collection section: how a sweep point's shots are cut and stopped.

sinter's CollectionOptions as yaml (sinter/_data/_collection_options.py:
34-38), at the top of a file or in one sweep block, whose keys override
the top's one by one. None of its keys enters a point's id or a
configuration's id, as sinter's strong id leaves the collection options
out (sinter/_data/_task.py:167-204): collecting longer is the same point
run longer.
"""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.experiments.refusal as refusal

# QEC rounds per piece: a shot's cost grows with its rounds, so a piece
# sized in rounds takes about as long at any history length.
DEFAULT_PIECE_ROUNDS = 20000
KEYS = ("piece_rounds",)


@dataclasses.dataclass(frozen=True)
class CollectionSettings:
    """One point's collection: the rounds its pieces hold."""

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
        piece_rounds = merged.get("piece_rounds", DEFAULT_PIECE_ROUNDS)
        _check_count(piece_rounds, "piece_rounds", where)
        return cls(piece_rounds=piece_rounds)

    def piece_shots(self, rounds_per_shot: int) -> int:
        """The shots one piece holds: its rounds over a shot's, at least one."""
        shots = self.piece_rounds // rounds_per_shot
        return max(shots, 1)


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


def _check_count(value, key: str, where: str) -> None:
    """A count the collection reads is a whole number of at least one."""
    is_whole_number = isinstance(value, int) and not isinstance(value, bool)
    if is_whole_number and value >= 1:
        return
    raise refusal.RefusalError(
        f"{where} collection {key} must be a whole number of at least 1, "
        f"got {value!r}"
    )
