"""The settings of a round store: its capacity in rounds."""

import dataclasses
from collections.abc import Mapping
from typing import Optional


@dataclasses.dataclass(frozen=True)
class RoundStoreSettings:
    """The yaml's `round_store` and `strong_round_store` sections.

    Table row (ROUND_STORES, round_store.py): round_store. rounds bounds
    the store; None is unbounded. A full store makes the controller hold
    the finished round and write it in order once a slot frees, the
    backpressure real systems apply to their source (Caune et al.
    2410.05202: the sequencer stalls on the decoder's status register).
    """

    kind: str = "round_store"
    rounds: Optional[int] = None

    @classmethod
    def from_yaml(cls, section: Mapping) -> "RoundStoreSettings":
        """A store section: its kind, and `rounds`, a positive count or null."""
        kind = section.get("kind", "round_store")
        rounds = section.get("rounds")
        if rounds is not None and rounds < 1:
            raise ValueError(
                f"a round store holds at least one round, got {rounds!r}"
            )
        return cls(kind=kind, rounds=rounds)
