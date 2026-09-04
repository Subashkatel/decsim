"""The settings of a round store: its capacity in rounds."""

import dataclasses
from collections.abc import Mapping
from typing import Optional

import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer


@dataclasses.dataclass(frozen=True)
class RoundStoreSettings:
    """The yaml's `round_store` and `strong_round_store` sections.

    Table row (decsim/machine.py): round_store. rounds bounds the store;
    None is unbounded. A full Buffer 0 makes the controller hold the
    finished round and publish it in order once a slot frees, the
    backpressure real systems apply to their source; overflowing
    syndrome buffer 1 is a hard error. The memory model is a Python-only
    observer of retained storage.
    """

    kind: str = "round_store"
    rounds: Optional[int] = None
    memory_model: Optional[syndrome_buffer.MemoryModel] = None

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
