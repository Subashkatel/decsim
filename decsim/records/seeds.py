"""The run-level seed graph's records: a path edge, a child, a reservation.

A component's seed comes from the run's root seed and its framed path in
the machine, so the path bytes here are what makes a rerun of the same
machine draw the same numbers (decsim/seeding.py).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

_SEED_PATH_TAG = {"field": b"F", "string_key": b"S"}


@dataclass(frozen=True)
class RunSeedPathSegment:
    """One framed semantic edge in the run-level seed component graph."""

    kind: str
    value: Any

    def canonical_bytes(self) -> bytes:
        """The normative typed and length-framed seed-path bytes.

        An unknown kind has no tag and fails here.
        """
        if self.kind == "none_key":
            return b"N" + (0).to_bytes(4, "big")
        if self.kind == "integer_key":
            text = str(self.value)
            encoded_value = text.encode("ascii")
            value_length = len(encoded_value)
            framed_length = value_length.to_bytes(4, "big")
            return b"I" + framed_length + encoded_value
        encoded_value = self.value.encode()
        tag = _SEED_PATH_TAG[self.kind]
        value_length = len(encoded_value)
        framed_length = value_length.to_bytes(4, "big")
        return tag + framed_length + encoded_value


@dataclass(frozen=True)
class RunSeedChild:
    """One semantic child edge exposed by a seed-graph composite."""

    relative_path: tuple[RunSeedPathSegment, ...]
    child: Any


@dataclass(frozen=True, eq=False)
class RunSeedReservation:
    """A leaf-owned prepared RNG replacement plus manifest seed provenance."""

    proposed_seed_source: str
    proposed_seed: Optional[int]
    prepared_state: Any = field(repr=False)
