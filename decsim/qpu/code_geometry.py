"""The code cards: the numbers the simulator needs from a QEC code.

A card is a small frozen record, not a stabilizer code. The simulator
prices decoder timing, so all it takes from a code is its distance, its
window sizes, the size of the decoding graph per round, and the syndrome
bits per round. The numbers can be set by hand or copied from an upstream
tool's output; decsim never imports such a tool.

The rotated surface-code card follows Stim's generated
``surface_code:rotated_memory_z`` circuit (Stim, src/stim/gen/
gen_surface_code.cc): a distance-d patch has d*d data qubits and d*d - 1
measure qubits, and every round reads out every measure qubit once. The
bivariate-bicycle card follows Bravyi et al., High-threshold and
low-overhead fault-tolerant quantum memory (Nature 627, 778, 2024; arXiv
2308.07915): a [[n, k, d]] code whose round measures n/2 X checks and n/2
Z checks, the [[144, 12, 12]] gross code by default.
"""

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class SurfaceCodeModel:
    """Timing and sizing card of one rotated surface-code patch.

    A buffer below the (d, d) floor degrades windowed accuracy (Skoric et
    al. 2209.08552; Tan et al., PRX Quantum 4, 040344). Running there on
    purpose, as a sweep may, needs a written reason in
    window_floor_justification, the way a free frame write does.
    """

    distance: int = 3
    # None: the run's cadence.
    round_microseconds: Optional[float] = None
    # None: the distance.
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None
    window_floor_justification: Optional[str] = None

    def __post_init__(self) -> None:
        round_microseconds = _optional_float(self.round_microseconds)
        object.__setattr__(self, "round_microseconds", round_microseconds)
        _check_justification(self.window_floor_justification)

    @property
    def name(self) -> str:
        """The routing and readout identity of this card."""
        return f"rotated surface code (d={self.distance})"

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle: the distance."""
        return self.distance

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""
        return self.round_microseconds

    def commit_rounds(self) -> int:
        """Rounds committed per decode window; the distance by default."""
        if self.commit_rounds_override is not None:
            return self.commit_rounds_override
        return self.distance

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window; the distance by default."""
        if self.buffer_rounds_override is not None:
            return self.buffer_rounds_override
        return self.distance

    def buffering_floor(self) -> tuple[int, int]:
        """The smallest leading and trailing buffers: (d, d)."""
        return (self.distance, self.distance)

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph nodes per round: d*d per patch, plus a d-node seam.

        The seam strip is a heuristic for a multi-patch operation.
        """
        nodes_per_patch = self.distance * self.distance
        seam_nodes = 0
        if num_patches > 1:
            seam_nodes = self.distance
        return num_patches * nodes_per_patch + seam_nodes

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Bits read out per round: the d*d - 1 stabilizers of every patch."""
        stabilizer_count = self.distance * self.distance - 1
        return num_patches * stabilizer_count


@dataclasses.dataclass(frozen=True)
class BBCodeModel:
    """Timing and sizing card of one bivariate-bicycle CSS code.

    One modeled round is one complete extraction cycle: n/2 X checks and
    n/2 Z checks. The detector error model owns the exact window-local
    detector rows.
    """

    qubit_count: int = 144
    logical_qubit_count: int = 12
    distance: int = 12
    # None: the run's cadence.
    round_microseconds: Optional[float] = None
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None
    window_floor_justification: Optional[str] = None

    def __post_init__(self) -> None:
        _check_justification(self.window_floor_justification)
        _require_positive_int(self.qubit_count, "qubit_count")
        _require_positive_int(self.logical_qubit_count, "logical_qubit_count")
        _require_positive_int(self.distance, "distance")
        if self.qubit_count % 2:
            raise ValueError(
                f"qubit_count must be even; got {self.qubit_count!r}"
            )
        if self.logical_qubit_count > self.qubit_count:
            raise ValueError(
                "logical_qubit_count must not exceed qubit_count; got "
                f"logical_qubit_count={self.logical_qubit_count!r}, "
                f"qubit_count={self.qubit_count!r}"
            )
        if self.distance > self.qubit_count:
            raise ValueError(
                "distance must not exceed qubit_count; got "
                f"distance={self.distance!r}, qubit_count={self.qubit_count!r}"
            )
        if self.commit_rounds_override is not None:
            _require_positive_int(
                self.commit_rounds_override, "commit_rounds_override"
            )
        has_buffer_override = self.buffer_rounds_override is not None
        if has_buffer_override and self.buffer_rounds_override < 0:
            raise ValueError("buffer_rounds_override must be nonnegative")
        round_microseconds = _optional_float(self.round_microseconds)
        object.__setattr__(self, "round_microseconds", round_microseconds)

    @property
    def name(self) -> str:
        """The routing and readout identity of this card."""
        return (
            f"bivariate-bicycle code [[{self.qubit_count},"
            f"{self.logical_qubit_count},{self.distance}]]"
        )

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle: the distance."""
        return self.distance

    def round_period_us(self) -> Optional[float]:
        """The card's own round period, or None for the run's cadence."""
        return self.round_microseconds

    def buffering_floor(self) -> tuple[int, int]:
        """No buffering floor: (0, 0)."""
        return (0, 0)

    def commit_rounds(self) -> int:
        """Rounds committed per decode window; the distance by default."""
        if self.commit_rounds_override is None:
            return self.distance
        return self.commit_rounds_override

    def buffer_rounds(self) -> int:
        """Look-ahead rounds per decode window; none by default."""
        if self.buffer_rounds_override is None:
            return 0
        return self.buffer_rounds_override

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph nodes per round: the n checks of every patch."""
        return num_patches * self.qubit_count

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Bits read out per round: the n X-plus-Z checks of every patch."""
        return num_patches * self.qubit_count


def _require_positive_int(value, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}")


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _check_justification(value) -> None:
    if value is None:
        return
    is_text = isinstance(value, str)
    if not is_text or not value.strip():
        raise ValueError(
            "window_floor_justification must be a non-empty string or None"
        )
