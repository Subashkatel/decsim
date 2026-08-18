"""Code models used by planning, timing, and metrics.

A code model is a small frozen card of numbers, not a stabilizer code:
the simulator prices decoder timing, so all it needs from a QEC code is
window sizes, decoding-graph size per round, and syndrome bandwidth.
The numbers can be set by hand or taken from any upstream tool's captured
output (e.g. a QLX decoder-params artifact); decsim itself never imports
or requires such tools and runs standalone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _require_positive_int(value, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive; got {value!r}")


def _check_round_us(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class SurfaceCodeModel:
    """Rotated surface-code timing and sizing model."""

    d: int = 3                                   # code distance
    round_us: Optional[float] = None             # per-code round period; None = global cadence
    commit_rounds_override: Optional[int] = None  # window commit size; None = d
    buffer_rounds_override: Optional[int] = None  # window look-ahead size; None = d

    def __post_init__(self) -> None:
        """Normalize the optional per-code cadence."""
        object.__setattr__(self, "round_us", _check_round_us(self.round_us))

    @property
    def name(self) -> str:
        """Stable routing and readout identity for this code card."""
        return f"rotated surface code (d={self.d})"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def round_period_us(self) -> Optional[float]:
        return self.round_us

    def commit_rounds(self) -> int:
        """Rounds committed per decode window (commit_rounds_override, else d)."""
        return self.commit_rounds_override if self.commit_rounds_override is not None else self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window (buffer_rounds_override, else d)."""
        return self.buffer_rounds_override if self.buffer_rounds_override is not None else self.d

    def buffering_floor(self) -> tuple[int, int]:
        """Return the minimum leading and trailing buffers: ``(d, d)``."""
        return (self.d, self.d)

    def buffer_floor_override_active(self) -> bool:
        return self.buffer_rounds_override is not None

    def spatial_nodes(self, num_patches: int) -> int:
        """Bulk timing proxy with a heuristic d-node multi-patch seam strip."""
        return num_patches * self.d * self.d + (
            self.d if num_patches > 1 else 0
        )

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round: the d^2 - 1 stabilizers of a
        rotated surface-code patch."""
        return num_patches * (self.d * self.d - 1)


@dataclass(frozen=True)
class BBCodeModel:
    """Timing model for a bivariate-bicycle CSS code.

    One modeled syndrome round is one complete extraction cycle measuring
    ``n/2`` X checks and ``n/2`` Z checks. Exact window-local detector rows
    remain owned by the detector error model.
    """

    n: int = 144                     # physical qubits
    k: int = 12                      # logical qubits
    d: int = 12                      # code distance
    round_us: Optional[float] = None  # per-code round period; None = global cadence
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the built-in BB timing and sizing card."""
        for field_name in ("n", "k", "d"):
            _require_positive_int(getattr(self, field_name), field_name)
        if self.n % 2:
            raise ValueError(f"n must be even; got {self.n!r}")
        if self.k > self.n:
            raise ValueError(f"k must not exceed n; got k={self.k!r}, n={self.n!r}")
        if self.d > self.n:
            raise ValueError(f"d must not exceed n; got d={self.d!r}, n={self.n!r}")
        if self.commit_rounds_override is not None:
            _require_positive_int(
                self.commit_rounds_override, "commit_rounds_override"
            )
        if self.buffer_rounds_override is not None and self.buffer_rounds_override < 0:
            raise ValueError("buffer_rounds_override must be nonnegative")
        object.__setattr__(self, "round_us", _check_round_us(self.round_us))

    @property
    def name(self) -> str:
        """Stable routing and readout identity for this code card."""
        return f"bivariate-bicycle code [[{self.n},{self.k},{self.d}]]"

    @property
    def distance(self) -> int:
        """Configured code distance."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Logical cycle modeled as d syndrome rounds."""
        return self.d

    def round_period_us(self) -> Optional[float]:
        return self.round_us

    def buffering_floor(self) -> tuple[int, int]:
        return (0, 0)

    def buffer_floor_override_active(self) -> bool:
        return self.buffer_rounds_override is not None

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return (
            self.d
            if self.commit_rounds_override is None
            else self.commit_rounds_override
        )

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return (
            0
            if self.buffer_rounds_override is None
            else self.buffer_rounds_override
        )

    def spatial_nodes(self, num_patches: int) -> int:
        """Combined check count used as a bulk timing proxy."""
        return num_patches * self.n

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Raw X-plus-Z check bits measured per modeled round."""
        return num_patches * self.n
