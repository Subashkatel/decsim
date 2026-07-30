"""Code models used by planning, timing, and metrics.

A code model is a small frozen card of numbers, not a stabilizer code:
the simulator prices decoder timing, so all it needs from a QEC code is
window sizes, decoding-graph size per round, and syndrome bandwidth.
The numbers can be set by hand or taken from any upstream tool's captured
output (e.g. a QLX decoder-params artifact); decsim itself never imports
or requires such tools and runs standalone."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .config import us


def _check_round_us(value):
    if value is not None and (
        value <= 0 or not math.isfinite(value) or us(value) < 1
    ):
        raise ValueError(f"round_us must be finite and at least one tick; got {value!r}")


@dataclass(frozen=True)
class SurfaceCodeModel:
    """Rotated surface-code timing and sizing model."""

    d: int = 3                                   # code distance
    round_us: Optional[float] = None             # per-code round period; None = global cadence
    commit_rounds_override: Optional[int] = None  # window commit size; None = d
    buffer_rounds_override: Optional[int] = None  # window look-ahead size; None = d

    def __post_init__(self) -> None:
        """Validate the built-in Surface timing and sizing card."""
        for label, value in (
            ("d", self.d),
            ("commit_rounds_override", self.commit_rounds_override),
            ("buffer_rounds_override", self.buffer_rounds_override),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive; got {value!r}")
        _check_round_us(self.round_us)

    @property
    def name(self) -> str:
        """The code's human-readable name."""
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
        """Literature buffering floor per side: (lead, trail) = (d, d)
        (Skoric n_buf=d, arXiv:2209.08552; Bombin b>=d, arXiv:2303.04846)."""
        return (self.d, self.d)

    def buffer_floor_override_active(self) -> bool:
        return self.buffer_rounds_override is not None

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count per round for this many patches (drives
        decode latency). Multi-patch ops add one d-node strip for the seam
        where patches merge."""
        patch_count = max(1, num_patches)
        return patch_count * self.d * self.d + (self.d if patch_count > 1 else 0)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round: the d^2 - 1 stabilizers of a
        rotated surface-code patch."""
        return max(1, num_patches) * (self.d * self.d - 1)


@dataclass(frozen=True)
class BBCodeModel:
    """Bivariate-bicycle gross-code estimate model ([[144,12,12]],
    Bravyi et al. arXiv:2308.07915).

    This is the CodeModel port's second implementation: a code whose
    decoding graph is NOT d^2 per patch keeps surface-code assumptions
    from leaking into the planning seams.

    n_detectors was captured from a reference gross-code memory-experiment
    DEM; note n_detectors/d = 78 detectors per round, which is not the
    code's 132 checks (DEM detectors differ from raw checks at the first/
    last rounds). The same capture also recorded num_checks=132 and
    n_faults=8784, kept here for the record since nothing consumes them."""

    n: int = 144                     # physical qubits
    k: int = 12                      # logical qubits
    d: int = 12                      # code distance
    n_detectors: int = 936           # captured DEM detector count (see above)
    round_us: Optional[float] = None  # per-code round period; None = global cadence

    def __post_init__(self) -> None:
        """Validate the built-in BB timing and sizing card."""
        for field_name in ("n", "k", "d", "n_detectors"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.n_detectors % self.d != 0:
            raise ValueError(
                "n_detectors must be divisible by d; "
                f"got n_detectors={self.n_detectors!r}, d={self.d!r}"
            )
        _check_round_us(self.round_us)

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"bivariate-bicycle / gross code [[{self.n},{self.k},{self.d}]]"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def round_period_us(self) -> Optional[float]:
        return self.round_us

    def buffering_floor(self) -> tuple[int, int]:
        """Literature buffering floor per side: (lead, trail) = (d, d)
        (Skoric n_buf=d, arXiv:2209.08552; Bombin b>=d, arXiv:2303.04846)."""
        return (self.d, self.d)

    def buffer_floor_override_active(self) -> bool:
        return False

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Per-round detector count used for accounting."""
        return max(1, num_patches) * (self.n_detectors // self.d)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round (~ checks per round)."""
        return max(1, num_patches) * (self.n_detectors // self.d)
