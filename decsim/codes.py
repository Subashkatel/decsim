
"""Code models used by planning, timing, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SurfaceCodeModel:
    """Rotated surface-code timing and sizing model."""

    d: int = 3
    round_us: Optional[float] = None
    commit_rounds_override: Optional[int] = None
    buffer_rounds_override: Optional[int] = None
    mu_mem: float = 0.019
    lam_mem: float = 9.3

    def __post_init__(self) -> None:
        """Reject non-positive commit/buffer overrides (None means 'use d')."""
        for label, value in (("commit_rounds_override", self.commit_rounds_override),
                             ("buffer_rounds_override", self.buffer_rounds_override)):
            if value is not None and value < 1:
                raise ValueError(f"{label} must be a positive number of rounds; got {value}")

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

    def rounds_per_op(self) -> int:
        """Syndrome rounds run for one operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window (commit_rounds_override, else d)."""
        return self.commit_rounds_override if self.commit_rounds_override is not None else self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window (buffer_rounds_override, else d)."""
        return self.buffer_rounds_override if self.buffer_rounds_override is not None else self.d

    def memory_error(self, rounds: int) -> float:
        """Analytic logical error for idle memory rounds."""
        return self.mu_mem * self.d * rounds * self.lam_mem ** (-(self.d + 1) / 2)

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        patch_count = max(1, num_patches)
        return patch_count * self.d * self.d + (self.d if patch_count > 1 else 0)

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return max(1, num_patches) * (self.d * self.d - 1)


@dataclass(frozen=True)
class BBCodeModel:
    """Bivariate-bicycle gross-code estimate model."""

    n: int = 144
    k: int = 12
    d: int = 12
    num_checks: int = 132
    n_detectors: int = 936
    n_faults: int = 8784
    round_us: Optional[float] = None

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

    def rounds_per_op(self) -> int:
        """Syndrome rounds run for one operation."""
        return self.rounds_per_logical_cycle()

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


@dataclass(frozen=True)
class ColorCodeModel:
    """Triangular color-code estimate model."""

    d: int = 3
    node_factor: float = 0.75
    round_us: Optional[float] = None

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"triangular color code estimate (d={self.d})"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Syndrome rounds run for one operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        return max(1, int(round(max(1, num_patches) * self.node_factor * self.d * self.d)))

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return self.spatial_nodes(num_patches)


@dataclass(frozen=True)
class ToricCodeModel:
    """Toric-code estimate model."""

    d: int = 3
    round_us: Optional[float] = None

    @property
    def name(self) -> str:
        """The code's human-readable name."""
        return f"toric code estimate (d={self.d})"

    @property
    def distance(self) -> int:
        """Code distance d (errors up to ~d/2 are corrected)."""
        return self.d

    def rounds_per_logical_cycle(self) -> int:
        """Syndrome rounds per logical cycle."""
        return self.d

    def rounds_per_op(self) -> int:
        """Syndrome rounds run for one operation."""
        return self.rounds_per_logical_cycle()

    def commit_rounds(self) -> int:
        """Rounds committed per decode window."""
        return self.d

    def buffer_rounds(self) -> int:
        """Look-ahead buffer rounds per window."""
        return self.d

    def spatial_nodes(self, num_patches: int) -> int:
        """Decoding-graph node count for this many patches (drives decode latency)."""
        return max(1, num_patches) * 2 * self.d * self.d

    def syndrome_bits_per_round(self, num_patches: int) -> int:
        """Syndrome bits measured per round."""
        return max(1, num_patches) * 2 * self.d * self.d
