"""Core non-link timing and integer-tick conversions."""
from __future__ import annotations

from dataclasses import dataclass

TICKS_PER_US = 1_000_000


def us(microseconds: float) -> int:
    """Convert microseconds to integer ticks."""
    return int(round(microseconds * TICKS_PER_US))


def fmt(ticks: int) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


@dataclass(frozen=True)
class TimingConfig:
    """Run-wide non-link timing quantities, expressed in microseconds."""

    round_us: float = 1.1          # QEC round period (one syndrome-extraction cycle)
    t_pack_us: float = 0.0         # controller packet assembly before CWD send

    def ticks(self, name: str) -> int:
        """Return the one named non-link timing quantity in integer ticks."""
        if name != "t_pack":
            raise ValueError(f"unknown non-link timing quantity {name!r}")
        return us(self.t_pack_us)

    @property
    def round_ticks(self) -> int:
        return us(self.round_us)
