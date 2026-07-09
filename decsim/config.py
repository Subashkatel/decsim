"""Core timing: integer ticks, conversions, and the classical-stack link latencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TICKS_PER_US = 1_000_000


def us(microseconds: float) -> int:
    """Convert microseconds to integer ticks."""
    return int(round(microseconds * TICKS_PER_US))


def fmt(ticks: int) -> str:
    """Format ticks as microseconds for readability in logs."""
    return f"{ticks / TICKS_PER_US:7.3f} us"


@dataclass(frozen=True)
class TimingConfig:
    """Round clock + per-hop link latencies (µs), converted on demand to ticks."""

    round_us: float = 1.1          # QEC round period (one syndrome-extraction cycle)
    t_qc_us: float = 0.15          # QPU -> controller: raw syndrome readout hop
    t_cd_us: float = 2.0           # controller -> decoder: syndrome packet delivery
    t_dd_us: float = 0.5           # decoder -> decoder: window-dependency handoff
    t_do_us: float = 1.0           # decoder -> orchestrator: decode result publish
    t_oc_us: float = 4.0           # orchestrator -> controller: decision return
    t_cq_us: float = 0.15          # controller -> QPU: feedback/correction delivery
    t_ws_us: Optional[float] = None  # weak -> strong escalation handoff; None = t_dd
    t_pack_us: float = 0.0         # controller packing time per packet, paid before t_cd

    def ticks(self, name: str) -> int:
        """Tick cost of a named hop, e.g. ticks('t_dd').

        t_ws=None falls back to this config's t_dd_us (the weak→strong
        handoff rides a decoder→decoder link unless priced separately)."""
        value = getattr(self, f"{name}_us")
        if name == "t_ws" and value is None:
            value = self.t_dd_us
        return us(value)

    @property
    def round_ticks(self) -> int:
        return us(self.round_us)
