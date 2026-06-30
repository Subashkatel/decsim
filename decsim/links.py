"""Classical communication links used by the runtime stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .config import us


@dataclass
class Link:
    """One classical channel: constant latency, plus size-dependent cost if bandwidth set; serialize=True queues on a shared bus."""
    latency_ticks: int
    bandwidth_bits_per_us: Optional[float] = None  # None = infinite (size ignored)
    serialize: bool = False                        # True = shared bus, queues messages
    next_free_tick: int = 0                        # shared-bus bookkeeping (internal)

    def __post_init__(self):
        if self.latency_ticks < 0:
            raise ValueError(f"latency_ticks must be >= 0 (got {self.latency_ticks})")
        if self.bandwidth_bits_per_us is not None and self.bandwidth_bits_per_us <= 0:
            raise ValueError(f"bandwidth_bits_per_us must be > 0 "
                             f"(got {self.bandwidth_bits_per_us})")

    def cost(self, bits: Optional[int] = None, now: Optional[int] = None) -> int:
        """Return ticks from now until this message is delivered."""
        serialization = us(bits / self.bandwidth_bits_per_us) \
            if (bits is not None and self.bandwidth_bits_per_us is not None) else 0
        if self.serialize and now is not None:
            start = max(now, self.next_free_tick)
            self.next_free_tick = start + serialization
            return (start - now) + serialization + self.latency_ticks
        return serialization + self.latency_ticks


def _as_link(value: Union[int, Link]) -> Link:
    """Accept a plain latency (ticks) or a full Link object."""
    return value if isinstance(value, Link) else Link(int(value))


class LinkModel:
    """Named classical-stack links (qc/cd/dd/do/oc/cq + ws weak->strong handoff); each arg is a latency or Link."""  # ref: DecLat Table 2
    def __init__(self, qc: Union[int, Link] = us(0.15), cd: Union[int, Link] = us(2.0),
                 dd: Union[int, Link] = us(0.5), do: Union[int, Link] = us(1.0),
                 oc: Union[int, Link] = us(4.0), cq: Union[int, Link] = us(0.15),
                 ws: Union[int, Link] = us(0.5)):
        """Store all link objects."""
        self.qc = _as_link(qc)
        self.cd = _as_link(cd)
        self.dd = _as_link(dd)
        self.do = _as_link(do)
        self.oc = _as_link(oc)
        self.cq = _as_link(cq)
        self.ws = _as_link(ws)
