"""Classical-stack links: constant latency plus optional bandwidth/queueing.

Part module: Link (one channel) and LinkModel (the named qc/cd/dd/do/oc/cq/ws
set used across the controller, window manager, and decoder hops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .config import us


@dataclass
class Link:
    """One classical channel: constant latency, plus size-dependent cost if
    bandwidth set; serialize=True queues on a shared bus."""

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

    def run_manifest_config(self):
        return {"kind": "link"}

    def cost(self, bits: Optional[int] = None, now: Optional[int] = None) -> int:
        """Return ticks from now until this message is delivered."""
        serialization = us(bits / self.bandwidth_bits_per_us) \
            if (bits is not None and self.bandwidth_bits_per_us is not None) else 0
        if self.serialize and now is not None:
            start = max(now, self.next_free_tick)
            self.next_free_tick = start + serialization
            return (start - now) + serialization + self.latency_ticks
        return serialization + self.latency_ticks


def _as_link(value: Union[int, "Link"]) -> "Link":
    return value if isinstance(value, Link) else Link(int(value))


class LinkModel:
    """Named classical-stack links (qc/cd/dd/do/oc/cq + ws weak->strong)."""

    def __init__(self, qc=us(0.15), cd=us(2.0), dd=us(0.5), do=us(1.0),
                 oc=us(4.0), cq=us(0.15), ws=us(0.5)):
        self.qc = _as_link(qc)
        self.cd = _as_link(cd)
        self.dd = _as_link(dd)
        self.do = _as_link(do)
        self.oc = _as_link(oc)
        self.cq = _as_link(cq)
        self.ws = _as_link(ws)

    @classmethod
    def from_timing(cls, timing) -> "LinkModel":
        """Build from a core TimingConfig."""
        return cls(qc=timing.ticks("t_qc"), cd=timing.ticks("t_cd"),
                   dd=timing.ticks("t_dd"), do=timing.ticks("t_do"),
                   oc=timing.ticks("t_oc"), cq=timing.ticks("t_cq"),
                   ws=timing.ticks("t_ws"))


def link_compression_decision(raw_bits_per_msg: float,
                              packed_bits_per_msg: float,
                              msgs_per_us: float,
                              bandwidth_bits_per_us: float,
                              headroom: float = 0.9,
                              buffer_bound: bool = False) -> dict:
    """The deck's row-22 rule: compress ON THE LINK only when
    BANDWIDTH is the binding constraint (Gate 7 P18).

    util_* = offered bits/us over bandwidth. When the buffer is the
    binding constraint instead, compression belongs in the STORE
    (V23 packed retention), not the wire — the rule returns
    compress_link=False with binding="buffer" so callers route the
    effort to the right place. sufficient=False flags bandwidth-
    binding cases packing alone cannot relieve.
    """
    if bandwidth_bits_per_us <= 0 or msgs_per_us < 0:
        raise ValueError("need bandwidth > 0 and msgs_per_us >= 0")
    util_raw = raw_bits_per_msg * msgs_per_us / bandwidth_bits_per_us
    util_packed = packed_bits_per_msg * msgs_per_us / bandwidth_bits_per_us
    bandwidth_binding = util_raw > headroom
    if bandwidth_binding:
        binding = "bandwidth"
    elif buffer_bound:
        binding = "buffer"
    else:
        binding = "none"
    compress_link = bandwidth_binding and util_packed <= headroom
    sufficient = (not bandwidth_binding) or util_packed <= headroom
    return {"util_raw": util_raw, "util_packed": util_packed,
            "binding": binding, "compress_link": compress_link,
            "sufficient": sufficient}
