"""Soft-output metrics (SoftOutputMetric seam): a confidence ``g`` per decoding window.

Interchangeable behind one interface: ComplementaryGapMetric (MWPM) and ClusterGapMetric (UF).
"""
# ref: Toshio et al. 2510.25222 Sec. II.B
from typing import Protocol, runtime_checkable

from ..message import SoftOutput


@runtime_checkable
class SoftOutputMetric(Protocol):
    """Computes a soft output g per window (smaller g = lower confidence);
    swappable across metrics."""  # ref: paper Sec. II.B

    @property
    def name(self) -> str: ...

    def evaluate(self, syndrome) -> SoftOutput: ...


from .cluster import (
    ClusterGapMetric,
    RECONSTRUCTED_CLUSTER_GAP_CORRECTED_SOURCE,
    RECONSTRUCTED_CLUSTER_GAP_SOURCE,
)
from .complementary import (
    COMPLEMENTARY_GAP_SOURCE,
    ComplementaryGapMetric,
    dem_to_matrices,
)
from .decoder import SoftOutputDecoder

__all__ = [
    "SoftOutput",
    "SoftOutputMetric",
    "ComplementaryGapMetric",
    "COMPLEMENTARY_GAP_SOURCE",
    "ClusterGapMetric",
    "RECONSTRUCTED_CLUSTER_GAP_SOURCE",
    "RECONSTRUCTED_CLUSTER_GAP_CORRECTED_SOURCE",
    "dem_to_matrices",
    "SoftOutputDecoder",
]
