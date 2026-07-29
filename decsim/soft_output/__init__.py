"""Soft-output confidence for decoder switching."""
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


from .complementary import (
    COMPLEMENTARY_GAP_SOURCE,
    ComplementaryGapMetric,
    ComplementaryGapMetricFactory,
    dem_to_matrices,
)
from .decoder import SoftOutputDecoder
from .cluster import (
    UNION_FIND_CLUSTER_GAP_SOURCE,
    UnionFindClusterGapDecoder,
)

__all__ = [
    "SoftOutput",
    "SoftOutputMetric",
    "ComplementaryGapMetric",
    "ComplementaryGapMetricFactory",
    "COMPLEMENTARY_GAP_SOURCE",
    "dem_to_matrices",
    "SoftOutputDecoder",
    "UNION_FIND_CLUSTER_GAP_SOURCE",
    "UnionFindClusterGapDecoder",
]
