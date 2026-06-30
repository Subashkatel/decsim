"""Soft-output metrics (SoftOutputMetric seam): a confidence ``g`` per decoding window.

Interchangeable behind one interface: ComplementaryGapMetric (MWPM) and ClusterGapMetric (UF).
"""
# ref: Toshio et al. 2510.25222 Sec. II.B
from ..message import SoftOutput
from ..protocols import SoftOutputMetric
from .cluster import ClusterGapMetric
from .complementary import ComplementaryGapMetric, dem_to_matrices
from .decoder import SoftOutputDecoder

__all__ = [
    "SoftOutput",
    "SoftOutputMetric",
    "ComplementaryGapMetric",
    "ClusterGapMetric",
    "dem_to_matrices",
    "SoftOutputDecoder",
]
