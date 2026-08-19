"""Window-protocol and temporal-boundary scientific policy.

Holds the Tan zero-seam graphlike contract and the closed-temporal-boundary
check over built models.  This is the single site where support for a further
window protocol would be added; the current dispatch accepts exactly the
GENERIC and TAN_ZERO_SEAM_GRAPHLIKE members and rejects every other, so the
protocol seam is closed rather than pluggable.

Carries no runtime dependency on the slicing engine: WindowSlicer and
WindowErrorModel are used only through attribute reads, member calls and string
annotations, so their imports are TYPE_CHECKING-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..message import WindowProtocol
from .fault_model_contracts import FaultRepresentation

if TYPE_CHECKING:
    from .fault_model_contracts import (
        DecoderFaultModelRequirement,
        WindowErrorModel,
    )
    from .window_slicer import WindowSlicer


def _validate_closed_temporal_boundary_windows(
    slicer: WindowSlicer,
    models: list[WindowErrorModel],
    dependency_edges: Optional[tuple],
    closed_windows: tuple[int, ...],
) -> None:
    """Reject a declared closed time boundary if it cuts a global fault."""
    if not closed_windows:
        return
    destinations = {destination for _, destination in dependency_edges}
    for window_index in closed_windows:
        if window_index not in destinations:
            raise ValueError(
                "closed temporal boundary window must be a dependency destination"
            )
        model = models[window_index]
        local_detector_ids = set(model.detector_ids)
        for representation, catalog in slicer.catalogs.items():
            faults = model.require_faults(representation)
            for source_fault_id in faults.source_fault_ids:
                global_detector_ids = set(
                    catalog.detector_sets[source_fault_id]
                )
                local_effect = global_detector_ids & local_detector_ids
                if local_effect and local_effect != global_detector_ids:
                    raise ValueError(
                        f"{representation.value} closed temporal boundary "
                        f"window {window_index} truncates global fault "
                        f"{source_fault_id}; a smooth B boundary cannot "
                        "contain an artificial boundary generator"
                    )


def _validate_window_protocol(
    entries: tuple[tuple[int, int, int, int], ...],
    window_protocol: WindowProtocol,
    dependency_edges: Optional[tuple],
    closed_windows: tuple[int, ...],
    fault_model_requirement: DecoderFaultModelRequirement,
) -> None:
    """Fail closed unless a Tan plan has the exact zero-seam contract."""
    if window_protocol is WindowProtocol.GENERIC:
        return
    if window_protocol is not WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE:
        raise ValueError("unsupported window protocol")

    seam_indices = tuple(range(1, len(entries), 2))
    if seam_indices and fault_model_requirement.representations != frozenset({
        FaultRepresentation.GRAPHLIKE
    }):
        raise ValueError(
            "Tan's validated zero-seam memory construction requires exactly the "
            "graphlike correction-edge representation"
        )
    if set(closed_windows) != set(seam_indices):
        raise ValueError(
            "every Tan type-2 seam, and only a seam, must be temporally closed"
        )
    for seam_index in seam_indices:
        buffer_lo, commit_lo, commit_hi, buffer_hi = entries[seam_index]
        if not buffer_lo == commit_lo == commit_hi == buffer_hi:
            raise ValueError("a zero-offset Tan type-2 seam must be one detector layer")
    expected_edges = tuple(
        edge
        for seam_index in seam_indices
        for edge in ((seam_index - 1, seam_index),
                     (seam_index + 1, seam_index))
    )
    if set(dependency_edges or ()) != set(expected_edges):
        raise ValueError(
            "each Tan type-2 seam must depend on its two adjacent type-1 tasks"
        )
