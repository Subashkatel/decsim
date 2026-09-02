"""What a window plan must look like to be decoded under a named protocol.

GENERIC accepts any plan. TAN_ZERO_SEAM_GRAPHLIKE is the sandwich
construction of Tan et al. (2209.09219, section "Sandwich decoder"):
odd-numbered windows are seams one detector layer wide, each seam depends
on the two windows beside it and on nothing else, and every seam is a
closed time boundary. The construction is validated for the graphlike
(matching) representation only, so a plan that has seams and asks for
anything else is refused.

A closed time boundary (Tan's word; Skoric et al. 2209.08552 call the
same boundary smooth) is one where the decoder must not invent an
artificial boundary edge. So a fault that flips a detector inside such a
window and another outside it is refused, because slicing it to the
window would create exactly that edge.

The protocol seam is closed: the dispatch accepts these two members and
rejects every other, and a third protocol needs an edit here.
"""

from typing import Optional

import decsim.message as message
from decsim.detector_error_model import fault_model_contracts, window_slicer


def validate_closed_temporal_boundary_windows(
    slicer: window_slicer.WindowSlicer,
    models: list[fault_model_contracts.WindowErrorModel],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_windows: tuple[int, ...],
) -> None:
    """Refuse a closed time boundary that cuts a fault of the circuit."""
    if not closed_windows:
        return
    # A closed window waits for the windows beside it, so it is always a
    # dependency destination; a plan with no edges has none.
    edges = dependency_edges or ()
    destinations = {destination for _, destination in edges}
    for window_index in closed_windows:
        if window_index not in destinations:
            raise ValueError(
                "closed temporal boundary window must be a dependency "
                "destination"
            )
        _check_window_cuts_no_fault(slicer, models[window_index], window_index)


def validate_window_protocol(
    entries: tuple[tuple[int, int, int, int], ...],
    window_protocol: message.WindowProtocol,
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_windows: tuple[int, ...],
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
) -> None:
    """Refuse a plan that does not meet its protocol's contract."""
    if window_protocol is message.WindowProtocol.GENERIC:
        return
    if window_protocol is not message.WindowProtocol.TAN_ZERO_SEAM_GRAPHLIKE:
        raise ValueError("unsupported window protocol")
    seam_indices = tuple(range(1, len(entries), 2))
    graphlike_only = frozenset(
        {fault_model_contracts.FaultRepresentation.GRAPHLIKE}
    )
    is_graphlike_only = (
        fault_model_requirement.representations == graphlike_only
    )
    if seam_indices and not is_graphlike_only:
        raise ValueError(
            "Tan's validated zero-seam memory construction requires exactly "
            "the graphlike correction-edge representation"
        )
    if set(closed_windows) != set(seam_indices):
        raise ValueError(
            "every Tan type-2 seam, and only a seam, must be temporally closed"
        )
    for seam_index in seam_indices:
        _check_seam_is_one_layer(entries[seam_index])
        _check_seam_has_a_task_after_it(seam_index, len(entries))
    expected_edges = _seam_edges(seam_indices)
    declared_edges = dependency_edges or ()
    if set(declared_edges) != set(expected_edges):
        raise ValueError(
            "each Tan type-2 seam must depend on its two adjacent type-1 tasks"
        )


def _check_window_cuts_no_fault(
    slicer: window_slicer.WindowSlicer,
    model: fault_model_contracts.WindowErrorModel,
    window_index: int,
) -> None:
    local_detector_ids = set(model.detector_ids)
    for representation, catalog in slicer.catalogs.items():
        faults = model.require_faults(representation)
        for source_fault_id in faults.source_fault_ids:
            global_detector_ids = set(catalog.detector_sets[source_fault_id])
            _check_fault_not_truncated(
                global_detector_ids,
                local_detector_ids,
                representation,
                window_index,
                source_fault_id,
            )


def _check_fault_not_truncated(
    global_detector_ids: set[int],
    local_detector_ids: set[int],
    representation: fault_model_contracts.FaultRepresentation,
    window_index: int,
    source_fault_id: int,
) -> None:
    local_effect = global_detector_ids & local_detector_ids
    if local_effect and local_effect != global_detector_ids:
        raise ValueError(
            f"{representation.value} closed temporal boundary "
            f"window {window_index} truncates global fault "
            f"{source_fault_id}; a smooth B boundary cannot "
            "contain an artificial boundary generator"
        )


def _check_seam_is_one_layer(entry: tuple[int, int, int, int]) -> None:
    first_buffer, first_commit, last_commit, last_buffer = entry
    if not first_buffer == first_commit == last_commit == last_buffer:
        raise ValueError(
            "a zero-offset Tan type-2 seam must be one detector layer"
        )


def _check_seam_has_a_task_after_it(seam_index: int, window_count: int) -> None:
    """A sandwich ends on a type-1 task, so its window count is odd."""
    window_after = seam_index + 1
    if window_after >= window_count:
        raise ValueError(
            f"Tan type-2 seam {seam_index} has no type-1 task after it: "
            f"window {window_after} is outside the plan of {window_count} "
            "windows"
        )


def _seam_edges(seam_indices: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Each seam depends on the window before it and the window after it."""
    edges = []
    for seam_index in seam_indices:
        window_before = seam_index - 1
        window_after = seam_index + 1
        edges.append((window_before, seam_index))
        edges.append((window_after, seam_index))
    return tuple(edges)
