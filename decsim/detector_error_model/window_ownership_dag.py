"""Compile one fault owner per window from the window dependency DAG.

Validates the declared edges, computes dependency depths and ancestor closures,
assigns every fault to its shallowest owning window, and collects the fault
sets a window's predecessors already own.

Carries no runtime dependency on the slicing engine: WindowSlicer is used only
through attribute reads and a string annotation, so its import is
TYPE_CHECKING-only and this compiler is importable and testable on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .fault_model_contracts import FaultRepresentation
    from .window_slicer import WindowSlicer


def _dependency_depths(window_count: int, dependency_edges: tuple) -> tuple[int, ...]:
    """Validate the dependency DAG and return each window's depth."""
    predecessors = [set() for _ in range(window_count)]
    for source, destination in dependency_edges:
        if source < 0 or destination < 0:
            raise ValueError("window dependency indices must be nonnegative")
        predecessors[destination].add(source)

    depths: list[Optional[int]] = [None] * window_count
    while any(depth is None for depth in depths):
        progressed = False
        for window_index, incoming in enumerate(predecessors):
            if depths[window_index] is not None:
                continue
            if any(depths[source] is None for source in incoming):
                continue
            depths[window_index] = (
                0
                if not incoming
                else 1 + max(depths[source] for source in incoming)
            )
            progressed = True
        if not progressed:
            raise ValueError("window dependencies must form an acyclic graph")
    return tuple(depths)


def _dependency_ancestors(
    window_count: int,
    dependency_edges: tuple,
    depths: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """Return every direct and indirect predecessor of each window."""
    incoming = [set() for _ in range(window_count)]
    for source, destination in dependency_edges:
        incoming[destination].add(source)
    ancestors = [set() for _ in range(window_count)]
    for destination in sorted(range(window_count), key=depths.__getitem__):
        for source in incoming[destination]:
            ancestors[destination].add(source)
            ancestors[destination].update(ancestors[source])
    return tuple(frozenset(nodes) for nodes in ancestors)


def _explicit_prior_faults(
    ownership: tuple[dict[FaultRepresentation, set[int]], ...],
    ancestors: tuple[frozenset[int], ...],
) -> tuple[dict[FaultRepresentation, set[int]], ...]:
    """Collect the faults owned by each window's predecessors."""
    representations = tuple(ownership[0])
    return tuple(
        {
            representation: set().union(*(
                ownership[ancestor][representation]
                for ancestor in ancestor_indices
            ))
            for representation in representations
        }
        for ancestor_indices in ancestors
    )


def _explicit_fault_ownership(
    slicer: WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    depths: tuple[int, ...],
    *,
    round_count: int,
) -> tuple[dict[FaultRepresentation, set[int]], ...]:
    """Assign each fault to one shallowest commit window in the DAG.

    Equal-depth ambiguity is rejected. A plan that covers the full operation
    must assign every fault in every requested representation.
    """
    ownership = [
        {representation: set() for representation in slicer.catalogs}
        for _ in entries
    ]
    covers_full_operation = (
        entries[0][1] == 1 and entries[-1][2] == round_count
    )
    for representation, catalog in slicer.catalogs.items():
        for fault_index, detector_ids in enumerate(catalog.detector_sets):
            candidates = [
                window_index
                for window_index, (_, commit_lo, commit_hi, _) in enumerate(entries)
                if any(
                    commit_lo <= slicer.round_of[detector_id] <= commit_hi
                    for detector_id in detector_ids
                )
            ]
            if not candidates:
                if covers_full_operation:
                    raise ValueError(
                        f"{representation.value} fault {fault_index} touches no "
                        "window commit region"
                    )
                continue
            earliest_depth = min(depths[index] for index in candidates)
            earliest = [
                index for index in candidates
                if depths[index] == earliest_depth
            ]
            if len(earliest) != 1:
                raise ValueError(
                    f"{representation.value} fault {fault_index} straddles "
                    "independent commit regions without a causal owner"
                )
            ownership[earliest[0]][representation].add(fault_index)
    return tuple(ownership)
