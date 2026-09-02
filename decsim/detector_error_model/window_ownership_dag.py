"""Decides which window owns each fault when windows form a dependency graph.

A parallel window plan (Tan et al. 2209.09219; Skoric et al. 2209.08552,
section III) decodes some windows first and lets the windows between them
wait for both neighbours' committed corrections. The plan says so with
edges from a window to the windows that depend on it. This module checks
that those edges form an acyclic graph, gives every window a depth (zero
for a window that waits for nothing), and assigns every fault to the
shallowest window whose commit rounds it touches; a fault that touches
two windows of the same depth has no causal owner and is refused. It also
answers, for each window, which faults its ancestors own, so the slicer
can leave those out of the window's columns.
"""

from collections.abc import Container
from typing import Optional

from decsim.detector_error_model import fault_model_contracts, window_slicer


def dependency_depths(
    window_count: int, dependency_edges: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    """Each window's depth in the dependency graph.

    Raises ValueError for an index outside the plan or a cycle.
    """
    predecessors = [set() for _ in range(window_count)]
    for source, destination in dependency_edges:
        if source < 0 or destination < 0:
            raise ValueError("window dependency indices must be nonnegative")
        if source >= window_count or destination >= window_count:
            raise ValueError(
                f"window dependency edge ({source}, {destination}) names a "
                f"window outside the plan of {window_count} windows"
            )
        predecessors[destination].add(source)
    depths: list[Optional[int]] = [None] * window_count
    while any(depth is None for depth in depths):
        progressed = _assign_ready_depths(predecessors, depths)
        if not progressed:
            raise ValueError("window dependencies must form an acyclic graph")
    return tuple(depths)


def dependency_ancestors(
    window_count: int,
    dependency_edges: tuple[tuple[int, int], ...],
    depths: tuple[int, ...],
) -> tuple[frozenset[int], ...]:
    """Every direct and indirect predecessor of each window."""
    incoming = [set() for _ in range(window_count)]
    for source, destination in dependency_edges:
        incoming[destination].add(source)
    ancestors = [set() for _ in range(window_count)]

    def depth_of(window_index: int) -> int:
        return depths[window_index]

    by_depth = sorted(range(window_count), key=depth_of)
    for destination in by_depth:
        for source in incoming[destination]:
            ancestors[destination].add(source)
            ancestors[destination].update(ancestors[source])
    return tuple(frozenset(nodes) for nodes in ancestors)


def explicit_fault_ownership(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    depths: tuple[int, ...],
    *,
    round_count: int,
) -> tuple[dict[fault_model_contracts.FaultRepresentation, set[int]], ...]:
    """The faults each window owns: the shallowest window it touches.

    A plan that covers the whole operation must give every fault an owner.
    Raises ValueError for a fault with no owner or with two owners of the
    same depth.
    """
    ownership = [
        {representation: set() for representation in slicer.catalogs}
        for _ in entries
    ]
    first_commit_round = entries[0][1]
    last_commit_round = entries[-1][2]
    covers_full_operation = (
        first_commit_round == 1 and last_commit_round == round_count
    )
    windows_by_commit_round = _windows_by_commit_round(entries)
    for representation, catalog in slicer.catalogs.items():
        _assign_owners(
            ownership,
            representation,
            len(catalog.detector_sets),
            slicer.fault_index.fault_rounds[representation],
            windows_by_commit_round,
            depths,
            covers_full_operation,
        )
    return tuple(ownership)


def explicit_prior_faults(
    ownership: tuple[
        dict[fault_model_contracts.FaultRepresentation, set[int]], ...
    ],
    ancestors: tuple[frozenset[int], ...],
) -> tuple[
    dict[fault_model_contracts.FaultRepresentation, Container[int]], ...
]:
    """For each window, the faults its ancestors own, by representation.

    Each map answers membership only; it holds the owner map, not a set.
    """
    representations = tuple(ownership[0])
    owner_of_fault = {}
    for representation in representations:
        owner_of_fault[representation] = _owner_by_fault(
            ownership, representation
        )
    prior_faults = []
    for ancestor_indices in ancestors:
        by_representation = {}
        for representation in representations:
            by_representation[representation] = _AncestorOwnedFaults(
                owner_of_fault[representation], ancestor_indices
            )
        prior_faults.append(by_representation)
    return tuple(prior_faults)


class _AncestorOwnedFaults:
    """The faults a window's ancestors own, answered by membership.

    Holding the owner map instead of a materialised set keeps a long chain
    of windows linear in the catalog size.
    """

    def __init__(
        self, owner_of_fault: dict[int, int], ancestor_indices: frozenset[int]
    ):
        self.owner_of_fault = owner_of_fault
        self.ancestor_indices = ancestor_indices

    def __contains__(self, fault_index) -> bool:
        owner = self.owner_of_fault.get(fault_index)
        return owner in self.ancestor_indices


def _assign_ready_depths(
    predecessors: list, depths: list[Optional[int]]
) -> bool:
    """Give a depth to every window whose predecessors all have one."""
    progressed = False
    for window_index, incoming in enumerate(predecessors):
        if depths[window_index] is not None:
            continue
        if any(depths[source] is None for source in incoming):
            continue
        depths[window_index] = _depth_below(incoming, depths)
        progressed = True
    return progressed


def _depth_below(incoming: set, depths: list[Optional[int]]) -> int:
    if not incoming:
        return 0
    deepest_predecessor = max(depths[source] for source in incoming)
    return 1 + deepest_predecessor


def _windows_by_commit_round(entries: tuple) -> dict[int, list[int]]:
    """The windows whose commit rounds include each round."""
    windows_by_round: dict[int, list[int]] = {}
    for window_index, entry in enumerate(entries):
        _, first_commit_round, last_commit_round, _ = entry
        after_last_commit_round = last_commit_round + 1
        for round_index in range(first_commit_round, after_last_commit_round):
            windows = windows_by_round.setdefault(round_index, [])
            windows.append(window_index)
    return windows_by_round


def _assign_owners(
    ownership: list,
    representation,
    fault_count: int,
    fault_rounds: tuple,
    windows_by_commit_round: dict,
    depths: tuple,
    covers_full_operation: bool,
) -> None:
    """Record the owner of every fault of one representation."""
    for fault_index in range(fault_count):
        owner = _owner_window(
            fault_index,
            fault_rounds[fault_index],
            windows_by_commit_round,
            depths,
            representation,
            covers_full_operation,
        )
        if owner is None:
            continue
        ownership[owner][representation].add(fault_index)


def _owner_window(
    fault_index: int,
    rounds: tuple,
    windows_by_commit_round: dict,
    depths: tuple,
    representation,
    covers_full_operation: bool,
) -> Optional[int]:
    """The shallowest window whose commit rounds the fault touches."""
    candidates = set()
    for round_index in rounds:
        windows = windows_by_commit_round.get(round_index, ())
        candidates.update(windows)
    if not candidates:
        if covers_full_operation:
            raise ValueError(
                f"{representation.value} fault {fault_index} touches no "
                "window commit region"
            )
        return None
    earliest_depth = min(depths[index] for index in candidates)
    earliest = [
        index for index in sorted(candidates) if depths[index] == earliest_depth
    ]
    if len(earliest) != 1:
        raise ValueError(
            f"{representation.value} fault {fault_index} straddles "
            "independent commit regions without a causal owner"
        )
    return earliest[0]


def _owner_by_fault(ownership: tuple, representation) -> dict[int, int]:
    """Which window owns each fault of one representation."""
    owner_by_fault = {}
    for window_index, owned in enumerate(ownership):
        for fault_index in owned[representation]:
            owner_by_fault[fault_index] = window_index
    return owner_by_fault
