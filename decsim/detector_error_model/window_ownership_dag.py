"""Decides which window owns each fault when windows form a dependency graph.

A parallel window plan (Tan et al. 2209.09219; Skoric et al. 2209.08552,
section I.C) decodes some windows first and lets the windows between them
wait for both neighbours' committed corrections. The plan says so with
edges from a window to the windows that depend on it. This module checks
that those edges form an acyclic graph, gives every window a depth (zero
for a window that waits for nothing), and assigns every fault to the
shallowest window whose commit rounds it touches; a fault that touches
two windows of the same depth has no causal owner and is refused. The
terminal window, whose commit rounds reach the last round, also owns
every fault its rows see that no commit round of the plan reaches, the
same law the slicer applies when it advances ownership itself (Skoric et
al. 2209.08552, the last paragraph of section III, Methods (text lines
697-700): the commit region of the last window runs from the bottom of
the regular commit region to the last round). An exclusion range is
decsim's own device with no paper referent: a strong re-decode leaves
the faults the weak decoder already committed uncommitted
(decsim/decoders/strong_escalation). A fault a range keeps uncommitted
has no owner at all, and a fault nobody has committed stays in the
decoding graph of every window that sees it, because only that window
can explain a defect the fault causes. The module also answers, for each
window, which faults its ancestors own, so the slicer can leave those
out of the window's columns, and it refuses a linked plan in which a
window would keep a physical fault while an ancestor owns one of its
graphlike components on the window's rows.
"""

from collections.abc import Container, Sequence
from typing import Optional

import scipy.sparse

from decsim.detector_error_model import (
    fault_model_contracts,
    window_placement,
    window_slicer,
)


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
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> tuple[dict[fault_model_contracts.FaultRepresentation, set[int]], ...]:
    """The faults each window owns: the shallowest window it touches.

    A fault outside every commit round of the plan has no owner, which is
    how a plan covers part of an operation; the terminal window still
    owns every such fault its rows see. A fault touching an excluded
    round has no owner in any window. Raises ValueError for a fault with
    two owners of the same depth.
    """
    ownership = [
        {representation: set() for representation in slicer.catalogs}
        for _ in entries
    ]
    excluded_faults = _excluded_faults(slicer, fault_exclusion_ranges)
    _assign_commit_round_owners(
        ownership, slicer, entries, depths, excluded_faults
    )
    terminal_window = _terminal_window(entries, round_count)
    if terminal_window is not None:
        _assign_unowned_faults_to_the_terminal_window(
            ownership,
            slicer,
            entries[terminal_window],
            terminal_window,
            excluded_faults,
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


def check_every_kept_physical_fault_keeps_its_components(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    ownership: tuple[
        dict[fault_model_contracts.FaultRepresentation, set[int]], ...
    ],
    ancestors: tuple[frozenset[int], ...],
) -> None:
    """A kept physical fault keeps every component on the window's rows.

    A component's candidate owners are a subset of its physical parent's,
    so the shallowest-window rule can give the parent and the component
    to windows on different branches: a window keeps the parent while an
    ancestor owns a component on the window's rows. This is the law of
    the projection (window_placement.local_link_projection) when the
    components of one fault share no detector, which Stim's decomposition
    guarantees; a hand-written model whose components share a detector
    may be refused here where the projection would still balance. Raises
    ValueError naming the window, the fault, the component and the
    ancestor; a plan without a link has nothing to check.
    """
    if slicer.catalog_link is None:
        return
    physical = fault_model_contracts.FaultRepresentation.PHYSICAL
    graphlike = fault_model_contracts.FaultRepresentation.GRAPHLIKE
    owner_of_physical_fault = _owner_by_fault(ownership, physical)
    owner_of_component = _owner_by_fault(ownership, graphlike)
    components_by_physical_fault = _components_by_physical_fault(
        slicer.catalog_link
    )
    for window_index, entry in enumerate(entries):
        _check_a_window_keeps_whole_physical_faults(
            slicer,
            entry,
            window_index,
            ancestors[window_index],
            owner_of_physical_fault,
            owner_of_component,
            components_by_physical_fault,
        )


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

    def __contains__(self, fault_index: object) -> bool:
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


def _assign_commit_round_owners(
    ownership: list,
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    depths: tuple[int, ...],
    excluded_faults: dict[fault_model_contracts.FaultRepresentation, set[int]],
) -> None:
    """Give every fault of every representation its commit-round owner."""
    windows_by_commit_round = _windows_by_commit_round(entries)
    for representation, catalog in slicer.catalogs.items():
        _assign_owners(
            ownership,
            representation,
            len(catalog.detector_sets),
            slicer.fault_index.fault_rounds[representation],
            windows_by_commit_round,
            depths,
            excluded_faults[representation],
        )


def _assign_owners(
    ownership: list,
    representation: fault_model_contracts.FaultRepresentation,
    fault_count: int,
    fault_rounds: tuple,
    windows_by_commit_round: dict,
    depths: tuple,
    excluded_faults: set[int],
) -> None:
    """Record the owner of every fault of one representation."""
    for fault_index in range(fault_count):
        if fault_index in excluded_faults:
            continue
        owner = _owner_window(
            fault_index,
            fault_rounds[fault_index],
            windows_by_commit_round,
            depths,
            representation,
        )
        if owner is None:
            continue
        ownership[owner][representation].add(fault_index)


def _owner_window(
    fault_index: int,
    rounds: tuple,
    windows_by_commit_round: dict,
    depths: tuple,
    representation: fault_model_contracts.FaultRepresentation,
) -> Optional[int]:
    """The shallowest window whose commit rounds the fault touches."""
    candidates = set()
    for round_index in rounds:
        windows = windows_by_commit_round.get(round_index, ())
        candidates.update(windows)
    if not candidates:
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


def _terminal_window(
    entries: tuple[tuple[int, int, int, int], ...], round_count: int
) -> Optional[int]:
    """The window whose commit rounds reach the last round, if any does."""
    for window_index, entry in enumerate(entries):
        _, _, last_commit_round, _ = entry
        if last_commit_round == round_count:
            return window_index
    return None


def _assign_unowned_faults_to_the_terminal_window(
    ownership: list,
    slicer: window_slicer.WindowSlicer,
    terminal_entry: tuple[int, int, int, int],
    terminal_window: int,
    excluded_faults: dict[fault_model_contracts.FaultRepresentation, set[int]],
) -> None:
    """Every fault the terminal window's rows see and no window owns."""
    for representation in slicer.catalogs:
        owner_by_fault = _owner_by_fault(ownership, representation)
        seen_faults = _faults_seen_by_the_window(
            slicer, terminal_entry, representation
        )
        unowned_faults = seen_faults - set(owner_by_fault)
        unowned_faults -= excluded_faults[representation]
        ownership[terminal_window][representation].update(unowned_faults)


def _faults_seen_by_the_window(
    slicer: window_slicer.WindowSlicer,
    entry: tuple[int, int, int, int],
    representation: fault_model_contracts.FaultRepresentation,
) -> set[int]:
    """The faults of one representation that touch the window's rounds."""
    first_buffer_round, _, _, last_buffer_round = entry
    after_last_buffer_round = last_buffer_round + 1
    seen_rounds = range(first_buffer_round, after_last_buffer_round)
    faults_by_round = slicer.fault_index.faults_by_round[representation]
    seen_faults = set()
    for round_index in seen_rounds:
        faults = faults_by_round.get(round_index, ())
        seen_faults.update(faults)
    return seen_faults


def _components_by_physical_fault(
    catalog_link: scipy.sparse.csc_matrix,
) -> tuple[tuple[int, ...], ...]:
    """Each physical fault's graphlike components, read off the link once."""
    column_starts = catalog_link.indptr.tolist()
    component_rows = catalog_link.indices.tolist()
    components = []
    physical_fault_count = catalog_link.shape[1]
    for physical_fault in range(physical_fault_count):
        next_physical_fault = physical_fault + 1
        first_entry = column_starts[physical_fault]
        after_last_entry = column_starts[next_physical_fault]
        rows = component_rows[first_entry:after_last_entry]
        components.append(tuple(sorted(rows)))
    return tuple(components)


def _check_a_window_keeps_whole_physical_faults(
    slicer: window_slicer.WindowSlicer,
    entry: tuple[int, int, int, int],
    window_index: int,
    ancestor_indices: frozenset[int],
    owner_of_physical_fault: dict[int, int],
    owner_of_component: dict[int, int],
    components_by_physical_fault: tuple[tuple[int, ...], ...],
) -> None:
    """Refuse the first kept physical fault whose component an ancestor owns."""
    seen_physical_faults = _faults_seen_by_the_window(
        slicer, entry, fault_model_contracts.FaultRepresentation.PHYSICAL
    )
    seen_components = _faults_seen_by_the_window(
        slicer, entry, fault_model_contracts.FaultRepresentation.GRAPHLIKE
    )
    for physical_fault in sorted(seen_physical_faults):
        physical_owner = owner_of_physical_fault.get(physical_fault)
        if physical_owner in ancestor_indices:
            continue
        dropped = _component_an_ancestor_owns(
            components_by_physical_fault[physical_fault],
            seen_components,
            owner_of_component,
            ancestor_indices,
        )
        if dropped is None:
            continue
        component, component_owner = dropped
        raise ValueError(
            "a linked fault model requirement is refused for this plan: "
            f"window {window_index} keeps physical fault {physical_fault} "
            f"while window {component_owner}, which it depends on, owns "
            f"graphlike component {component} of that fault on rows window "
            f"{window_index} decodes"
        )


def _component_an_ancestor_owns(
    components: tuple[int, ...],
    seen_components: set[int],
    owner_of_component: dict[int, int],
    ancestor_indices: frozenset[int],
) -> Optional[tuple[int, int]]:
    """The first component on the window's rows an ancestor owns, and who."""
    for component in components:
        if component not in seen_components:
            continue
        component_owner = owner_of_component.get(component)
        if component_owner in ancestor_indices:
            return component, component_owner
    return None


def _excluded_faults(
    slicer: window_slicer.WindowSlicer,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> dict[fault_model_contracts.FaultRepresentation, set[int]]:
    """The faults of each representation that touch an excluded round."""
    excluded_faults = {}
    for representation, catalog in slicer.catalogs.items():
        every_fault = list(range(len(catalog.detector_sets)))
        excluded_faults[representation] = (
            window_placement.faults_touching_excluded_rounds(
                slicer.fault_index.fault_rounds[representation],
                fault_exclusion_ranges,
                every_fault,
            )
        )
    return excluded_faults


def _owner_by_fault(
    ownership: Sequence,
    representation: fault_model_contracts.FaultRepresentation,
) -> dict[int, int]:
    """Which window owns each fault of one representation."""
    owner_by_fault = {}
    for window_index, owned in enumerate(ownership):
        for fault_index in owned[representation]:
            owner_by_fault[fault_index] = window_index
    return owner_by_fault
