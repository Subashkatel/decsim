"""The entry points that turn a window plan into window models.

build_window_error_models slices a whole plan: it checks the plan against
its protocol, checks that the commit rounds run without gap or overlap,
compiles fault ownership from the dependency graph when the plan has one
(ownership advances in plan order otherwise), and slices every window. A
plan may cover part of the operation: a fault no window's rows reach
stays unowned, while the terminal window owns every uncommitted fault it
sees, front buffer rounds included, whether ownership advances in plan
order or is compiled from the dependency graph (Skoric et al.
2209.08552, section I.B; qLDPC's SlidingWindowDecoder, whose last window
commits all it holds). The two single-window builders serve the runtime
paths that decode one window on its own with faults it may see but must
not commit; a window built alone is never terminal.

The exclusion ranges arrive as any sequence of (first, last) round pairs
and are checked once at entry. Ownership advances per representation, so
two linked plans are refused at entry: an exclusion range on a plan of
more than one window, where the machine cannot keep a physical fault
uncommitted past its own component, and dependency edges on a plan whose
first commit round is after round 1, where the terminal window would own
a graphlike component of a physical fault that a window depending on it
owns.

Nothing inside the package imports this module.
"""

from collections.abc import Container, Sequence
from typing import Optional

import stim

import decsim.message as message
from decsim.detector_error_model import (
    fault_model_contracts,
    window_ownership_dag,
    window_placement,
    window_protocol_policy,
    window_slicer,
)


def build_window_error_models(
    circuit: stim.Circuit,
    plan: list[tuple[int, ...]],
    *,
    round_count: int,
    detector_rounds: Optional[dict[int, int]] = None,
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
    fault_exclusion_ranges: Sequence[Sequence[int]],
    dependency_edges: Optional[tuple[tuple[int, int], ...]] = None,
    closed_temporal_boundary_windows: tuple[int, ...] = (),
    window_protocol: message.WindowProtocol = message.WindowProtocol.GENERIC,
) -> list[fault_model_contracts.WindowErrorModel]:
    """One window model per plan entry, in plan order."""
    exclusion_ranges = window_placement.checked_fault_exclusion_ranges(
        fault_exclusion_ranges, round_count
    )
    entries = _checked_plan(
        plan,
        round_count,
        exclusion_ranges,
        window_protocol,
        dependency_edges,
        closed_temporal_boundary_windows,
        fault_model_requirement,
    )
    slicer = _new_slicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    return _slice_checked_plan(
        slicer,
        entries,
        round_count,
        exclusion_ranges,
        dependency_edges,
        closed_temporal_boundary_windows,
    )


def build_single_window_error_model(
    circuit: stim.Circuit,
    window_entry: tuple[int, ...],
    *,
    round_count: int,
    detector_rounds: Optional[dict[int, int]] = None,
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
    exclude_faults_touching: Optional[Sequence[int]] = None,
) -> fault_model_contracts.WindowErrorModel:
    """One window on its own, with one inclusive round range it may not commit.

    A fault touching `exclude_faults_touching` stays in the window to
    explain the syndrome but is never owned by it. A window built alone is
    never terminal: it serves the strong re-decode of one window
    (decsim/decoders/strong_escalation), which commits only what touches
    its commit rounds.
    """
    fault_exclusion_ranges = ()
    if exclude_faults_touching is not None:
        fault_exclusion_ranges = (exclude_faults_touching,)
    return _build_single_window_error_model(
        circuit,
        window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def build_single_window_error_model_with_exclusions(
    circuit: stim.Circuit,
    window_entry: tuple[int, ...],
    *,
    round_count: int,
    detector_rounds: Optional[dict[int, int]] = None,
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
    fault_exclusion_ranges: Sequence[Sequence[int]],
) -> fault_model_contracts.WindowErrorModel:
    """One window on its own, with several round ranges it may not commit.

    Never terminal, like build_single_window_error_model.
    """
    return _build_single_window_error_model(
        circuit,
        window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


def _new_slicer(
    circuit: stim.Circuit,
    *,
    round_count: int,
    detector_rounds: Optional[dict[int, int]],
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
) -> window_slicer.WindowSlicer:
    return window_slicer.WindowSlicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )


def _checked_plan(
    plan: list[tuple[int, ...]],
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    window_protocol: message.WindowProtocol,
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_temporal_boundary_windows: tuple[int, ...],
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
) -> tuple[tuple[int, int, int, int], ...]:
    """The plan's entries, checked against the protocol and for contiguity."""
    if not plan:
        raise ValueError("a window plan must hold at least one window")
    entries = tuple(
        window_placement.parse_window_entry(window_entry)
        for window_entry in plan
    )
    _check_the_plan_fits_a_linked_requirement(
        entries,
        fault_exclusion_ranges,
        dependency_edges,
        fault_model_requirement,
    )
    window_protocol_policy.validate_window_protocol(
        entries,
        window_protocol,
        dependency_edges,
        closed_temporal_boundary_windows,
        fault_model_requirement,
    )
    _check_commit_rounds_are_contiguous(entries, round_count)
    _check_windows_end_inside_the_operation(entries, round_count)
    return entries


def _slice_checked_plan(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_temporal_boundary_windows: tuple[int, ...],
) -> list[fault_model_contracts.WindowErrorModel]:
    """Every window of the plan, with its boundaries and ownership checked.

    A window listed in `closed_temporal_boundary_windows` is refused if
    slicing cut a fault of the circuit at its edge.
    """
    ownership, prior_faults = _compiled_ownership(
        slicer, entries, dependency_edges, round_count, fault_exclusion_ranges
    )
    models = _slice_plan(
        slicer,
        entries,
        round_count,
        fault_exclusion_ranges,
        ownership,
        prior_faults,
    )
    window_protocol_policy.validate_closed_temporal_boundary_windows(
        slicer, models, dependency_edges, closed_temporal_boundary_windows
    )
    return models


def _check_the_plan_fits_a_linked_requirement(
    entries: tuple[tuple[int, int, int, int], ...],
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
) -> None:
    """A linked plan keeps every physical fault beside its components.

    Ownership advances per representation. With an exclusion range on
    several windows, an earlier window would commit a graphlike component
    while the range kept its physical parent uncommitted. With dependency
    edges and a first commit round after round 1, the terminal window
    would own a component that no commit round reaches while the physical
    fault belongs to a window depending on it, and that window would drop
    the component and keep the fault.
    """
    if not fault_model_requirement.require_physical_to_graphlike_link:
        return
    if fault_exclusion_ranges and len(entries) > 1:
        raise ValueError(
            "a linked fault model requirement with fault_exclusion_ranges is "
            "refused for a plan of more than one window, because the "
            "machine cannot keep a physical fault uncommitted past its own "
            "component"
        )
    first_commit_rounds = [entry[1] for entry in entries]
    first_commit_round = min(first_commit_rounds)
    if dependency_edges and first_commit_round > 1:
        raise ValueError(
            "a linked fault model requirement with dependency edges is "
            "refused for a plan whose first commit round is after round 1, "
            "because the terminal window would own a graphlike component "
            "of a physical fault that a window depending on it owns"
        )


def _check_commit_rounds_are_contiguous(
    entries: tuple[tuple[int, int, int, int], ...], round_count: int
) -> None:
    next_commit_round = entries[0][1]
    for _, first_commit_round, last_commit_round, _ in entries:
        if first_commit_round != next_commit_round:
            raise ValueError(
                "window commit regions must be contiguous in plan order "
                "without gaps or overlaps"
            )
        if last_commit_round > round_count:
            raise ValueError("window commit region exceeds round_count")
        next_commit_round = last_commit_round + 1


def _check_windows_end_inside_the_operation(
    entries: tuple[tuple[int, int, int, int], ...], round_count: int
) -> None:
    """The runtime clamps a buffer at the last round before calling.

    A window built alone runs only this check, and its commit rounds end
    inside its buffer, so a commit past round_count is caught here too.
    """
    for _, _, _, last_buffer_round in entries:
        if last_buffer_round > round_count:
            raise ValueError(
                "window commit or buffer region exceeds round_count"
            )


def _compiled_ownership(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> tuple[
    Optional[
        tuple[dict[fault_model_contracts.FaultRepresentation, set[int]], ...]
    ],
    Optional[
        tuple[
            dict[fault_model_contracts.FaultRepresentation, Container[int]],
            ...,
        ]
    ],
]:
    """Owner sets and prior sets per window, or (None, None) without edges."""
    if dependency_edges is None:
        return None, None
    depths = window_ownership_dag.dependency_depths(
        len(entries), dependency_edges
    )
    ancestors = window_ownership_dag.dependency_ancestors(
        len(entries), dependency_edges, depths
    )
    ownership = window_ownership_dag.explicit_fault_ownership(
        slicer, entries, depths, round_count, fault_exclusion_ranges
    )
    prior_faults = window_ownership_dag.explicit_prior_faults(
        ownership, ancestors
    )
    return ownership, prior_faults


def _slice_plan(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    ownership: Optional[
        tuple[dict[fault_model_contracts.FaultRepresentation, set[int]], ...]
    ],
    prior_faults: Optional[
        tuple[
            dict[fault_model_contracts.FaultRepresentation, Container[int]],
            ...,
        ]
    ],
) -> list[fault_model_contracts.WindowErrorModel]:
    """Every window of the plan, in plan order."""
    models = []
    for window_index, window_entry in enumerate(entries):
        # The contiguity check makes the window whose commit rounds reach
        # the last round the last entry; only it is terminal.
        is_last = window_entry[2] == round_count
        owned = _entry_of(ownership, window_index)
        prior = _entry_of(prior_faults, window_index)
        model = slicer.slice_window(
            *window_entry,
            is_last=is_last,
            fault_exclusion_ranges=fault_exclusion_ranges,
            explicitly_owned_faults=owned,
            explicitly_prior_faults=prior,
        )
        models.append(model)
    return models


def _entry_of(
    per_window: Optional[
        tuple[
            dict[fault_model_contracts.FaultRepresentation, Container[int]],
            ...,
        ]
    ],
    window_index: int,
) -> Optional[dict[fault_model_contracts.FaultRepresentation, Container[int]]]:
    if per_window is None:
        return None
    return per_window[window_index]


def _build_single_window_error_model(
    circuit: stim.Circuit,
    window_entry: tuple[int, ...],
    *,
    round_count: int,
    detector_rounds: Optional[dict[int, int]],
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
    fault_exclusion_ranges: Sequence[Sequence[int]],
) -> fault_model_contracts.WindowErrorModel:
    exclusion_ranges = window_placement.checked_fault_exclusion_ranges(
        fault_exclusion_ranges, round_count
    )
    bounds = window_placement.parse_window_entry(window_entry)
    _check_windows_end_inside_the_operation((bounds,), round_count)
    slicer = _new_slicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    return slicer.slice_window(
        *bounds,
        is_last=False,
        fault_exclusion_ranges=exclusion_ranges,
    )
