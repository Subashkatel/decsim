"""The entry points that turn a window plan into window models.

build_window_error_models slices a whole plan: it checks the plan against
its protocol, checks that the commit rounds run without gap or overlap,
compiles fault ownership from the dependency graph when the plan has one
(ownership advances in plan order otherwise), and slices every window. A
plan may cover part of the operation; a fault outside it stays unowned.
The two single-window builders serve the runtime paths that decode one
window on its own with faults it may see but must not commit.

Nothing inside the package imports this module.
"""

from collections.abc import Container
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
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]] = None,
    closed_temporal_boundary_windows: tuple[int, ...] = (),
    window_protocol: message.WindowProtocol = message.WindowProtocol.GENERIC,
) -> list[fault_model_contracts.WindowErrorModel]:
    """One window model per plan entry, in plan order.

    Only a window whose commit rounds reach `round_count` is terminal.
    """
    entries = _checked_plan(
        plan,
        round_count,
        fault_exclusion_ranges,
        window_protocol,
        dependency_edges,
        closed_temporal_boundary_windows,
        fault_model_requirement,
    )
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    return _slice_checked_plan(
        slicer,
        entries,
        round_count,
        fault_exclusion_ranges,
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
    exclude_faults_touching: Optional[tuple[int, int]] = None,
) -> fault_model_contracts.WindowErrorModel:
    """One window on its own, with one inclusive round range it may not commit.

    A fault touching `exclude_faults_touching` stays in the window to
    explain the syndrome but is never owned by it.
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
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> fault_model_contracts.WindowErrorModel:
    """One window on its own, with several round ranges it may not commit."""
    return _build_single_window_error_model(
        circuit,
        window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
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
    """The plan's entries, checked against the protocol and for contiguity.

    The exclusion ranges are checked here too, once for the whole plan.
    """
    window_placement.validate_fault_exclusion_ranges(fault_exclusion_ranges)
    if not plan:
        raise ValueError("a window plan must hold at least one window")
    entries = tuple(
        window_placement.parse_window_entry(window_entry)
        for window_entry in plan
    )
    window_protocol_policy.validate_window_protocol(
        entries,
        window_protocol,
        dependency_edges,
        closed_temporal_boundary_windows,
        fault_model_requirement,
    )
    _check_commit_rounds_are_contiguous(entries, round_count)
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
        slicer, entries, dependency_edges
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


def _compiled_ownership(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
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
        slicer, entries, depths
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
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> fault_model_contracts.WindowErrorModel:
    window_placement.validate_fault_exclusion_ranges(fault_exclusion_ranges)
    slicer = window_slicer.WindowSlicer(
        circuit,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
    )
    bounds = window_placement.parse_window_entry(window_entry)
    return slicer.slice_window(
        *bounds,
        is_last=False,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )
