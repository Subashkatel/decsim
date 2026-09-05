"""The entry points that turn a window plan into window models.

build_window_error_models checks a plan against its protocol and for
contiguous commit rounds, compiles fault ownership from the dependency
graph when the plan has one (ownership advances in plan order otherwise),
and slices every window (Skoric et al. 2209.08552, section I.B and the
last paragraph of section III; qLDPC's SlidingWindowDecoder). The two
single-window builders serve the strong re-decode of one window
(decsim/escalation/strong_window_shapes), whose exclusion ranges are decsim's
own device: the faults the weak decoder already committed stay
uncommitted. Nothing inside the package imports this module.
"""

from collections.abc import Container, Mapping, Sequence
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
        fault_exclusion_ranges
    )
    entries = _checked_plan(
        plan,
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
    never terminal: it commits only what touches its commit rounds.
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
    """One window on its own, with several round ranges it may not commit."""
    return _build_single_window_error_model(
        circuit,
        window_entry,
        round_count=round_count,
        detector_rounds=detector_rounds,
        fault_model_requirement=fault_model_requirement,
        fault_exclusion_ranges=fault_exclusion_ranges,
    )


# Per-window fault tables by representation; None when ownership advances
# in plan order. _OwnedFaultsPerWindow holds the faults each window owns
# and _PriorFaultsPerWindow the faults each window's ancestors own;
# _FaultsPerWindow is what _entry_of reads, and both others satisfy it.
_OwnedFaultsPerWindow = Optional[
    tuple[dict[fault_model_contracts.FaultRepresentation, set[int]], ...]
]
_PriorFaultsPerWindow = Optional[
    tuple[
        dict[fault_model_contracts.FaultRepresentation, Container[int]],
        ...,
    ]
]
_FaultsPerWindow = Optional[
    tuple[
        Mapping[fault_model_contracts.FaultRepresentation, Container[int]],
        ...,
    ]
]


def _checked_plan(
    plan: list[tuple[int, ...]],
    window_protocol: message.WindowProtocol,
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_temporal_boundary_windows: tuple[int, ...],
    fault_model_requirement: fault_model_contracts.DecoderFaultModelRequirement,
) -> tuple[tuple[int, int, int, int], ...]:
    """The plan's entries, checked against the protocol and for contiguity."""
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
    window_protocol_policy.validate_closed_windows_are_dependency_destinations(
        dependency_edges, closed_temporal_boundary_windows
    )
    _check_commit_rounds_are_contiguous(entries)
    return entries


def _slice_checked_plan(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    round_count: int,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    closed_temporal_boundary_windows: tuple[int, ...],
) -> list[fault_model_contracts.WindowErrorModel]:
    """Every window of the plan; a closed window may not cut a fault."""
    ownership, prior_faults = _compiled_ownership(
        slicer, entries, dependency_edges, fault_exclusion_ranges
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
        slicer, models, closed_temporal_boundary_windows
    )
    return models


def _check_commit_rounds_are_contiguous(
    entries: tuple[tuple[int, int, int, int], ...],
) -> None:
    next_commit_round = entries[0][1]
    for _, first_commit_round, last_commit_round, _ in entries:
        if first_commit_round != next_commit_round:
            raise ValueError(
                "window commit regions must be contiguous in plan order "
                "without gaps or overlaps"
            )
        next_commit_round = last_commit_round + 1


def _compiled_ownership(
    slicer: window_slicer.WindowSlicer,
    entries: tuple[tuple[int, int, int, int], ...],
    dependency_edges: Optional[tuple[tuple[int, int], ...]],
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> tuple[_OwnedFaultsPerWindow, _PriorFaultsPerWindow]:
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
        slicer, entries, depths, fault_exclusion_ranges
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
    ownership: _OwnedFaultsPerWindow,
    prior_faults: _PriorFaultsPerWindow,
) -> list[fault_model_contracts.WindowErrorModel]:
    models = []
    for window_index, window_entry in enumerate(entries):
        # Contiguity makes the window whose commit rounds reach the last
        # round the last entry; only it is terminal.
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
    per_window: _FaultsPerWindow,
    window_index: int,
) -> Optional[
    Mapping[fault_model_contracts.FaultRepresentation, Container[int]]
]:
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
        fault_exclusion_ranges
    )
    bounds = window_placement.parse_window_entry(window_entry)
    slicer = window_slicer.WindowSlicer(
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
