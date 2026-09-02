"""The entry points that turn a window plan into window models.

build_window_error_models slices a whole plan: it checks the plan against
its protocol, checks that the commit rounds run without gap or overlap,
compiles fault ownership from the dependency graph when the plan has one,
slices every window, and checks that the owners of a full-operation plan
partition the catalog. The two single-window builders serve the runtime
paths that decode one window on its own with faults it may see but must
not commit.

Nothing inside the package imports this module.
"""

from typing import Optional

import decsim.message as message
from decsim.detector_error_model import (
    fault_model_contracts,
    window_ownership_dag,
    window_placement,
    window_protocol_policy,
    window_slicer,
)


def build_window_error_models(
    circuit,
    plan: list,
    *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
    fault_exclusion_ranges: tuple,
    dependency_edges: Optional[tuple] = None,
    closed_temporal_boundary_windows: tuple[int, ...] = (),
    window_protocol: message.WindowProtocol = message.WindowProtocol.GENERIC,
) -> list:
    """One window model per plan entry, in plan order.

    With `dependency_edges`, ownership is compiled from the dependency
    graph and a window leaves out what its ancestors own; without them
    ownership advances in list order. A plan may cover any contiguous run
    of the operation; only a window whose commit rounds reach
    `round_count` is terminal, and a fault outside the plan stays unowned.
    """
    entries = _checked_entries(
        plan,
        round_count,
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
    circuit,
    window_entry: tuple,
    *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
    exclude_faults_touching: Optional[tuple] = None,
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
    circuit,
    window_entry: tuple,
    *,
    round_count: int,
    detector_rounds: Optional[dict] = None,
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
    fault_exclusion_ranges: tuple,
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


def _checked_entries(
    plan: list,
    round_count: int,
    window_protocol: message.WindowProtocol,
    dependency_edges: Optional[tuple],
    closed_temporal_boundary_windows: tuple,
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
) -> tuple:
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
    _check_commit_rounds_are_contiguous(entries, round_count)
    return entries


def _slice_checked_plan(
    slicer: window_slicer.WindowSlicer,
    entries: tuple,
    round_count: int,
    fault_exclusion_ranges: tuple,
    dependency_edges: Optional[tuple],
    closed_temporal_boundary_windows: tuple,
) -> list:
    """Every window of the plan, with its boundaries and ownership checked.

    A window listed in `closed_temporal_boundary_windows` is refused if
    slicing cut a fault of the circuit at its edge.
    """
    ownership, prior_faults = _compiled_ownership(
        slicer, entries, dependency_edges, round_count
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
    if ownership is not None and not fault_exclusion_ranges:
        _check_full_plan_ownership(slicer, models, entries, round_count)
    return models


def _check_commit_rounds_are_contiguous(
    entries: tuple, round_count: int
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
    entries: tuple,
    dependency_edges: Optional[tuple],
    round_count: int,
) -> tuple[Optional[tuple], Optional[tuple]]:
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
        slicer, entries, depths, round_count=round_count
    )
    prior_faults = window_ownership_dag.explicit_prior_faults(
        ownership, ancestors
    )
    return ownership, prior_faults


def _slice_plan(
    slicer: window_slicer.WindowSlicer,
    entries: tuple,
    round_count: int,
    fault_exclusion_ranges: tuple,
    ownership: Optional[tuple],
    prior_faults: Optional[tuple],
) -> list:
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


def _entry_of(per_window: Optional[tuple], window_index: int):
    if per_window is None:
        return None
    return per_window[window_index]


def _check_full_plan_ownership(
    slicer: window_slicer.WindowSlicer,
    models: list,
    entries: tuple,
    round_count: int,
) -> None:
    """Every catalog fault of a full plan must be owned by exactly one window.

    A plan that does not cover the whole operation leaves faults unowned
    by design. Compiled ownership gives each fault one owner by
    construction, so a fault no window placed would be a slicer bug; that
    raises RuntimeError.
    """
    covers_full_operation = entries[0][1] == 1 and entries[-1][2] == round_count
    if not covers_full_operation:
        return
    for representation, catalog in slicer.catalogs.items():
        owned = _owned_fault_ids(models, representation)
        if owned != set(range(len(catalog.detector_sets))):
            raise RuntimeError(
                f"{representation.value} dependency ownership does not "
                "partition the full fault catalog"
            )


def _owned_fault_ids(models: list, representation) -> set:
    """The catalog faults the models own, in one representation."""
    owned = set()
    for model in models:
        faults = model.require_faults(representation)
        pairs = zip(faults.source_fault_ids, faults.owned)
        owned.update(fault_id for fault_id, is_owned in pairs if is_owned)
    return owned


def _build_single_window_error_model(
    circuit,
    window_entry: tuple,
    *,
    round_count: int,
    detector_rounds: Optional[dict],
    fault_model_requirement: (
        fault_model_contracts.DecoderFaultModelRequirement
    ),
    fault_exclusion_ranges: tuple,
) -> fault_model_contracts.WindowErrorModel:
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
