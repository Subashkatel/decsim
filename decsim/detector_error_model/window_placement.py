"""Per-window geometry, ownership policy and local array placement.

Parses window plan entries, selects the detector rows and fault columns of one
window, decides which window owns each fault and which flips are handed off
forward, builds the aligned decoder arrays, and projects the window-local link
between the two fault domains.

Package-internal seams: _parse_window_entry is imported by
window_model_builders; WindowPlacementContext, _detectors_in_window,
_validate_fault_exclusion_ranges, _placed_faults_for_window and
_local_physical_to_graphlike_detector_projection are imported by window_slicer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .fault_model_contracts import (
    FaultRepresentation,
    PlacedFaultModel,
    _FaultCatalog,
)
from .fault_identity_validation import (
    validate_graphlike_matrices,
    validate_placed_fault_matrices,
)


@dataclass(frozen=True)
class WindowPlacementContext:
    """The one window that a fault model is being placed into.

    A parameter bundle, not an immutability boundary: frozen=True only prevents
    rebinding the eight fields, and the list, dict and set members stay exactly
    as mutable as the objects the caller already owns.
    """

    rows: list[int]
    row_index: dict[int, int]
    lead_rows: set[int]
    round_of: dict[int, int]
    n_obs: int
    commit_lo: int
    commit_hi: int
    is_last: bool


def _parse_window_entry(window_entry: tuple) -> tuple[int, int, int, int]:
    """Normalize and validate a 3-value or 4-value window plan entry."""
    if any(type(bound) is not int for bound in window_entry):
        raise TypeError("window bounds must be built-in ints")
    if any(bound < 1 for bound in window_entry):
        raise ValueError("window bounds must be positive")
    if len(window_entry) == 4:
        buffer_lo, commit_lo, commit_hi, buffer_hi = window_entry
    else:
        commit_lo, commit_hi, buffer_hi = window_entry
        buffer_lo = commit_lo
    if not buffer_lo <= commit_lo <= commit_hi <= buffer_hi:
        raise ValueError("window geometry bounds are not ordered")
    return buffer_lo, commit_lo, commit_hi, buffer_hi


def _detectors_in_window(round_of: dict, buffer_lo: int, buffer_hi: int,
                         *, is_last: bool) -> list:
    """Choose the detector rows for this window."""
    if is_last:
        return sorted(detector_id
                      for detector_id, round_index in round_of.items()
                      if round_index >= buffer_lo)

    return sorted(detector_id
                  for detector_id, round_index in round_of.items()
                  if buffer_lo <= round_index <= buffer_hi)


def _fault_columns_for_window(
    det_sets: tuple,
    row_index: dict,
    lead_rows: set,
    committed_elsewhere: set,
    *,
    include_committed_leading: bool,
) -> list:
    """Choose candidate columns, excluding causally prior committed faults."""
    columns: list = []
    for fault_index, detectors in enumerate(det_sets):
        touches_window = any(detector_id in row_index for detector_id in detectors)
        if not touches_window:
            continue

        if fault_index not in committed_elsewhere:
            columns.append(fault_index)
            continue

        touches_leading_buffer = any(detector_id in lead_rows
                                     for detector_id in detectors)
        if include_committed_leading and touches_leading_buffer:
            columns.append(fault_index)
    return columns


def _fault_owned_by_window(
    fault_index: int,
    fault_rounds: tuple,
    committed_elsewhere: set,
    unowned_faults: set,
    explicitly_owned_faults: Optional[set],
    context: WindowPlacementContext,
) -> bool:
    """True when this window is responsible for committing the fault."""
    if fault_index in unowned_faults:
        return False
    if explicitly_owned_faults is not None:
        return fault_index in explicitly_owned_faults
    if fault_index in committed_elsewhere:
        return False
    if context.is_last:
        return True
    return any(context.commit_lo <= round_index <= context.commit_hi
               for round_index in fault_rounds[fault_index])


def _fill_detector_and_observable_columns(
    check, obs, *, column_index: int, fault_index: int, det_sets: tuple,
    obs_sets: tuple, context: WindowPlacementContext,
) -> None:
    """Fill the detector and observable entries for one fault column."""
    for detector_id in det_sets[fault_index]:
        if detector_id in context.row_index:
            check[context.row_index[detector_id], column_index] = 1

    for observable_id in obs_sets[fault_index]:
        obs[observable_id, column_index] = 1


def _future_flips_after_commit(det_sets: tuple, fault_index: int,
                               context: WindowPlacementContext) -> tuple:
    """Return detector flips that must be handed to a later window."""
    if context.is_last:
        return ()
    return tuple(detector_id
                 for detector_id in det_sets[fault_index]
                 if context.round_of[detector_id] > context.commit_hi)


def _build_window_arrays(*, context: WindowPlacementContext, columns: list,
                         det_sets: tuple, obs_sets: tuple,
                         fault_rounds: tuple,
                         committed_elsewhere: set, unowned_faults: set,
                         explicitly_owned_faults: Optional[set]) -> tuple:
    """Build check, observable, ownership, and residual-defect arrays."""
    import numpy as np

    check = np.zeros((len(context.rows), len(columns)), dtype=np.uint8)
    obs = np.zeros((context.n_obs, len(columns)), dtype=np.uint8)
    owned = np.zeros(len(columns), dtype=bool)
    future_flips: dict = {}
    boundary_flips: dict = {}

    for column_index, fault_index in enumerate(columns):
        _fill_detector_and_observable_columns(
            check, obs, column_index=column_index, fault_index=fault_index,
            det_sets=det_sets, obs_sets=obs_sets, context=context)

        owns_fault = _fault_owned_by_window(
            fault_index, fault_rounds, committed_elsewhere, unowned_faults,
            explicitly_owned_faults, context)
        if not owns_fault:
            continue

        owned[column_index] = True
        if explicitly_owned_faults is None:
            committed_elsewhere.add(fault_index)
        beyond_commit = _future_flips_after_commit(
            det_sets, fault_index, context)
        if beyond_commit:
            future_flips[column_index] = beyond_commit
        # Keep the complete global detector effect. The destination intersects
        # it with its own rows, so the same correction can travel left or right.
        detector_effect = tuple(det_sets[fault_index])
        if detector_effect:
            boundary_flips[column_index] = detector_effect

    return check, obs, owned, future_flips, boundary_flips


def _validate_fault_exclusion_ranges(fault_exclusion_ranges: tuple) -> None:
    """Validate explicit inclusive round ranges without changing ownership."""
    for exclusion in fault_exclusion_ranges:
        exclude_lo, exclude_hi = exclusion
        if not all(type(endpoint) is int
                   for endpoint in (exclude_lo, exclude_hi)):
            raise TypeError(
                "each fault-exclusion range must use built-in integer "
                f"(lo, hi) pair, got {exclusion!r}")
        if exclude_lo > exclude_hi:
            raise ValueError(
                f"fault-exclusion range {exclude_lo}-{exclude_hi} "
                f"is inverted")


def _unowned_faults(
    fault_rounds: tuple[tuple[int, ...], ...],
    fault_exclusion_ranges: tuple,
) -> set[int]:
    """Return source columns prevented from being committed by this slice."""
    return {
        fault_index
        for fault_index, rounds in enumerate(fault_rounds)
        if any(
            exclude_lo <= round_index <= exclude_hi
            for exclude_lo, exclude_hi in fault_exclusion_ranges
            for round_index in rounds
        )
    }


def _placed_faults_for_window(
    *,
    catalog: _FaultCatalog,
    context: WindowPlacementContext,
    committed_elsewhere: set[int],
    explicitly_owned_faults: Optional[set[int]],
    explicitly_prior_faults: Optional[set[int]],
    fault_exclusion_ranges: tuple,
) -> PlacedFaultModel:
    """Build one window's local matrix from one global fault catalog."""
    import numpy as np

    detector_sets = catalog.detector_sets
    observable_sets = catalog.observable_sets
    fault_rounds = tuple(
        tuple(context.round_of[detector_id] for detector_id in detectors)
        for detectors in catalog.detector_sets
    )
    columns = _fault_columns_for_window(
        detector_sets,
        context.row_index,
        context.lead_rows,
        (
            committed_elsewhere
            if explicitly_prior_faults is None
            else explicitly_prior_faults
        ),
        include_committed_leading=(explicitly_prior_faults is None),
    )
    check, observables, owned, future_flips, boundary_flips = (
        _build_window_arrays(
            context=context,
            columns=columns,
            det_sets=detector_sets,
            obs_sets=observable_sets,
            fault_rounds=fault_rounds,
            committed_elsewhere=committed_elsewhere,
            unowned_faults=_unowned_faults(
                fault_rounds,
                fault_exclusion_ranges,
            ),
            explicitly_owned_faults=explicitly_owned_faults,
        )
    )
    placed = PlacedFaultModel(
        representation=catalog.representation,
        check=check,
        priors=np.asarray(
            [catalog.priors[fault_index] for fault_index in columns],
            dtype=float,
        ),
        observables=observables,
        owned=owned,
        future_flips=future_flips,
        source_fault_ids=tuple(columns),
        boundary_flips=boundary_flips,
    )
    if catalog.representation is FaultRepresentation.GRAPHLIKE:
        validate_graphlike_matrices(
            placed.check,
            placed.observables,
            location="placed graphlike fault model",
        )
    else:
        validate_placed_fault_matrices(
            placed.check,
            placed.observables,
            location="placed physical fault model",
        )
    return placed


def _local_physical_to_graphlike_detector_projection(
    graphlike: PlacedFaultModel,
    physical: PlacedFaultModel,
    catalog_link,
):
    """Slice and exactly validate the link between the two local views.

    Window-local half of the linked fault domain; the global half is
    stim_dem_catalog._prepare_linked_fault_catalogs.
    """
    import numpy as np

    local_link = np.asarray(catalog_link, dtype=np.uint8)[np.ix_(
        graphlike.source_fault_ids,
        physical.source_fault_ids,
    )]
    detector_identity = (
        np.asarray(graphlike.check, dtype=np.uint64)
        @ local_link.astype(np.uint64)
    ) % 2
    if not np.array_equal(detector_identity, physical.check):
        raise ValueError(
            "local physical detector identities do not equal their "
            "graphlike component XOR"
        )
    # A physical fault can decompose into a component whose detectors lie
    # wholly beyond this window while that component carries a logical tag.
    # The catalog-level check above preserves the complete observable identity;
    # the local link is deliberately only the detector-row projection consumed
    # by belief propagation.  Each decoder commits observables from its own
    # explicit placed view, never through this projected link.
    return local_link
