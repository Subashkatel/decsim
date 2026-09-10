"""Places the faults of one window: its rows, its columns, what it owns.

A window's rows are the detectors of its rounds, its columns every
catalog fault that flips one of them and that no earlier window
committed, and it owns the faults touching its commit rounds (Skoric et
al. 2209.08552, section I.B; qLDPC's SlidingWindowDecoder, whose
`d_errors[addressed] = False` removes a committed fault from later
windows). An owned column hands off its whole detector effect, so a
window before or after in time can intersect it with its own rows (Tan
et al. 2209.09219).
"""

import dataclasses
from collections.abc import Container, Sequence
from typing import Optional

import numpy
import scipy.sparse

from decsim.detector_error_model import fault_model_contracts


@dataclasses.dataclass(frozen=True)
class WindowPlacementContext:
    """The window a fault model is being placed into.

    A parameter bundle, not an immutability boundary: the list and dict
    members stay as mutable as the objects the caller owns.
    """

    rows: list[int]
    row_by_detector: dict[int, int]
    observable_count: int
    first_commit_round: int
    last_commit_round: int
    is_last: bool


def parse_window_entry(
    window_entry: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """A plan entry as (first buffer, first commit, last commit, last buffer).

    A three-value entry (first commit, last commit, last buffer) has no
    buffer before its commit rounds.
    """
    if any(bound < 1 for bound in window_entry):
        raise ValueError("window bounds must be positive")
    if len(window_entry) == 4:
        first_buffer, first_commit, last_commit, last_buffer = window_entry
    else:
        first_commit, last_commit, last_buffer = window_entry
        first_buffer = first_commit
    if not first_buffer <= first_commit <= last_commit <= last_buffer:
        raise ValueError("window geometry bounds are not ordered")
    return first_buffer, first_commit, last_commit, last_buffer


def detectors_in_window(
    detectors_by_round: dict[int, list[int]],
    first_buffer_round: int,
    last_buffer_round: int,
) -> list[int]:
    """The window's rows: the detectors of its buffer rounds, sorted."""
    rounds = [
        round_index
        for round_index in detectors_by_round
        if first_buffer_round <= round_index <= last_buffer_round
    ]
    rows = []
    for round_index in rounds:
        rows.extend(detectors_by_round[round_index])
    return sorted(rows)


def checked_fault_exclusion_ranges(
    fault_exclusion_ranges: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """The ranges as a tuple of (first, last) round pairs, none inverted."""
    checked_ranges = []
    for first_excluded, last_excluded in fault_exclusion_ranges:
        if first_excluded > last_excluded:
            raise ValueError(
                f"fault-exclusion range {first_excluded}-{last_excluded} "
                f"is inverted"
            )
        checked_ranges.append((first_excluded, last_excluded))
    return tuple(checked_ranges)


def faults_touching_excluded_rounds(
    fault_rounds: tuple[tuple[int, ...], ...],
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    candidate_faults: list[int],
) -> set[int]:
    """The candidate faults that touch an excluded round.

    Such a fault is owned by nobody: the window that sees it decodes it
    and never commits it, on either ownership path.
    """
    if not fault_exclusion_ranges:
        return set()
    unowned = set()
    for fault_index in candidate_faults:
        if _touches_excluded_round(
            fault_rounds[fault_index], fault_exclusion_ranges
        ):
            unowned.add(fault_index)
    return unowned


def placed_faults_for_window(
    *,
    catalog: fault_model_contracts.FaultCatalog,
    context: WindowPlacementContext,
    fault_rounds: tuple[tuple[int, ...], ...],
    candidate_faults: list[int],
    committed_elsewhere: set[int],
    explicitly_owned_faults: Optional[set[int]],
    explicitly_prior_faults: Optional[Container[int]],
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> fault_model_contracts.PlacedFaultModel:
    """One window's columns from one catalog.

    `fault_rounds` lists each catalog fault's rounds and
    `candidate_faults` the faults that touch this window's rounds; both
    are indexed once by the slicer so a window costs its own size.
    """
    prior_faults = _prior_faults(committed_elsewhere, explicitly_prior_faults)
    columns = _fault_columns_for_window(candidate_faults, prior_faults)
    unowned_faults = faults_touching_excluded_rounds(
        fault_rounds, fault_exclusion_ranges, candidate_faults
    )
    arrays = _build_window_arrays(
        context=context,
        columns=columns,
        detector_sets=catalog.detector_sets,
        observable_sets=catalog.observable_sets,
        fault_rounds=fault_rounds,
        committed_elsewhere=committed_elsewhere,
        unowned_faults=unowned_faults,
        explicitly_owned_faults=explicitly_owned_faults,
    )
    return _placed_model(catalog, columns, arrays)


def local_link_projection(
    graphlike: fault_model_contracts.PlacedFaultModel,
    physical: fault_model_contracts.PlacedFaultModel,
    catalog_link: scipy.sparse.csc_matrix,
) -> scipy.sparse.csc_matrix:
    """The catalog link restricted to this window's columns, checked.

    Raises RuntimeError when the detector rows do not add up, which
    happens when a window keeps a physical fault while an earlier window
    or an ancestor owns a component of it touching this window's rows:
    that is the contract between the slicer's per-representation
    ownership and this projection. Only the detector rows are checked: a
    component carrying a logical flip may lie wholly outside the window,
    and each decoder commits observables from its own placed view.
    """
    graphlike_rows = list(graphlike.source_fault_ids)
    physical_columns = list(physical.source_fault_ids)
    row_slice = catalog_link[graphlike_rows, :]
    local_link = row_slice[:, physical_columns]
    local_link = local_link.tocsc()
    graphlike_check = graphlike.check.astype(numpy.int64)
    detector_identity = graphlike_check @ local_link.astype(numpy.int64)
    detector_identity.data %= 2
    detector_identity.eliminate_zeros()
    physical_check = physical.check.astype(numpy.int64)
    difference = detector_identity != physical_check
    if difference.nnz:
        raise RuntimeError(
            "local physical detector identities do not equal their "
            "graphlike component XOR"
        )
    return local_link


# The check matrix, the observable matrix, the owned mask, the handoff map.
_WindowArrays = tuple[
    scipy.sparse.csc_matrix,
    scipy.sparse.csc_matrix,
    numpy.ndarray,
    dict[int, tuple[int, ...]],
]


def _prior_faults(
    committed_elsewhere: set[int],
    explicitly_prior_faults: Optional[Container[int]],
) -> Container[int]:
    """What earlier windows own: the running set, or the compiled one."""
    if explicitly_prior_faults is None:
        return committed_elsewhere
    return explicitly_prior_faults


def _fault_columns_for_window(
    candidate_faults: list[int], prior_faults: Container[int]
) -> list[int]:
    """The candidate faults that no earlier window has committed."""
    return [
        fault_index
        for fault_index in candidate_faults
        if fault_index not in prior_faults
    ]


def _fault_owned_by_window(
    fault_index: int,
    fault_rounds: tuple,
    committed_elsewhere: set,
    unowned_faults: set,
    explicitly_owned_faults: Optional[set],
    context: WindowPlacementContext,
) -> bool:
    """Whether this window commits the fault.

    Precedence: an excluded fault is never owned; an explicit owner set
    decides next; a fault committed elsewhere is not owned; the terminal
    window owns the rest; any other window owns what touches its commit
    rounds.
    """
    if fault_index in unowned_faults:
        return False
    if explicitly_owned_faults is not None:
        return fault_index in explicitly_owned_faults
    if fault_index in committed_elsewhere:
        return False
    if context.is_last:
        return True
    return any(
        context.first_commit_round <= round_index <= context.last_commit_round
        for round_index in fault_rounds[fault_index]
    )


def _build_window_arrays(
    *,
    context: WindowPlacementContext,
    columns: list,
    detector_sets: tuple,
    observable_sets: tuple,
    fault_rounds: tuple,
    committed_elsewhere: set,
    unowned_faults: set,
    explicitly_owned_faults: Optional[set],
) -> _WindowArrays:
    """The check and observable matrices, the owned mask, the handoff map.

    Entries are collected as (row, column) pairs and assembled once, the
    way qLDPC builds its DetectorErrorModelArrays.
    """
    check = _check_matrix(context, columns, detector_sets)
    observables = _observable_matrix(context, columns, observable_sets)
    owned = numpy.zeros(len(columns), dtype=bool)
    boundary_flips: dict = {}
    for column_index, fault_index in enumerate(columns):
        owns_fault = _fault_owned_by_window(
            fault_index,
            fault_rounds,
            committed_elsewhere,
            unowned_faults,
            explicitly_owned_faults,
            context,
        )
        if not owns_fault:
            continue
        owned[column_index] = True
        if explicitly_owned_faults is None:
            committed_elsewhere.add(fault_index)
        # The whole detector effect travels with the handoff; the receiving
        # window keeps the part that lands on its own rows.
        boundary_flips[column_index] = tuple(detector_sets[fault_index])
    return check, observables, owned, boundary_flips


def _check_matrix(
    context: WindowPlacementContext, columns: list, detector_sets: tuple
) -> scipy.sparse.csc_matrix:
    """Rows by columns, a one where a column flips one of the window's rows."""
    rows: list = []
    entry_columns: list = []
    for column_index, fault_index in enumerate(columns):
        local_rows = _local_rows(
            detector_sets[fault_index], context.row_by_detector
        )
        rows.extend(local_rows)
        column_entries = [column_index] * len(local_rows)
        entry_columns.extend(column_entries)
    ones = numpy.ones(len(rows), dtype=numpy.uint8)
    return scipy.sparse.csc_matrix(
        (ones, (rows, entry_columns)),
        shape=(len(context.rows), len(columns)),
    )


def _local_rows(detectors: tuple, row_by_detector: dict) -> list[int]:
    """The window rows of the detectors that lie inside the window."""
    return [
        row_by_detector[detector_id]
        for detector_id in detectors
        if detector_id in row_by_detector
    ]


def _observable_matrix(
    context: WindowPlacementContext, columns: list, observable_sets: tuple
) -> scipy.sparse.csc_matrix:
    """Observables by columns; every flip counts, local rows or not."""
    rows: list = []
    entry_columns: list = []
    for column_index, fault_index in enumerate(columns):
        for observable_id in observable_sets[fault_index]:
            rows.append(observable_id)
            entry_columns.append(column_index)
    ones = numpy.ones(len(rows), dtype=numpy.uint8)
    return scipy.sparse.csc_matrix(
        (ones, (rows, entry_columns)),
        shape=(context.observable_count, len(columns)),
    )


def _touches_excluded_round(
    rounds: tuple, fault_exclusion_ranges: tuple
) -> bool:
    for first_excluded, last_excluded in fault_exclusion_ranges:
        if any(
            first_excluded <= round_index <= last_excluded
            for round_index in rounds
        ):
            return True
    return False


def _placed_model(
    catalog: fault_model_contracts.FaultCatalog,
    columns: list,
    arrays: _WindowArrays,
) -> fault_model_contracts.PlacedFaultModel:
    check, observables, owned, boundary_flips = arrays
    column_priors = [catalog.priors[fault_index] for fault_index in columns]
    priors = numpy.asarray(column_priors, dtype=float)
    return fault_model_contracts.PlacedFaultModel(
        representation=catalog.representation,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(columns),
        boundary_flips=boundary_flips,
    )
