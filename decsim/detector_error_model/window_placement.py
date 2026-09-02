"""Places the faults of one window: its rows, its columns, what it owns.

A window covers a run of rounds. Its rows are the detectors of those
rounds. Its columns are every catalog fault that flips one of its rows
and that no earlier window has already committed; the decoder needs the
uncommitted faults reaching in from outside to explain what it sees. That
is the overlapping recovery of Dennis et al. as Skoric et al. (2209.08552,
section I.B) and qLDPC's SlidingWindowDecoder apply it.

A window owns, and later commits, the faults that touch its commit
rounds; the buffer rounds after the commit rounds are decoded but not
committed, so a chain that crosses the commit boundary is cut there and
its far end becomes an artificial defect for the next window (Skoric et
al., Fig. 2). A fault a window owns is never offered to a later window
again (qLDPC's `d_errors[addressed] = False`). The last window of an
operation owns everything it sees, because nothing comes after it. When
a plan compiled from a dependency graph supplies the owner sets
explicitly, they replace this incremental rule.

Every owned column keeps its whole detector effect in boundary_flips, so
the window that receives the handoff can intersect it with its own rows
whether it comes before or after in time (Tan et al. 2209.09219, whose
seam windows receive flips from both neighbours).
"""

import dataclasses
from collections.abc import Container
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
    if len(window_entry) not in (3, 4):
        raise ValueError(
            f"a window entry has three or four bounds, got {len(window_entry)}"
        )
    for bound in window_entry:
        if type(bound) is not int:
            raise ValueError("window bounds must be built-in ints")
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
    *,
    is_last: bool,
) -> list[int]:
    """The window's rows: the detectors of its buffer rounds, sorted.

    The last window takes every round from its start, so a plan that ends
    early still sees the whole tail.
    """
    if is_last:
        rounds = [
            round_index
            for round_index in detectors_by_round
            if round_index >= first_buffer_round
        ]
    else:
        rounds = [
            round_index
            for round_index in detectors_by_round
            if first_buffer_round <= round_index <= last_buffer_round
        ]
    rows = []
    for round_index in rounds:
        rows.extend(detectors_by_round[round_index])
    return sorted(rows)


def validate_fault_exclusion_ranges(
    fault_exclusion_ranges: tuple[tuple[int, int], ...], round_count: int
) -> None:
    """Refuse a range that is not an ordered pair of ints inside the operation.

    A round outside 1..round_count holds no detector, so a range reaching
    there is a caller's mistake rather than an empty exclusion.
    """
    if not isinstance(fault_exclusion_ranges, tuple):
        raise ValueError(
            "fault_exclusion_ranges must be a tuple of ranges, got "
            f"{fault_exclusion_ranges!r}"
        )
    for exclusion in fault_exclusion_ranges:
        _check_range_is_a_pair_of_ints(exclusion)
        first_excluded, last_excluded = exclusion
        if first_excluded > last_excluded:
            raise ValueError(
                f"fault-exclusion range {first_excluded}-{last_excluded} "
                f"is inverted"
            )
        if first_excluded < 1 or last_excluded > round_count:
            raise ValueError(
                f"fault-exclusion range {first_excluded}-{last_excluded} "
                f"lies outside rounds 1..{round_count}"
            )


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
    unowned_faults = _unowned_faults(
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

    Only the detector rows are checked. They fail to add up when an
    earlier window committed a component while an exclusion range kept
    its physical parent uncommitted: the parent then reaches this window
    without the component, and no consistent link exists, so the slice is
    refused. That is the contract between the slicer's per-representation
    ownership and this projection; the builders refuse the input that
    reaches it, a linked requirement with an exclusion range on a plan of
    more than one window. A physical fault can have a component whose
    detectors all lie outside this window while that component carries a
    logical flip, so the observable rows of a window need not add up;
    each decoder commits observables from its own placed view, never
    through this link.
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


def _check_range_is_a_pair_of_ints(exclusion: object) -> None:
    if not isinstance(exclusion, tuple):
        _refuse_exclusion_range(exclusion)
    if len(exclusion) != 2:
        _refuse_exclusion_range(exclusion)
    for endpoint in exclusion:
        if type(endpoint) is not int:
            _refuse_exclusion_range(exclusion)


def _refuse_exclusion_range(exclusion: object) -> None:
    raise ValueError(
        "each fault-exclusion range must be a pair of built-in integers, "
        f"got {exclusion!r}"
    )


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
    decides next; a fault committed elsewhere is not owned; the last
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


def _unowned_faults(
    fault_rounds: tuple, fault_exclusion_ranges: tuple, candidate_faults: list
) -> set[int]:
    """The candidate faults that touch an excluded round."""
    if not fault_exclusion_ranges:
        return set()
    unowned = set()
    for fault_index in candidate_faults:
        if _touches_excluded_round(
            fault_rounds[fault_index], fault_exclusion_ranges
        ):
            unowned.add(fault_index)
    return unowned


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
