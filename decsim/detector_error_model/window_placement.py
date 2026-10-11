"""Places the faults of one window: its rows, its columns, what it owns.

A window's rows are the detectors of its rounds, its columns every
catalog fault that flips one of them and that no earlier window
committed, and it owns the faults touching its commit rounds (Skoric et
al. 2209.08552, section I.B; qLDPC's SlidingWindowDecoder, whose
`d_errors[addressed] = False` removes a committed fault from later
windows). An owned column hands off its whole detector effect, so a
window before or after in time can intersect it with its own rows (Tan
et al. 2209.09219).

The catalog is laid out once as compressed columns (CatalogArrays), the
layout of qLDPC's DetectorErrorModelArrays and of scipy's csc_matrix,
and a window is placed with array operations on its columns' entries: a
Python pass over a strong region's columns was a third of a switching
d11 shot (stage7_plan.md).
"""

import dataclasses
import itertools
from collections.abc import Container, Sequence
from typing import Optional

import numpy
import scipy.sparse

from decsim.records import fault_model_contracts


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


@dataclasses.dataclass(frozen=True)
class CatalogArrays:
    """One catalog and its faults' rounds as compressed columns.

    Fault f's detectors are detector_ids[detector_pointer[f]:
    detector_pointer[f + 1]] in the catalog's order, and entry_rounds
    holds the round of each of those detectors; its observables are the
    same slice of observable_ids by observable_pointer.
    """

    detector_pointer: numpy.ndarray
    detector_ids: numpy.ndarray
    entry_rounds: numpy.ndarray
    observable_pointer: numpy.ndarray
    observable_ids: numpy.ndarray
    priors: numpy.ndarray


def catalog_arrays(
    catalog: fault_model_contracts.FaultCatalog,
    fault_rounds: tuple[tuple[int, ...], ...],
) -> CatalogArrays:
    """The catalog laid out once, with each fault's rounds beside it."""
    detector_pointer, detector_ids = _compressed(catalog.detector_sets)
    _round_pointer, entry_rounds = _compressed(fault_rounds)
    observable_pointer, observable_ids = _compressed(catalog.observable_sets)
    priors = numpy.asarray(catalog.priors, dtype=float)
    return CatalogArrays(
        detector_pointer=detector_pointer,
        detector_ids=detector_ids,
        entry_rounds=entry_rounds,
        observable_pointer=observable_pointer,
        observable_ids=observable_ids,
        priors=priors,
    )


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
    arrays: CatalogArrays,
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
    candidate_faults: list[int],
) -> set[int]:
    """The candidate faults that touch an excluded round.

    Such a fault is owned by nobody: the window that sees it decodes it
    and never commits it, on either ownership path.
    """
    faults = numpy.asarray(candidate_faults, dtype=numpy.int64)
    entries = _column_entries(arrays.detector_pointer, faults)
    touching = _columns_touching(
        arrays.entry_rounds, entries, fault_exclusion_ranges, len(faults)
    )
    touching_faults = faults[touching]
    found = touching_faults.tolist()
    return set(found)


def placed_faults_for_window(
    *,
    catalog: fault_model_contracts.FaultCatalog,
    context: WindowPlacementContext,
    arrays: CatalogArrays,
    candidate_faults: list[int],
    committed_elsewhere: set[int],
    explicitly_owned_faults: Optional[set[int]],
    explicitly_prior_faults: Optional[Container[int]],
    fault_exclusion_ranges: tuple[tuple[int, int], ...],
) -> fault_model_contracts.PlacedFaultModel:
    """One window's columns from one catalog.

    `arrays` is the catalog laid out once and `candidate_faults` the
    faults that touch this window's rounds; the slicer indexes both
    once, so a window costs its own size.
    """
    prior_faults = _prior_faults(committed_elsewhere, explicitly_prior_faults)
    columns = _fault_columns_for_window(candidate_faults, prior_faults)
    column_faults = numpy.asarray(columns, dtype=numpy.int64)
    entries = _column_entries(arrays.detector_pointer, column_faults)
    unowned = _columns_touching(
        arrays.entry_rounds, entries, fault_exclusion_ranges, len(columns)
    )
    owners = _Owners(committed_elsewhere, explicitly_owned_faults, unowned)
    owned = _owned_columns(context, arrays, column_faults, entries, owners)
    owned_columns = numpy.flatnonzero(owned)
    owned_faults = column_faults[owned_columns]
    owned_fault_list = owned_faults.tolist()
    if explicitly_owned_faults is None:
        committed_elsewhere.update(owned_fault_list)
    # The whole detector effect travels with the handoff; the receiving
    # window keeps the part that lands on its own rows.
    boundary_flips = _boundary_flips(
        owned_columns, owned_fault_list, catalog.detector_sets
    )
    check = _check_matrix(context, arrays, entries, len(columns))
    observables = _observable_matrix(context, arrays, column_faults)
    priors = arrays.priors[column_faults]
    return fault_model_contracts.PlacedFaultModel(
        representation=catalog.representation,
        check=check,
        priors=priors,
        observables=observables,
        owned=owned,
        source_fault_ids=tuple(columns),
        boundary_flips=boundary_flips,
    )


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


@dataclasses.dataclass(frozen=True)
class _Entries:
    """A window's columns' flat entries: where each sits, and its column."""

    positions: numpy.ndarray
    columns: numpy.ndarray


@dataclasses.dataclass(frozen=True)
class _Owners:
    """What decides a window's ownership besides its commit rounds."""

    committed_elsewhere: set
    explicitly_owned_faults: Optional[set]
    # the window's columns that touch an excluded round
    unowned: numpy.ndarray


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


def _owned_columns(
    context: WindowPlacementContext,
    arrays: CatalogArrays,
    column_faults: numpy.ndarray,
    entries: _Entries,
    owners: _Owners,
) -> numpy.ndarray:
    """Which of the window's columns it commits.

    Precedence: an excluded fault is never owned; an explicit owner set
    decides next; a fault committed elsewhere is not owned; the terminal
    window owns the rest; any other window owns what touches its commit
    rounds.
    """
    may_own = ~owners.unowned
    if owners.explicitly_owned_faults is not None:
        named = _array_of(owners.explicitly_owned_faults)
        is_named = numpy.isin(column_faults, named)
        return may_own & is_named
    committed = _array_of(owners.committed_elsewhere)
    is_committed = numpy.isin(column_faults, committed)
    may_own &= ~is_committed
    if context.is_last:
        return may_own
    commit_rounds = ((context.first_commit_round, context.last_commit_round),)
    column_count = len(column_faults)
    in_commit = _columns_touching(
        arrays.entry_rounds, entries, commit_rounds, column_count
    )
    return may_own & in_commit


def _boundary_flips(
    owned_columns: numpy.ndarray, owned_faults: list, detector_sets: tuple
) -> dict:
    """Each owned column's every detector, keyed by its column."""
    columns = owned_columns.tolist()
    pairs = zip(columns, owned_faults, strict=True)
    return {column: detector_sets[fault] for column, fault in pairs}


def _check_matrix(
    context: WindowPlacementContext,
    arrays: CatalogArrays,
    entries: _Entries,
    column_count: int,
) -> scipy.sparse.csc_matrix:
    """Rows by columns, a one where a column flips one of the window's rows.

    The rows are sorted, so each entry's detector finds its row by
    binary search, and an entry whose detector is no row is dropped.
    """
    window_rows = numpy.asarray(context.rows, dtype=numpy.int64)
    row_count = len(window_rows)
    detectors = arrays.detector_ids[entries.positions]
    places = numpy.searchsorted(window_rows, detectors)
    inside = places < row_count
    kept_places = places[inside]
    kept_detectors = detectors[inside]
    found = window_rows[kept_places] == kept_detectors
    rows = kept_places[found]
    entry_columns = entries.columns[inside]
    columns = entry_columns[found]
    ones = numpy.ones(len(rows), dtype=numpy.uint8)
    shape = (row_count, column_count)
    return scipy.sparse.csc_matrix((ones, (rows, columns)), shape=shape)


def _observable_matrix(
    context: WindowPlacementContext,
    arrays: CatalogArrays,
    column_faults: numpy.ndarray,
) -> scipy.sparse.csc_matrix:
    """Observables by columns; every flip counts, local rows or not."""
    entries = _column_entries(arrays.observable_pointer, column_faults)
    rows = arrays.observable_ids[entries.positions]
    ones = numpy.ones(len(rows), dtype=numpy.uint8)
    shape = (context.observable_count, len(column_faults))
    return scipy.sparse.csc_matrix((ones, (rows, entries.columns)), shape=shape)


def _column_entries(
    pointer: numpy.ndarray, column_faults: numpy.ndarray
) -> _Entries:
    """Every entry of the columns' faults, column by column, in order."""
    next_faults = column_faults + 1
    starts = pointer[column_faults]
    ends = pointer[next_faults]
    counts = ends - starts
    column_numbers = numpy.arange(len(column_faults))
    columns = numpy.repeat(column_numbers, counts)
    first_positions = numpy.cumsum(counts) - counts
    total = counts.sum()
    offsets = numpy.arange(total)
    column_firsts = numpy.repeat(first_positions, counts)
    column_starts = numpy.repeat(starts, counts)
    within = offsets - column_firsts
    positions = column_starts + within
    return _Entries(positions=positions, columns=columns)


def _columns_touching(
    entry_rounds: numpy.ndarray,
    entries: _Entries,
    round_ranges: tuple[tuple[int, int], ...],
    column_count: int,
) -> numpy.ndarray:
    """Whether each column has an entry inside any of the round ranges."""
    touching = numpy.zeros(column_count, dtype=bool)
    rounds = entry_rounds[entries.positions]
    for first_round, last_round in round_ranges:
        after_first = rounds >= first_round
        before_last = rounds <= last_round
        inside = after_first & before_last
        touched_columns = entries.columns[inside]
        touching[touched_columns] = True
    return touching


def _compressed(sets: tuple) -> tuple:
    """A tuple of tuples as a pointer and its flat entries, in order."""
    sizes = map(len, sets)
    lengths = numpy.fromiter(sizes, dtype=numpy.int64, count=len(sets))
    ends = numpy.cumsum(lengths)
    first = numpy.zeros(1, dtype=numpy.int64)
    pointer = numpy.concatenate((first, ends))
    flat = itertools.chain.from_iterable(sets)
    entries = numpy.fromiter(flat, dtype=numpy.int64)
    return pointer, entries


def _array_of(faults: set) -> numpy.ndarray:
    return numpy.fromiter(faults, dtype=numpy.int64, count=len(faults))
