"""Reduces fault identities modulo two and checks decoder input matrices.

A target listed twice in one error cancels, and a fault that flips an
observable but no detector is undetectable and refused (Stim,
doc/file_format_dem_detector_error_model.md); the matrix check reads one
column per fault the way PyMatching's from_check_matrix does.
"""

from collections.abc import Iterable
from typing import Optional

import numpy
import scipy.sparse


def xor_target_ids(target_ids: Iterable[int]) -> tuple[int, ...]:
    """The sorted ids that occur an odd number of times."""
    odd_ids: set[int] = set()
    for target_id in target_ids:
        if target_id in odd_ids:
            odd_ids.remove(target_id)
        else:
            odd_ids.add(target_id)
    return tuple(sorted(odd_ids))


def validate_fault_identity(
    detector_ids: Iterable[int],
    logical_observable_ids: Iterable[int],
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Reduce one fault modulo two; None when it flips nothing.

    Raises ValueError for a fault that flips an observable but no detector.
    """
    detectors = xor_target_ids(detector_ids)
    logical_observables = xor_target_ids(logical_observable_ids)
    if not detectors:
        if logical_observables:
            raise ValueError(
                f"{location} is a detectorless logical fault: "
                f"logical observables {logical_observables}"
            )
        return None
    return detectors, logical_observables


def validate_graphlike_fault(
    detector_ids: Iterable[int],
    logical_observable_ids: Iterable[int],
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Reduce one fault and require at most two detectors."""
    fault = validate_fault_identity(
        detector_ids, logical_observable_ids, location=location
    )
    if fault is None:
        return None
    detectors, logical_observables = fault
    if len(detectors) > 2:
        raise ValueError(
            f"{location} is a detector hyperedge with detectors {detectors}; "
            "this graphlike decoder supports one or two detectors per fault"
        )
    return detectors, logical_observables


def validate_placed_fault_matrices(
    check: object, observables: object, *, location: str
) -> None:
    """Refuse a column that has lost its logical identity."""
    faults = _placed_matrix_faults(check, observables, location=location)
    for fault_index, detector_ids, logical_ids in faults:
        validate_fault_identity(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def _binary_matrix(value, *, location: str, name: str):
    """One rank-2 binary matrix as a uint8 csc_matrix with sorted indices."""
    if scipy.sparse.issparse(value):
        # The placed matrices are frozen; the copy is what gets normalised.
        column_matrix = value.tocsc()
        matrix = column_matrix.copy()
    else:
        dense = numpy.asarray(value)
        if dense.ndim != 2:
            raise ValueError(f"{location} {name} must be a rank-2 matrix")
        matrix = scipy.sparse.csc_matrix(dense)
    matrix.sum_duplicates()
    matrix.sort_indices()
    is_zero = matrix.data == 0
    is_one = matrix.data == 1
    is_binary = is_zero | is_one
    if not numpy.all(is_binary):
        raise ValueError(f"{location} {name} must contain only binary values")
    matrix.eliminate_zeros()
    return matrix.astype(numpy.uint8, copy=False)


def _column_rows(matrix, column: int):
    """The sorted row indices of one column of a csc_matrix."""
    start = matrix.indptr[column]
    end = matrix.indptr[column + 1]
    return matrix.indices[start:end]


def _placed_matrix_faults(
    check, observables, *, location: str
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    """Every column as (index, detector ids, logical observable ids)."""
    check_matrix = _binary_matrix(check, location=location, name="check")
    observable_matrix = _binary_matrix(
        observables, location=location, name="observable matrix"
    )
    if check_matrix.shape[1] != observable_matrix.shape[1]:
        raise ValueError(
            f"{location} check and observable matrices have different fault "
            f"counts: {check_matrix.shape[1]} and "
            f"{observable_matrix.shape[1]}"
        )
    faults = []
    for fault_index in range(check_matrix.shape[1]):
        detector_ids = _column_rows(check_matrix, fault_index)
        logical_observable_ids = _column_rows(observable_matrix, fault_index)
        detectors = tuple(int(value) for value in detector_ids)
        logicals = tuple(int(value) for value in logical_observable_ids)
        faults.append((fault_index, detectors, logicals))
    return tuple(faults)
