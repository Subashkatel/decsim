"""Reduces fault identities modulo two and checks decoder input matrices.

A target listed twice in one error cancels, and a fault that flips an
observable but no detector is undetectable and refused (Stim,
doc/file_format_dem_detector_error_model.md); the matrix checks read one
column per fault the way PyMatching's from_check_matrix and the
belief-matching decoder (Higgott et al., beliefmatching) do.
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


def validate_graphlike_matrices(
    check: object, observables: object, *, location: str
) -> None:
    """Refuse a column a matching decoder cannot represent."""
    faults = _placed_matrix_faults(check, observables, location=location)
    for fault_index, detector_ids, logical_ids in faults:
        validate_graphlike_fault(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def validate_belief_matching_matrices(
    check: object,
    observables: object,
    hyperedge_check: object,
    hyperedge_priors: object,
    hyperedge_to_edge: object,
    *,
    location: str,
) -> None:
    """Check the linked physical and graphlike belief-matching inputs."""
    check_matrix, observable_matrix = _component_matrices(
        check, observables, location=location
    )
    hyperedge_check_matrix, hyperedge_to_edge_matrix = _physical_matrices(
        hyperedge_check, hyperedge_to_edge, location=location
    )
    _validate_linked_shapes(
        check_matrix,
        hyperedge_check_matrix,
        hyperedge_to_edge_matrix,
        location=location,
    )
    _validate_priors(
        hyperedge_priors, hyperedge_check_matrix.shape[1], location=location
    )
    validate_graphlike_matrices(
        check_matrix,
        observable_matrix,
        location=f"{location} component graph",
    )
    for physical_index in range(hyperedge_check_matrix.shape[1]):
        _validate_physical_column(
            physical_index,
            check_matrix,
            observable_matrix,
            hyperedge_check_matrix,
            hyperedge_to_edge_matrix,
            location=location,
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


def _component_matrices(check, observables, *, location: str):
    """The graphlike check and observable matrices, normalised."""
    check_matrix = _binary_matrix(
        check, location=location, name="component check"
    )
    observable_matrix = _binary_matrix(
        observables, location=location, name="component observable matrix"
    )
    return check_matrix, observable_matrix


def _physical_matrices(hyperedge_check, hyperedge_to_edge, *, location: str):
    """The physical check and the physical-to-component map, normalised."""
    hyperedge_check_matrix = _binary_matrix(
        hyperedge_check, location=location, name="physical check"
    )
    hyperedge_to_edge_matrix = _binary_matrix(
        hyperedge_to_edge,
        location=location,
        name="physical-to-component map",
    )
    return hyperedge_check_matrix, hyperedge_to_edge_matrix


def _validate_linked_shapes(
    check_matrix, hyperedge_check_matrix, hyperedge_to_edge_matrix, *, location
) -> None:
    if check_matrix.shape[0] != hyperedge_check_matrix.shape[0]:
        raise ValueError(
            f"{location} component and physical checks have different "
            "detector counts"
        )
    expected_map_shape = (
        check_matrix.shape[1],
        hyperedge_check_matrix.shape[1],
    )
    if hyperedge_to_edge_matrix.shape != expected_map_shape:
        raise ValueError(
            f"{location} physical-to-component map has shape "
            f"{hyperedge_to_edge_matrix.shape}; expected {expected_map_shape}"
        )


def _validate_priors(
    hyperedge_priors, physical_fault_count: int, *, location: str
) -> None:
    priors = numpy.asarray(hyperedge_priors, dtype=float)
    if priors.ndim != 1:
        raise ValueError(f"{location} physical priors must be rank 1")
    if priors.shape[0] != physical_fault_count:
        raise ValueError(
            f"{location} has {priors.shape[0]} physical priors for "
            f"{physical_fault_count} physical fault columns"
        )
    is_finite = numpy.isfinite(priors)
    if not numpy.all(is_finite):
        raise ValueError(f"{location} physical priors must be finite")
    is_at_least_zero = 0.0 <= priors
    is_at_most_one = priors <= 1.0
    is_probability = is_at_least_zero & is_at_most_one
    if not numpy.all(is_probability):
        raise ValueError(
            f"{location} physical priors must lie in the inclusive range [0, 1]"
        )


def _validate_physical_column(
    physical_index: int,
    check_matrix,
    observable_matrix,
    hyperedge_check_matrix,
    hyperedge_to_edge_matrix,
    *,
    location: str,
) -> None:
    """A physical column must be the parity of its graphlike components."""
    component_indices = _column_rows(hyperedge_to_edge_matrix, physical_index)
    derived_detector_bits = _column_parity(check_matrix, component_indices)
    stored_detector_bits = numpy.zeros(check_matrix.shape[0], dtype=numpy.uint8)
    stored_rows = _column_rows(hyperedge_check_matrix, physical_index)
    stored_detector_bits[stored_rows] = 1
    if not numpy.array_equal(derived_detector_bits, stored_detector_bits):
        raise ValueError(
            f"{location} physical column {physical_index} detector "
            "identity does not equal its component XOR"
        )
    derived_logical_bits = _column_parity(observable_matrix, component_indices)
    stored_detectors = numpy.nonzero(stored_detector_bits)
    derived_logicals = numpy.nonzero(derived_logical_bits)
    identity = validate_fault_identity(
        stored_detectors[0],
        derived_logicals[0],
        location=f"{location} physical column {physical_index}",
    )
    if identity is None:
        raise ValueError(
            f"{location} physical column {physical_index} is inert and "
            "must be removed before belief matching"
        )


def _column_parity(matrix, column_indices):
    """The parity, row by row, of the chosen columns of a binary matrix."""
    chosen = matrix[:, column_indices]
    column_sums = chosen.sum(axis=1)
    flat_sums = numpy.asarray(column_sums)
    flat = flat_sums.ravel()
    return flat % 2
