"""Parity reduction and the exported decoder-input matrix validators.

Reduces target ids to their odd-multiplicity identity and validates fault
identities, placed matrices, graphlike matrices and belief-matching matrices.

Leaf module: it references no Stim object, no window concept and no other
module of this package, so an external backend can validate its own matrices
with it.

Package-internal seam: _xor_target_ids lives here, with the identity
validators that consume it, and is also imported by stim_dem_catalog.
"""

from __future__ import annotations

from typing import Optional


def _xor_target_ids(target_ids) -> tuple[int, ...]:
    """Return the sorted ids with even multiplicities removed."""
    odd_ids: set[int] = set()
    for target_id in target_ids:
        if target_id in odd_ids:
            odd_ids.remove(target_id)
        else:
            odd_ids.add(target_id)
    return tuple(sorted(odd_ids))


def validate_fault_identity(
    detector_ids,
    logical_observable_ids,
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Canonicalize one placed fault without changing its decoder domain."""
    detectors = _xor_target_ids(detector_ids)
    logical_observables = _xor_target_ids(logical_observable_ids)
    if not detectors:
        if logical_observables:
            raise ValueError(
                f"{location} is a detectorless logical fault: "
                f"logical observables {logical_observables}"
            )
        return None
    return detectors, logical_observables


def validate_graphlike_fault(
    detector_ids,
    logical_observable_ids,
    *,
    location: str,
) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Canonicalize one fault and require the one-/two-detector domain."""
    fault = validate_fault_identity(
        detector_ids,
        logical_observable_ids,
        location=location,
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


def _binary_matrix(value, *, location: str, name: str):
    """Return one rank-2 binary matrix without changing its identities."""
    import numpy as np

    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{location} {name} must be a rank-2 matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{location} {name} must contain only binary values")
    return matrix.astype(np.uint8, copy=False)


def _placed_matrix_faults(
    check,
    observables,
    *,
    location: str,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    """Return canonical ids for every placed matrix fault column."""
    import numpy as np

    check_matrix = _binary_matrix(
        check,
        location=location,
        name="check",
    )
    observable_matrix = _binary_matrix(
        observables,
        location=location,
        name="observable matrix",
    )
    if check_matrix.shape[1] != observable_matrix.shape[1]:
        raise ValueError(
            f"{location} check and observable matrices have different fault "
            f"counts: {check_matrix.shape[1]} and "
            f"{observable_matrix.shape[1]}"
        )
    faults = []
    for fault_index in range(check_matrix.shape[1]):
        detector_ids = np.nonzero(check_matrix[:, fault_index])[0]
        logical_observable_ids = np.nonzero(
            observable_matrix[:, fault_index]
        )[0]
        faults.append(
            (
                fault_index,
                tuple(int(value) for value in detector_ids),
                tuple(int(value) for value in logical_observable_ids),
            )
        )
    return tuple(faults)


def validate_placed_fault_matrices(
    check,
    observables,
    *,
    location: str,
) -> None:
    """Reject lost logical identity while preserving a decoder's degree domain."""
    for fault_index, detector_ids, logical_ids in _placed_matrix_faults(
        check,
        observables,
        location=location,
    ):
        validate_fault_identity(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def validate_graphlike_matrices(
    check,
    observables,
    *,
    location: str,
) -> None:
    """Validate every placed fault column at a graphlike consumer boundary."""
    for fault_index, detector_ids, logical_ids in _placed_matrix_faults(
        check,
        observables,
        location=location,
    ):
        validate_graphlike_fault(
            detector_ids,
            logical_ids,
            location=f"{location} column {fault_index}",
        )


def validate_belief_matching_matrices(
    check,
    observables,
    hyperedge_check,
    hyperedge_priors,
    hyperedge_to_edge,
    *,
    location: str,
) -> None:
    """Validate the linked physical and graphlike belief-matching domains."""
    import numpy as np

    check_matrix = _binary_matrix(
        check,
        location=location,
        name="component check",
    )
    observable_matrix = _binary_matrix(
        observables,
        location=location,
        name="component observable matrix",
    )
    hyperedge_check_matrix = _binary_matrix(
        hyperedge_check,
        location=location,
        name="physical check",
    )
    hyperedge_to_edge_matrix = _binary_matrix(
        hyperedge_to_edge,
        location=location,
        name="physical-to-component map",
    )
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

    priors = np.asarray(hyperedge_priors, dtype=float)
    if priors.ndim != 1:
        raise ValueError(f"{location} physical priors must be rank 1")
    if priors.shape[0] != hyperedge_check_matrix.shape[1]:
        raise ValueError(
            f"{location} has {priors.shape[0]} physical priors for "
            f"{hyperedge_check_matrix.shape[1]} physical fault columns"
        )
    if not np.all(np.isfinite(priors)):
        raise ValueError(f"{location} physical priors must be finite")
    if not np.all((0.0 <= priors) & (priors <= 1.0)):
        raise ValueError(
            f"{location} physical priors must lie in the inclusive range [0, 1]"
        )

    validate_graphlike_matrices(
        check_matrix,
        observable_matrix,
        location=f"{location} component graph",
    )
    for physical_index in range(hyperedge_check_matrix.shape[1]):
        component_indices = np.nonzero(
            hyperedge_to_edge_matrix[:, physical_index]
        )[0]
        derived_detector_bits = (
            check_matrix[:, component_indices].sum(axis=1) % 2
        )
        stored_detector_bits = hyperedge_check_matrix[:, physical_index]
        if not np.array_equal(derived_detector_bits, stored_detector_bits):
            raise ValueError(
                f"{location} physical column {physical_index} detector "
                "identity does not equal its component XOR"
            )
        derived_logical_bits = (
            observable_matrix[:, component_indices].sum(axis=1) % 2
        )
        identity = validate_fault_identity(
            np.nonzero(stored_detector_bits)[0],
            np.nonzero(derived_logical_bits)[0],
            location=f"{location} physical column {physical_index}",
        )
        if identity is None:
            raise ValueError(
                f"{location} physical column {physical_index} is inert and "
                "must be removed before belief matching"
            )
