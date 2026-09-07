"""Reduces one fault identity modulo two, where the catalog is built.

A target listed twice in one error cancels, and a fault that flips an
observable but no detector is undetectable and refused (Stim,
doc/file_format_dem_detector_error_model.md).
"""

from collections.abc import Iterable
from typing import Optional


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
