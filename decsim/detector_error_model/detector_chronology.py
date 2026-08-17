"""Detector chronology and addressing.

Resolves one-based emitted rounds from a circuit or a supplied mapping, assigns
stable within-round positions, and projects all-or-nothing row coordinates.

Leaf module: it imports nothing from this package.

Package-internal seams: _detector_position_in_round and _coordinates_for_rows
are imported by window_slicer.
"""

from __future__ import annotations

import math
from typing import Optional


def resolve_detector_rounds(circuit, detector_rounds: Optional[dict],
                            round_count: int) -> dict[int, int]:
    """Resolve one finite source into decsim's one-based emitted rounds.

    Without a map, accept Stim repetition or decsim surface/toric coordinates.
    """
    if round_count < 1:
        raise ValueError("round_count must be positive")
    detector_count = circuit.num_detectors
    if detector_count < 1:
        raise ValueError("finite-memory chronology requires at least one detector")

    if detector_rounds is None:
        coordinates = circuit.get_detector_coordinates()
        arities = {len(coordinates.get(detector_id, ()))
                   for detector_id in range(detector_count)}
        if len(arities) != 1:
            raise ValueError("finite-memory detector coordinates need one arity")
        coordinate_arity = next(iter(arities))
        # Stim repetition coordinates have arity 2; surface/toric
        # coordinates have arity >= 3.
        if coordinate_arity < 2:
            raise ValueError(
                "finite-memory chronology requires supported coordinates or "
                "explicit detector_rounds"
            )
        raw_layers = {}
        for detector_id in range(detector_count):
            raw_value = coordinates[detector_id][-1]
            if not math.isfinite(raw_value) or raw_value != int(raw_value):
                raise ValueError("finite-memory detector layers must be finite integers")
            raw_layer = int(raw_value)
            raw_layers[detector_id] = raw_layer
        allowed_layers = set(range(round_count + 1))
        if not set(raw_layers.values()) <= allowed_layers:
            raise ValueError(
                "raw detector layers must lie inside the declared source duration"
            )
        resolved = {
            detector_id: (
                round_count if raw_layer == round_count else raw_layer + 1
            )
            for detector_id, raw_layer in raw_layers.items()
        }
    else:
        resolved = dict(detector_rounds)

    if set(resolved) != set(range(detector_count)):
        raise ValueError("detector-round map must cover every detector exactly")
    if not set(resolved.values()) <= set(range(1, round_count + 1)):
        raise ValueError("detector-round map must lie inside the emitted rounds")
    return resolved


def _detector_position_in_round(round_of: dict) -> dict:
    """Return detector id -> index within its round, using Stim detector order."""
    detectors_by_round: dict = {}
    for detector_id in sorted(round_of):
        round_index = round_of[detector_id]
        detectors_by_round.setdefault(round_index, []).append(detector_id)
    return {detector_id: index
            for detectors in detectors_by_round.values()
            for index, detector_id in enumerate(detectors)}


def _coordinates_for_rows(detector_coordinates: dict, rows: list[int]):
    """Return real circuit coordinates only when every local row has them."""
    if any(not detector_coordinates.get(detector_id) for detector_id in rows):
        return None
    return tuple(
        tuple(float(value) for value in detector_coordinates[detector_id])
        for detector_id in rows
    )
