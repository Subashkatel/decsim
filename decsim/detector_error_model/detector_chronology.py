"""Which round each detector belongs to, and where it sits in that round.

decsim numbers rounds from one. A detector belongs to the round whose
measurement packet completes it. Stim's generated circuits carry the round
in the last detector coordinate (Stim, doc/file_format_stim_circuit.md,
DETECTOR and SHIFT_COORDS): layer 0 holds the detectors compared against
the prepared state, layer t the ones formed after round t, and the final
data readout adds one more layer. Layer t therefore becomes round t + 1,
and the readout layer folds into the last round, which is where the
formation table and the window slicer expect it. A front end whose
schedule differs declares the map itself.

Inside a round, detectors keep Stim's index order; the decoders' syndrome
rows follow that order.
"""

import math
from typing import Optional


def resolve_detector_rounds(
    circuit, detector_rounds: Optional[dict], round_count: int
) -> dict[int, int]:
    """The round of every detector, declared or read off the coordinates.

    Raises ValueError unless the map covers every detector exactly once
    with rounds inside 1..round_count.
    """
    if round_count < 1:
        raise ValueError("round_count must be positive")
    detector_count = circuit.num_detectors
    if detector_count < 1:
        raise ValueError(
            "finite-memory chronology requires at least one detector"
        )
    if detector_rounds is None:
        resolved = _rounds_from_coordinates(
            circuit, detector_count, round_count
        )
    else:
        resolved = dict(detector_rounds)
    every_detector = set(range(detector_count))
    if set(resolved) != every_detector:
        raise ValueError("detector-round map must cover every detector exactly")
    after_last_round = round_count + 1
    emitted_rounds = set(range(1, after_last_round))
    resolved_rounds = resolved.values()
    if not set(resolved_rounds) <= emitted_rounds:
        raise ValueError(
            "detector-round map must lie inside the emitted rounds"
        )
    return resolved


def detectors_by_round(round_by_detector: dict) -> dict[int, list[int]]:
    """Each round's detectors in Stim index order."""
    grouped: dict[int, list[int]] = {}
    for detector_id in sorted(round_by_detector):
        round_index = round_by_detector[detector_id]
        detectors = grouped.setdefault(round_index, [])
        detectors.append(detector_id)
    return grouped


def detector_position_in_round(round_by_detector: dict) -> dict[int, int]:
    """Each detector's position among its round's detectors."""
    grouped = detectors_by_round(round_by_detector)
    position_by_detector = {}
    for detectors in grouped.values():
        for position, detector_id in enumerate(detectors):
            position_by_detector[detector_id] = position
    return position_by_detector


def coordinates_for_rows(
    detector_coordinates: dict, rows: list[int]
) -> Optional[tuple]:
    """Stim's coordinates for `rows`, or None when any row has none."""
    for detector_id in rows:
        if not detector_coordinates.get(detector_id):
            return None
    coordinates = []
    for detector_id in rows:
        row_coordinates = detector_coordinates[detector_id]
        coordinates.append(tuple(float(value) for value in row_coordinates))
    return tuple(coordinates)


def _rounds_from_coordinates(
    circuit, detector_count: int, round_count: int
) -> dict[int, int]:
    coordinates = circuit.get_detector_coordinates()
    arities = set()
    for detector_id in range(detector_count):
        detector_coordinates = coordinates.get(detector_id, ())
        arities.add(len(detector_coordinates))
    if len(arities) != 1:
        raise ValueError("finite-memory detector coordinates need one arity")
    coordinate_arity = next(iter(arities))
    # Stim's repetition code writes two coordinates, the surface and toric
    # codes three or more; the round is the last one either way.
    if coordinate_arity < 2:
        raise ValueError(
            "finite-memory chronology requires supported coordinates or "
            "explicit detector_rounds"
        )
    layer_by_detector = {}
    for detector_id in range(detector_count):
        layer_by_detector[detector_id] = _layer_of(coordinates[detector_id])
    after_last_round = round_count + 1
    allowed_layers = set(range(after_last_round))
    layers = layer_by_detector.values()
    if not set(layers) <= allowed_layers:
        raise ValueError(
            "raw detector layers must lie inside the declared source duration"
        )
    resolved = {}
    for detector_id, layer in layer_by_detector.items():
        resolved[detector_id] = _round_of_layer(layer, round_count)
    return resolved


def _layer_of(detector_coordinates) -> int:
    raw_value = detector_coordinates[-1]
    if not math.isfinite(raw_value) or raw_value != int(raw_value):
        raise ValueError(
            "finite-memory detector layers must be finite integers"
        )
    return int(raw_value)


def _round_of_layer(layer: int, round_count: int) -> int:
    # Layer t is formed after round t; the readout layer, one past the
    # last round, folds into the last round.
    if layer == round_count:
        return round_count
    return layer + 1
