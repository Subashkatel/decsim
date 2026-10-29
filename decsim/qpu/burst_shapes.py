"""A noise burst's shape: its share of the peak in each round, and its area.

A burst rises over its first rounds, then decays exponentially, as
McEwen et al. fit a cosmic-ray burst (2104.05219, "typical ~25 ms
exponential decay") and as qec-burst-scaling draws one
(qecburst/circuit.py exponential_decay_profile). Its area is the qubits
near a point of the chip, placed by their first two Stim coordinates.
The shot burst (stim_device.BurstStimDevice) and the stream burst
(flip_stream_device.BurstSettings) both read this shape.
"""

import math
from typing import Optional


def peak_share(
    round_index: int,
    onset_round: int,
    rise_rounds: int,
    decay_rounds: Optional[float],
) -> float:
    """The share of the burst's peak in one round, 0 before the onset.

    The share climbs (i + 1) / rise_rounds in the i-th round from the
    onset, then decays as exp(-(rounds since the peak) / decay_rounds);
    decay_rounds None holds the peak. In every round one of the two
    terms is exactly 1, so a peak times the share is the same number
    whichever term is multiplied first.
    """
    rounds_since_onset = round_index - onset_round
    if rounds_since_onset < 0:
        return 0.0
    rounds_climbed = rounds_since_onset + 1
    climbed_share = rounds_climbed / rise_rounds
    rise_share = min(climbed_share, 1.0)
    if decay_rounds is None:
        return rise_share
    rounds_past_peak = rounds_climbed - rise_rounds
    rounds_since_peak = max(rounds_past_peak, 0)
    exponent = -rounds_since_peak / decay_rounds
    decay = math.exp(exponent)
    return rise_share * decay


def coordinate_midpoint(coordinates: dict) -> tuple:
    """The centre of the box the qubits' first two coordinates span."""
    positions = [_planar(position) for position in coordinates.values()]
    x_values = [position_x for position_x, _ in positions]
    y_values = [position_y for _, position_y in positions]
    middle_x = (min(x_values) + max(x_values)) / 2
    middle_y = (min(y_values) + max(y_values)) / 2
    return (middle_x, middle_y)


def planar_distances(coordinates: dict, center: tuple) -> dict:
    """Each qubit's distance from the centre in its first two coordinates."""
    distances = {}
    for qubit, position in coordinates.items():
        position_x, position_y = _planar(position)
        offset_x = position_x - center[0]
        offset_y = position_y - center[1]
        distances[qubit] = math.hypot(offset_x, offset_y)
    return distances


def _planar(position) -> tuple:
    """A qubit's first two coordinates; a line's qubits sit at y = 0."""
    if len(position) < 2:
        return (position[0], 0.0)
    return (position[0], position[1])
