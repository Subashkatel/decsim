"""The detector layer counts every payload width in decsim rests on.

Stim lays a rotated surface-code memory in three kinds of layer: a
preparation layer that compares each check against the prepared state,
one bulk layer per round that compares a round against the one before
it, and a readout layer rebuilt from the data-qubit readout. The
preparation and readout layers hold half a round's detectors each and
the readout layer folds into the last round's packet, so a formed round
carries (d*d - 1)/2 events on the first round, d*d - 1 in the middle and
3(d*d - 1)/2 on the last, while the raw packet is d*d - 1 wide and
d*d - 1 + d*d on the last.

Those are the numbers the store hop and the tier's input link are priced
at. They are read here off stim.Circuit.generated, so a Stim release
that changed the layout would fail this file rather than a payload
arithmetic test somewhere else.
"""

import collections

import pytest
import stim

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.records.formation as formation_records

DISTANCES = (3, 5)


def _formation_table(distance: int, round_count: int):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=round_count,
        after_clifford_depolarization=0.001,
    )
    table = detector_formation.build_formation_table(circuit, round_count)
    return circuit, table


def _detectors_per_round(table) -> dict:
    per_round = collections.Counter()
    for recipe in table.detectors:
        per_round[recipe.round_index] += 1
    return per_round


@pytest.mark.parametrize("distance", DISTANCES)
def test_the_rounds_carry_half_bulk_and_one_and_a_half_of_stims_detectors(
    distance,
):
    """The readout folds into the last round; nothing is lost or doubled."""
    round_count = 4 * distance
    circuit, table = _formation_table(distance, round_count)
    per_round = _detectors_per_round(table)
    stabilizers = distance * distance - 1
    half = stabilizers // 2
    one_and_a_half = 3 * stabilizers // 2
    formed = len(table.detectors)

    assert per_round[1] == half
    assert per_round[2] == stabilizers
    assert per_round[round_count] == one_and_a_half
    assert formed == circuit.num_detectors


@pytest.mark.parametrize("distance", DISTANCES)
def test_the_three_layer_kinds_split_as_half_bulk_half(distance):
    round_count = 4 * distance
    _circuit, table = _formation_table(distance, round_count)
    by_kind = collections.Counter(recipe.kind for recipe in table.detectors)
    half = (distance * distance - 1) // 2

    assert by_kind[formation_records.LayerKind.PREPARATION] == half
    assert by_kind[formation_records.LayerKind.READOUT] == half


@pytest.mark.parametrize("distance", DISTANCES)
def test_the_raw_packet_is_one_bit_per_measure_qubit_plus_the_data_readout(
    distance,
):
    """The other width: what the QPU actually reads out each round."""
    round_count = 4 * distance
    _circuit, table = _formation_table(distance, round_count)
    widths = table.packet_width_by_round
    stabilizers = distance * distance - 1
    data_qubits = distance * distance

    assert widths[1] == stabilizers
    assert widths[2] == stabilizers
    assert widths[round_count] == stabilizers + data_qubits
