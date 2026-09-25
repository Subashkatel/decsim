"""The X and Z split against Stim's own decomposition of the same circuit.

Stim's decompose_errors=True model breaks every error into graphlike
pieces that each touch one detector type, and puts an observable only on
pieces of the type that detects it. XORing one error's pieces of one type
gives the column Relay-BP's row recipe keeps for that type (2506.01779
lines 674-700), so the two parts decsim builds must be exactly those
columns, merged as odd counts.
"""

import collections

import pytest
import stim

import decsim.detector_error_model.basis_split as basis_split
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.stim_fault_catalog as stim_fault_catalog
import decsim.detector_error_model.window_model_builders as builders
from tests.decoders import windows

REQUIREMENT = fault_models.PHYSICAL_FAULT_MODEL_REQUIRED.joined(
    fault_models.DETECTOR_BASES_REQUIRED
)


def _circuit(memory_basis: str, distance: int, rounds: int) -> stim.Circuit:
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.003,
        before_measure_flip_probability=0.003,
        after_reset_flip_probability=0.003,
        before_round_data_depolarization=0.003,
    )


def _stims_pieces(circuit: stim.Circuit) -> list:
    """Every error's graphlike pieces: (probability, [(detectors, obs)])."""
    model = circuit.detector_error_model(decompose_errors=True)
    errors = []
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        pieces = [([], [])]
        for target in instruction.targets_copy():
            _add_target(pieces, target)
        arguments = instruction.args_copy()
        errors.append((arguments[0], pieces))
    return errors


def _add_target(pieces: list, target) -> None:
    if target.is_separator():
        pieces.append(([], []))
        return
    detectors, observables = pieces[-1]
    if target.is_relative_detector_id():
        detectors.append(target.val)
        return
    observables.append(target.val)


def _stims_part(circuit: stim.Circuit, types: dict, basis: str) -> dict:
    """(detectors, observables) -> odd-count prior, from Stim's pieces."""
    merged = collections.defaultdict(float)
    for probability, pieces in _stims_pieces(circuit):
        key = _pieces_of_type(pieces, types, basis)
        if not key[0]:
            continue
        prior = stim_fault_catalog.merge_probability(merged[key], probability)
        merged[key] = prior
    return dict(merged)


def _pieces_of_type(pieces: list, types: dict, basis: str) -> tuple:
    """One error's pieces of one type XORed: (detectors, observables)."""
    detectors, observables = set(), set()
    for piece_detectors, piece_observables in pieces:
        piece_types = {types[detector] for detector in piece_detectors}
        if piece_types != {basis}:
            continue
        detectors ^= set(piece_detectors)
        observables ^= set(piece_observables)
    return frozenset(detectors), frozenset(observables)


def _decsims_part(part) -> dict:
    """(detector ids, observables) -> prior of each part column."""
    faults = part.physical_faults
    columns = {}
    for column in range(faults.check.shape[1]):
        rows = _column(faults.check, column)
        detectors = frozenset(part.detector_ids[row] for row in rows)
        flipped = _column(faults.observables, column)
        observables = frozenset(flipped)
        columns[(detectors, observables)] = float(faults.priors[column])
    return columns


def _column(matrix, column: int) -> list:
    start = matrix.indptr[column]
    end = matrix.indptr[column + 1]
    return [int(index) for index in matrix.indices[start:end]]


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("memory_basis", ["x", "z"])
def test_every_graphlike_piece_stim_decomposes_keeps_to_one_type(
    memory_basis, distance
):
    circuit = _circuit(memory_basis, distance, distance)
    types, _ = basis_split.circuit_bases(circuit)
    piece_types = set()
    for _, pieces in _stims_pieces(circuit):
        for detectors, _observables in pieces:
            kinds = frozenset(types[detector] for detector in detectors)
            piece_types.add(kinds)
    assert piece_types == {frozenset("X"), frozenset("Z")}


@pytest.mark.parametrize("memory_basis", ["x", "z"])
def test_the_observable_is_of_the_memorys_own_basis(memory_basis):
    circuit = _circuit(memory_basis, 3, 3)
    _, observable_bases = basis_split.circuit_bases(circuit)
    assert observable_bases == (memory_basis.upper(),)


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("memory_basis", ["x", "z"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_each_part_is_stims_decomposition_of_that_type(
    memory_basis, distance, basis
):
    circuit = _circuit(memory_basis, distance, distance)
    types, _ = basis_split.circuit_bases(circuit)
    model = windows.whole_circuit_window(circuit, distance, REQUIREMENT)
    parts = basis_split.split_by_basis(model)
    part = parts[basis]
    expected = _stims_part(circuit, types, basis)
    actual = _decsims_part(part)
    assert actual.keys() == expected.keys()
    assert actual == pytest.approx(expected, rel=1e-9)


def test_a_detector_of_neither_type_is_refused():
    circuit = stim.Circuit("RY 0\nTICK\nMY 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError) as refusal:
        basis_split.circuit_bases(circuit)
    assert str(refusal.value) == (
        "D0 is not of one type on the data qubits (it puts ['Y'] there); "
        "bases: apart splits a CSS code's detectors by type (Relay-BP "
        "2506.01779 lines 674-679)"
    )


def _restricted_columns(model, faults, basis: str, detectors_of) -> set:
    """Each column as (its type's detectors, observables, owned, flips)."""
    bases = model.detector_bases
    kept_observables = {
        index
        for index, observable_basis in enumerate(model.observable_bases)
        if observable_basis == basis
    }
    columns = set()
    for column in range(faults.check.shape[1]):
        detectors = detectors_of(column)
        typed = frozenset(d for d in detectors if bases[d] == basis)
        if not typed:
            continue
        flipped = _column(faults.observables, column)
        kept = kept_observables.intersection(flipped)
        observables = frozenset(kept)
        owned = bool(faults.owned[column])
        flips = faults.boundary_flips.get(column, ())
        typed_flips = frozenset(d for d in flips if bases[d] == basis)
        restricted = (typed, observables, owned, typed_flips)
        columns.add(restricted)
    return columns


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("window_entry", [(4, 6, 9), (2, 4, 6, 9)])
def test_a_windowed_part_keeps_what_its_window_commits_and_hands_on(
    window_entry, basis
):
    """Rows, owned mask and the flips handed on, each cut to one type."""
    circuit = _circuit("z", 3, 9)
    model = builders.build_single_window_error_model(
        circuit,
        window_entry,
        round_count=9,
        fault_model_requirement=REQUIREMENT,
    )
    whole = model.physical_faults
    parts = basis_split.split_by_basis(model)
    part = parts[basis]
    faults = part.physical_faults

    def whole_detectors(column):
        return [model.detector_ids[row] for row in _column(whole.check, column)]

    def part_detectors(column):
        return [part.detector_ids[row] for row in _column(faults.check, column)]

    expected = _restricted_columns(model, whole, basis, whole_detectors)
    actual = _restricted_columns(part, faults, basis, part_detectors)
    assert actual == expected
