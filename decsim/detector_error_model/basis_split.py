"""A CSS region's physical fault model split into its X and Z parts.

Every detector of a CSS code compares checks of one Pauli type, so a
region can be decoded as two smaller problems. Relay-BP's XZ-decoding
(Muller et al. 2506.01779 lines 674-700): "every row in H is either X
or Z type ... We construct HX by simply extracting rows from H
corresponding to Z-type detectors ... pX is identical to p", then the
identical columns merged with the probability "that an odd number of
errors indexed by J occur". cudaqx splits the same way
(libs/qec/lib/experiments.cpp:205-257, x_component and z_component).

A detector's type is read from Stim's detecting regions on the data
qubits: an ancilla's region changes basis at its Hadamards, so only the
data qubits say which errors a detector sees. Each observable goes to
the one part whose detectors see the errors that flip it, the way
Stim's decomposition puts it on that type's pieces only; kept on both,
an X decode would flip a Z observable it has no business touching.
"""

import dataclasses
from typing import Optional

import numpy
import scipy.sparse
import stim

from decsim.detector_error_model import (
    stim_fault_catalog,
)
from decsim.records import fault_model_contracts

BASES = ("X", "Z")
# the Stim instructions that measure a qubit; the last one in a memory
# circuit reads the data qubits out
_MEASUREMENTS = frozenset({"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"})


def circuit_bases(circuit: stim.Circuit) -> tuple[dict, tuple]:
    """(type by detector id, type by observable index) of a CSS circuit.

    A detector or observable whose region on the data qubits is not all
    X or all Z has no part to go to, and splitting it would decode it
    wrong, so it is refused here, where the circuit enters.
    """
    data_qubits = _data_qubits(circuit)
    regions = circuit.detecting_regions()
    detector_bases = {}
    for index in range(circuit.num_detectors):
        target = stim.target_relative_detector_id(index)
        pauli_by_tick = regions.get(target, {})
        detector_bases[index] = _basis_of(target, pauli_by_tick, data_qubits)
    observable_bases = []
    for index in range(circuit.num_observables):
        target = stim.target_logical_observable_id(index)
        pauli_by_tick = regions.get(target, {})
        basis = _basis_of(target, pauli_by_tick, data_qubits)
        observable_bases.append(basis)
    return detector_bases, tuple(observable_bases)


def split_by_basis(
    model: fault_model_contracts.WindowErrorModel,
) -> dict[str, fault_model_contracts.WindowErrorModel]:
    """The region's X part and Z part, each a window model of its own."""
    parts = {}
    for basis in BASES:
        parts[basis] = _part(model, basis)
    return parts


def rows_of_basis(
    model: fault_model_contracts.WindowErrorModel, basis: str
) -> list[int]:
    """The model's rows whose detector is of this type, in row order."""
    rows = []
    for row_index, detector_id in enumerate(model.detector_ids):
        if model.detector_bases[detector_id] == basis:
            rows.append(row_index)
    return rows


def column_rows(matrix: scipy.sparse.csc_matrix, column: int) -> tuple:
    """The rows one column of a csc matrix flips, as Python ints.

    A check or an observable matrix of a placed fault model is csc, so a
    column's rows are one slice of its indices.
    """
    start = matrix.indptr[column]
    end = matrix.indptr[column + 1]
    indices = matrix.indices[start:end]
    return tuple(int(index) for index in indices)


@dataclasses.dataclass
class _MergedColumns:
    """One part's columns, each identical restriction kept once.

    A column is identified by the part's rows it flips, the part's
    observables it flips, whether the window commits it and the part's
    detectors it flips anywhere, so merging never joins a committed
    column with one the window does not own.
    """

    prior_by_key: dict = dataclasses.field(default_factory=dict)
    fault_id_by_key: dict = dataclasses.field(default_factory=dict)

    def add(self, key: tuple, prior: float, fault_id: int) -> None:
        """One more fault on this key, its prior merged as odd counts."""
        current = self.prior_by_key.get(key, 0.0)
        merged = stim_fault_catalog.merge_probability(current, prior)
        self.prior_by_key[key] = merged
        self.fault_id_by_key.setdefault(key, fault_id)

    def placed(
        self, row_count: int, observable_count: int
    ) -> fault_model_contracts.PlacedFaultModel:
        """The merged columns as one placed physical model.

        A merged column keeps the catalog id of its first fault.
        """
        keys = list(self.prior_by_key)
        rows_by_column = [key[0] for key in keys]
        observables_by_column = [key[1] for key in keys]
        check = _column_matrix(rows_by_column, row_count)
        observables = _column_matrix(observables_by_column, observable_count)
        flips = {}
        for column, key in enumerate(keys):
            if key[2]:
                flips[column] = key[3]
        merged_priors = self.prior_by_key.values()
        priors = list(merged_priors)
        owned = [key[2] for key in keys]
        first_fault_ids = self.fault_id_by_key.values()
        fault_ids = list(first_fault_ids)
        return fault_model_contracts.PlacedFaultModel(
            representation=fault_model_contracts.FaultRepresentation.PHYSICAL,
            check=check,
            priors=priors,
            observables=observables,
            owned=owned,
            source_fault_ids=fault_ids,
            boundary_flips=flips,
        )


def _data_qubits(circuit: stim.Circuit) -> numpy.ndarray:
    """A mask of the qubits the circuit's last measurement reads."""
    last_measurement = None
    for instruction in circuit.flattened():
        if instruction.name in _MEASUREMENTS:
            last_measurement = instruction
    mask = numpy.zeros(circuit.num_qubits, dtype=bool)
    for target in last_measurement.targets_copy():
        mask[target.value] = True
    return mask


def _basis_of(target, pauli_by_tick: dict, data_qubits) -> str:
    """X or Z: the one Pauli its region puts on the data qubits."""
    letters = set()
    for pauli_string in pauli_by_tick.values():
        xs, zs = pauli_string.to_numpy()
        on_data = data_qubits[: len(xs)]
        tick_letters = _letters_on(xs, zs, on_data)
        letters.update(tick_letters)
    if letters == {"X"} or letters == {"Z"}:
        return letters.pop()
    raise ValueError(
        f"{target} is not of one type on the data qubits (it puts "
        f"{sorted(letters)} there); bases: apart splits a CSS code's "
        "detectors by type (Relay-BP 2506.01779 lines 674-679)"
    )


def _letters_on(xs, zs, on_data) -> set:
    """The Paulis one tick of a region puts on the data qubits."""
    letters = set()
    only_x = xs & ~zs & on_data
    only_z = zs & ~xs & on_data
    both = xs & zs & on_data
    for letter, present in (("X", only_x), ("Z", only_z), ("Y", both)):
        if present.any():
            letters.add(letter)
    return letters


def _part(
    model: fault_model_contracts.WindowErrorModel, basis: str
) -> fault_model_contracts.WindowErrorModel:
    """One type's rows, their columns, and the positions of its detectors."""
    rows = rows_of_basis(model, basis)
    detector_ids = tuple(model.detector_ids[row] for row in rows)
    positions = {}
    for detector_id, position in model.defect_positions.items():
        if model.detector_bases[detector_id] == basis:
            positions[detector_id] = position
    bases = dict.fromkeys(positions, basis)
    faults = _part_faults(model, basis, rows)
    coordinates = _part_coordinates(model.detector_coordinates, rows)
    return fault_model_contracts.WindowErrorModel(
        detector_ids=detector_ids,
        detector_coordinates=coordinates,
        defect_positions=positions,
        first_commit_round=model.first_commit_round,
        graphlike_faults=None,
        physical_faults=faults,
        detector_bases=bases,
        observable_bases=model.observable_bases,
    )


def _part_coordinates(coordinates, rows: list) -> Optional[tuple]:
    if coordinates is None:
        return None
    return tuple(coordinates[row] for row in rows)


def _part_faults(
    model: fault_model_contracts.WindowErrorModel, basis: str, rows: list
) -> fault_model_contracts.PlacedFaultModel:
    """Each column restricted to the part, the empty ones dropped."""
    faults = model.physical_faults
    row_matrix = faults.check.tocsr()
    part_check = row_matrix[rows, :].tocsc()
    kept_observables = _observables_of(model.observable_bases, basis)
    merged = _MergedColumns()
    for column in range(part_check.shape[1]):
        part_rows = column_rows(part_check, column)
        if not part_rows:
            continue
        key = _column_key(model, basis, column, part_rows, kept_observables)
        prior = float(faults.priors[column])
        merged.add(key, prior, faults.source_fault_ids[column])
    observable_count = faults.observables.shape[0]
    return merged.placed(len(rows), observable_count)


def _column_key(
    model, basis: str, column: int, part_rows: tuple, kept_observables
) -> tuple:
    """(part rows, part observables, owned, part detectors it flips)."""
    faults = model.physical_faults
    observables = column_rows(faults.observables, column)
    part_observables = tuple(
        index for index in observables if index in kept_observables
    )
    owned = bool(faults.owned[column])
    flips = faults.boundary_flips.get(column, ())
    detector_bases = model.detector_bases
    part_flips = tuple(
        detector for detector in flips if detector_bases[detector] == basis
    )
    return part_rows, part_observables, owned, part_flips


def _observables_of(observable_bases: tuple, basis: str) -> frozenset:
    """The observables whose type is this part's."""
    kept = set()
    for index, observable_basis in enumerate(observable_bases):
        if observable_basis == basis:
            kept.add(index)
    return frozenset(kept)


def _column_matrix(rows_by_column: list, row_count: int):
    """A uint8 csc matrix with the given rows set in each column."""
    row_indices = []
    column_indices = []
    for column, column_rows in enumerate(rows_by_column):
        row_indices.extend(column_rows)
        repeats = len(column_rows)
        repeated = [column] * repeats
        column_indices.extend(repeated)
    ones = numpy.ones(len(row_indices), dtype=numpy.uint8)
    shape = (row_count, len(rows_by_column))
    coordinates = (row_indices, column_indices)
    return scipy.sparse.csc_matrix((ones, coordinates), shape=shape)
