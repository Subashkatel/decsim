"""Which patches' past can flip a lattice-surgery merge's outcome, by Stim.

    python tools/membrane_check.py [DISTANCE]

A merge's corrected outcome is its raw parity plus the decoded errors on
its membrane (Bombín et al. 2303.04846 Eq. 4): a parity the noiseless
circuit fixes that holds the outcome's records. Three rotated
surface-code patches, A, B and C (distance 3 unless given), each start
in a logical Bell pair with a noiseless reference qubit, so a logical
value traces back to the start. A merges with C, splits, then merges
with B; the decision reads v, that A-B merge's outcome, the product of
its new seam stabilizers in the first merged round (Horsman et al.
1111.4022).

For each case the script asks, over GF(2), whether some membrane of
the decision is flipped by no single fault in C's past: a
data qubit's X or Z error, or a stabilizer measurement's flip, on C
alone, before the A-B merge. If one is, no decode of C's past is needed.
The membrane reads nothing after the decision's round, since a decision
cannot wait for the future. The same question for A's past is the
control, and it must always fail.

- same: the A-C merge measures the same logical product type as the
  A-B merge (C opposite B, across a parallel seam).
- other: the A-C merge measures the other type (across a perpendicular
  seam).
- same_byproduct: as same, but the decision reads v together with the
  A-C outcome, as when that outcome sets a Pauli frame update on A that
  does not commute with v (a byproduct, as in a CNOT by lattice surgery).

Each case runs with the A-B seam vertical and again horizontal, so v is
an X-type and then a Z-type product. Stabilizers are measured by MPP, so
the noise is phenomenological; the membrane's support does not depend
on the extraction circuit.
"""

import dataclasses
import itertools
import sys
from typing import Optional

import ldpc.mod2 as mod2
import numpy
import stim

DISTANCE = 3
SEPARATE_ROUNDS = 2
MERGED_ROUNDS = 2
# noiseless shots; a parity that varies passes for fixed with chance
# 2**-SAMPLES
SAMPLES = 2048
SEED = 7
PATCHES = ("A", "B", "C")
MERGES = (("A", "C"), ("A", "B"))
CASES = ("same", "other", "same_byproduct")
SEAMS = ("vertical", "horizontal")
# the step from A to its neighbour across a seam of this direction
AXIS_BY_SEAM = {"vertical": (1, 0), "horizontal": (0, 1)}
PERPENDICULAR = {"vertical": "horizontal", "horizontal": "vertical"}
IS_C_ACROSS_BY_CASE = {"same": True, "other": False, "same_byproduct": True}
# a plaquette's type by the parity of its column plus its row
KIND_BY_PARITY = ("X", "Z")
# the type a weight-2 plaquette keeps, on a side edge or on the top and
# bottom edges
HALF_KIND_BY_IS_SIDE = {True: "Z", False: "X"}
CORNER_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
# a merge measuring this type resets its seam in the other basis and
# splits by measuring the seam there (Horsman et al. 1111.4022)
SEAM_RESET = {"X": "R", "Z": "RX"}
SEAM_SPLIT = {"X": "M", "Z": "MX"}
READOUT = {"X": "MX", "Z": "M"}
PAST_PHASES = frozenset({"before", "ac", "split", "between"})
NEED_WORDS = {True: "not needed", False: "needed"}


@dataclasses.dataclass(frozen=True)
class Layout:
    """Where the patches sit and which Stim qubit each data qubit is."""

    boxes: (
        dict  # patch name -> (first column, last column, first row, last row)
    )
    merged_boxes: dict  # pair of patch names -> box
    index: dict  # data qubit -> Stim qubit
    references: dict  # patch name -> its reference's Stim qubit
    qubits_by_patch: dict  # patch name -> frozenset of its data qubits


@dataclasses.dataclass(frozen=True)
class FaultSite:
    """One fault: a Pauli on a qubit before an operation, or a flipped MPP."""

    position: int
    kind: str  # "X", "Z", or "M" for a measurement flip
    target: Optional[int]
    owner: str
    phase: str


@dataclasses.dataclass(frozen=True)
class Merge:
    """A merge's measured type, its outcome's records, its first round's end."""

    measured_type: str
    records: tuple
    first_round_end: int


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A built circuit and the records its decision reads."""

    recording: "Recording"
    decision_type: str
    decision_records: tuple
    round_end: int


class Recording:
    """A circuit as built, each measurement's owner and each fault site."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.operations = []  # (gate, targets)
        self.measurements = []  # (phase, owner)
        self.fault_sites = []

    def append(self, gate, targets) -> None:
        """One noiseless operation."""
        self.operations.append((gate, targets))

    def measure_stabilizer(self, kind, support, phase) -> None:
        """One MPP of the stabilizer, with its flip as a fault site."""
        owner = self.owner_of(support)
        position = len(self.operations)
        site = FaultSite(position, "M", None, owner, phase)
        self.fault_sites.append(site)
        targets = self._pauli_product(kind, support)
        self.append("MPP", targets)
        self.measurements.append((phase, owner))

    def note_data_faults(self, qubit, phase) -> None:
        """An X and a Z error on the data qubit before the next operation."""
        owner = self.owner_of((qubit,))
        position = len(self.operations)
        stim_qubit = self.layout.index[qubit]
        for kind in ("X", "Z"):
            site = FaultSite(position, kind, stim_qubit, owner, phase)
            self.fault_sites.append(site)

    def note_readouts(self, count, phase, owner) -> None:
        """Measurements that are not stabilizers and carry no fault site."""
        for _ in range(count):
            self.measurements.append((phase, owner))

    def owner_of(self, support) -> str:
        """The patch holding every qubit of the support, else the seam."""
        for name, qubits in self.layout.qubits_by_patch.items():
            if qubits.issuperset(support):
                return name
        return "seam"

    def circuit(self, fault: Optional[FaultSite] = None) -> stim.Circuit:
        """The circuit, with the one fault if given."""
        built = stim.Circuit()
        for position, (gate, targets) in enumerate(self.operations):
            is_faulted = fault is not None and fault.position == position
            if not is_faulted:
                built.append(gate, targets)
                continue
            _append_faulted(built, fault, gate, targets)
        return built

    def _pauli_product(self, kind, support) -> list:
        targets = []
        for qubit in support:
            stim_qubit = self.layout.index[qubit]
            target = stim.target_pauli(stim_qubit, kind)
            targets.append(target)
            combiner = stim.target_combiner()
            targets.append(combiner)
        return targets[:-1]


def _append_faulted(built, fault, gate, targets) -> None:
    """The operation with its fault: flipped, or after a Pauli error."""
    if fault.kind == "M":
        built.append(gate, targets, 1.0)
        return
    error = f"{fault.kind}_ERROR"
    built.append(error, [fault.target], 1.0)
    built.append(gate, targets)


# ---- the code


def rectangle_stabilizers(box) -> list:
    """The rotated surface code on the box's data qubits.

    A plaquette sits at each (column + 1/2, row + 1/2), X-type when
    column + row is even; its weight-2 halves stay as Z-type on the side
    edges and as X-type on the top and bottom ones.
    """
    first_column, last_column, first_row, last_row = box
    # plaquettes reach one column and one row past the box on each side
    column_start = first_column - 1
    column_stop = last_column + 1
    row_start = first_row - 1
    row_stop = last_row + 1
    columns = range(column_start, column_stop)
    rows = range(row_start, row_stop)
    stabilizers = []
    for column, row in itertools.product(columns, rows):
        stabilizer = _plaquette(box, column, row)
        if stabilizer is not None:
            stabilizers.append(stabilizer)
    return stabilizers


def _plaquette(box, column, row):
    """The plaquette's stabilizer, or None where the boundary drops it."""
    support = _inside(box, column, row)
    parity = (column + row) % 2
    kind = KIND_BY_PARITY[parity]
    if len(support) == 4:
        return (kind, support)
    if len(support) != 2:
        return None
    is_side = column < box[0] or column >= box[1]
    if HALF_KIND_BY_IS_SIDE[is_side] != kind:
        return None
    return (kind, support)


def _inside(box, column, row) -> tuple:
    """The plaquette's corners that are data qubits of the box."""
    first_column, last_column, first_row, last_row = box
    inside = []
    for column_offset, row_offset in CORNER_OFFSETS:
        corner_column = column + column_offset
        corner_row = row + row_offset
        is_column_in = first_column <= corner_column <= last_column
        if is_column_in and first_row <= corner_row <= last_row:
            inside.append((corner_column, corner_row))
    return tuple(inside)


def _ranges(box) -> tuple:
    """The box's columns and rows."""
    first_column, last_column, first_row, last_row = box
    column_stop = last_column + 1
    row_stop = last_row + 1
    columns = range(first_column, column_stop)
    rows = range(first_row, row_stop)
    return columns, rows


def qubits_of(box) -> list:
    """The box's data qubits, as (column, row)."""
    columns, rows = _ranges(box)
    grid = itertools.product(columns, rows)
    return list(grid)


def logical(box, kind) -> tuple:
    """A shortest logical of the kind: a row or column the code keeps."""
    stabilizers = rectangle_stabilizers(box)
    for line in _lines(box):
        if _is_logical(line, kind, stabilizers):
            return line
    raise AssertionError(f"no {kind} logical on {box}")


def _lines(box) -> list:
    """Every full row and every full column of the box."""
    columns, rows = _ranges(box)
    lines = []
    for row in rows:
        line = tuple((column, row) for column in columns)
        lines.append(line)
    for column in columns:
        line = tuple((column, row) for row in rows)
        lines.append(line)
    return lines


def _is_logical(line, kind, stabilizers) -> bool:
    """Whether the line commutes with the code and is no stabilizer."""
    if not _commutes(line, kind, stabilizers):
        return False
    same_kind = [support for other, support in stabilizers if other == kind]
    return not _in_span(line, same_kind)


def _commutes(line, kind, stabilizers) -> bool:
    on_line = set(line)
    for other, support in stabilizers:
        overlap = on_line.intersection(support)
        is_odd = len(overlap) % 2 == 1
        if other != kind and is_odd:
            return False
    return True


def _in_span(line, supports) -> bool:
    """Whether the line is a product of the supports, over GF(2)."""
    qubits = set(line)
    for support in supports:
        qubits.update(support)
    ordered = sorted(qubits)
    index = {qubit: position for position, qubit in enumerate(ordered)}
    columns = [_indicator(support, index) for support in supports]
    stacked = numpy.array(columns)
    matrix = stacked.T
    target = _indicator(line, index)
    return is_solvable(matrix, target)


def _indicator(support, index) -> numpy.ndarray:
    row = numpy.zeros(len(index), dtype=numpy.uint8)
    for qubit in support:
        row[index[qubit]] = 1
    return row


# ---- GF(2), by ldpc.mod2 (a run dependency)


def is_solvable(matrix, target) -> bool:
    """Whether some x has matrix @ x = target over GF(2)."""
    column = target.reshape(-1, 1)
    augmented = numpy.hstack([matrix, column])
    return mod2.rank(matrix) == mod2.rank(augmented)


def null_space(matrix) -> numpy.ndarray:
    """A basis of {x : matrix @ x = 0} over GF(2), as columns."""
    as_bits = numpy.array(matrix, dtype=numpy.uint8) % 2
    sparse_rows = mod2.nullspace(as_bits)
    basis_rows = sparse_rows.toarray()
    return basis_rows.T.astype(numpy.uint8)


# ---- the layout


def layout_of(case, seam, distance) -> Layout:
    """Each patch's box, each merge's box, and the qubits' Stim indices."""
    origin_by_patch = _origins(case, seam, distance)
    boxes = {}
    for name, origin in origin_by_patch.items():
        boxes[name] = patch_box(origin, distance)
    merged_boxes = {}
    for first, second in MERGES:
        merged_boxes[(first, second)] = merged_box(boxes[first], boxes[second])
    # the two merged boxes cover every patch and both seams
    every = set()
    for box in merged_boxes.values():
        box_qubits = qubits_of(box)
        every.update(box_qubits)
    ordered = sorted(every)
    index = {qubit: position for position, qubit in enumerate(ordered)}
    first_reference = len(ordered)
    references = {}
    for position, name in enumerate(PATCHES):
        references[name] = first_reference + position
    qubits_by_patch = _qubits_by_patch(boxes)
    return Layout(boxes, merged_boxes, index, references, qubits_by_patch)


def _origins(case, seam, distance) -> dict:
    """Each patch's lowest data qubit; neighbours sit d + 1 apart.

    An odd d keeps the step even, so the checkerboard of stabilizer
    types runs on across a seam.
    """
    step = distance + 1
    c_seam = seam
    if not IS_C_ACROSS_BY_CASE[case]:
        c_seam = PERPENDICULAR[seam]
    toward_c = AXIS_BY_SEAM[c_seam]
    toward_b = AXIS_BY_SEAM[seam]
    back = -step
    c_origin = _scaled(toward_c, step)
    b_origin = _scaled(toward_b, back)
    return {"A": (0, 0), "B": b_origin, "C": c_origin}


def _scaled(axis, factor) -> tuple:
    return tuple(factor * part for part in axis)


def _qubits_by_patch(boxes) -> dict:
    qubits_by_patch = {}
    for name, box in boxes.items():
        qubits = qubits_of(box)
        qubits_by_patch[name] = frozenset(qubits)
    return qubits_by_patch


def patch_box(origin, distance) -> tuple:
    """The patch's (first column, last column, first row, last row)."""
    column, row = origin
    last_column = column + distance - 1
    last_row = row + distance - 1
    return (column, last_column, row, last_row)


def merged_box(first, second) -> tuple:
    """The smallest box around two patches, their seam between."""
    first_column = min(first[0], second[0])
    last_column = max(first[1], second[1])
    first_row = min(first[2], second[2])
    last_row = max(first[3], second[3])
    return (first_column, last_column, first_row, last_row)


# ---- the circuit


def build(case, seam, distance) -> Experiment:
    """Prepare, idle, merge A with C, split, idle, merge A with B, read."""
    layout = layout_of(case, seam, distance)
    recording = Recording(layout)
    _prepare(recording)
    separate = [layout.boxes[name] for name in PATCHES]
    _rounds(recording, "before", separate, SEPARATE_ROUNDS)
    earlier = _merge(recording, "ac", ("A", "C"))
    _split(recording, ("A", "C"), earlier.measured_type)
    _rounds(recording, "between", separate, 1)
    later = _merge(recording, "ab", ("A", "B"))
    _read_out(recording, later.measured_type)
    decision_records = later.records
    if case == "same_byproduct":
        decision_records = later.records + earlier.records
    return Experiment(
        recording, later.measured_type, decision_records, later.first_round_end
    )


def _prepare(recording) -> None:
    """Each patch in a logical Bell pair with its reference.

    The data start in |0> and the references in |+>; a logical CNOT
    from each reference, onto its patch's X logical, entangles them.
    """
    layout = recording.layout
    stim_qubits = layout.index.values()
    data = list(stim_qubits)
    recording.append("R", data)
    reference_qubits = layout.references.values()
    references = list(reference_qubits)
    recording.append("RX", references)
    for name, box in layout.boxes.items():
        reference = layout.references[name]
        for qubit in logical(box, "X"):
            pair = [reference, layout.index[qubit]]
            recording.append("CX", pair)


def _rounds(recording, phase, boxes, count) -> None:
    """Rounds of every box's stabilizers, each after its data faults."""
    for _ in range(count):
        for box in boxes:
            _note_box_faults(recording, box, phase)
        for box in boxes:
            _measure_box(recording, box, phase)


def _note_box_faults(recording, box, phase) -> None:
    for qubit in qubits_of(box):
        recording.note_data_faults(qubit, phase)


def _measure_box(recording, box, phase) -> None:
    for kind, support in rectangle_stabilizers(box):
        recording.measure_stabilizer(kind, support, phase)


def _merge(recording, phase, names) -> Merge:
    """Merge two patches for its rounds; the outcome is its first round's."""
    layout = recording.layout
    box = layout.merged_boxes[names]
    merged = rectangle_stabilizers(box)
    separate = _patch_stabilizers(layout, names)
    new = [stabilizer for stabilizer in merged if stabilizer not in separate]
    measured_type = _measured_type(layout, new, names)
    seam = _seam_targets(layout, names)
    reset = SEAM_RESET[measured_type]
    recording.append(reset, seam)
    start = len(recording.measurements)
    others = [layout.boxes[name] for name in PATCHES if name not in names]
    boxes = [box] + others
    _rounds(recording, phase, boxes, 1)
    first_round_end = len(recording.measurements)
    later_rounds = MERGED_ROUNDS - 1
    _rounds(recording, phase, boxes, later_rounds)
    records = _records_of(start, merged, new, measured_type)
    return Merge(measured_type, records, first_round_end)


def _patch_stabilizers(layout, names) -> list:
    stabilizers = []
    for name in names:
        box = layout.boxes[name]
        stabilizers += rectangle_stabilizers(box)
    return stabilizers


def _measured_type(layout, new, names) -> str:
    """The type whose new stabilizers multiply to the patches' logicals."""
    separate = _patch_stabilizers(layout, names)
    for kind in ("X", "Z"):
        if _multiplies_to_logicals(layout, kind, new, names, separate):
            return kind
    raise AssertionError(f"merging {names} measures no logical product")


def _multiplies_to_logicals(layout, kind, new, names, separate) -> bool:
    product = _product_of_kind(new, kind)
    for name in names:
        line = logical(layout.boxes[name], kind)
        product.symmetric_difference_update(line)
    if not product:
        return True
    same_kind = [support for other, support in separate if other == kind]
    remainder = tuple(product)
    return _in_span(remainder, same_kind)


def _product_of_kind(stabilizers, kind) -> set:
    """The support of the product of the stabilizers of the kind."""
    product = set()
    for other, support in stabilizers:
        if other == kind:
            product.symmetric_difference_update(support)
    return product


def _seam_targets(layout, names) -> list:
    """The Stim qubits of the seam between two patches."""
    box = layout.merged_boxes[names]
    merged_qubits = qubits_of(box)
    seam = set(merged_qubits)
    for name in names:
        seam -= layout.qubits_by_patch[name]
    ordered = sorted(seam)
    return [layout.index[qubit] for qubit in ordered]


def _records_of(start, merged, new, kind) -> tuple:
    """The first merged round's records of the new stabilizers of the kind."""
    records = []
    for offset, stabilizer in enumerate(merged):
        is_wanted = stabilizer in new and stabilizer[0] == kind
        if is_wanted:
            record = start + offset
            records.append(record)
    return tuple(records)


def _split(recording, names, measured_type) -> None:
    seam = _seam_targets(recording.layout, names)
    basis = SEAM_SPLIT[measured_type]
    recording.append(basis, seam)
    recording.note_readouts(len(seam), "split", "seam")


def _read_out(recording, decision_type) -> None:
    """A noiseless readout of every data qubit and reference."""
    layout = recording.layout
    basis = READOUT[decision_type]
    stim_qubits = layout.index.values()
    data = list(stim_qubits)
    recording.append(basis, data)
    recording.note_readouts(len(data), "final", "data")
    reference_qubits = layout.references.values()
    references = list(reference_qubits)
    recording.append(basis, references)
    recording.note_readouts(len(references), "final", "reference")


# ---- the membranes


def flips(recording, site) -> numpy.ndarray:
    """The measurements one fault flips, against the noiseless run."""
    faulty = recording.circuit(site)
    simulator = stim.FlipSimulator(
        batch_size=1, disable_stabilizer_randomization=True
    )
    simulator.do(faulty)
    flipped = simulator.get_measurement_flips()
    shot = flipped[:, 0]
    return shot.astype(numpy.uint8)


def known_by_decision(experiment) -> list:
    """The measurements a decision may read: none after its round.

    The references are noiseless and untouched, so their readout only
    fixes each patch's starting logical value, whenever it is read.
    """
    recording = experiment.recording
    total = len(recording.measurements)
    reference_count = len(recording.layout.references)
    first_reference = total - reference_count
    references = list(range(first_reference, total))
    before = list(range(experiment.round_end))
    return before + references


def fixed_parities(recording, columns) -> numpy.ndarray:
    """Every parity of those measurements the noiseless circuit fixes."""
    noiseless = recording.circuit()
    sampler = noiseless.compile_sampler(seed=SEED)
    shots = sampler.sample(SAMPLES)
    known = shots[:, columns]
    varying = known ^ known[0]
    return null_space(varying)


def has_membrane_avoiding(experiment, owner) -> bool:
    """Whether a membrane of the decision avoids the owner's past faults.

    The membrane is a fixed parity holding every record the decision
    reads, and no single fault of the owner's before the decision may
    flip it; single faults span every fault, so none at all does.
    """
    recording = experiment.recording
    columns = known_by_decision(experiment)
    sites = _past_sites(recording, owner)
    flip_matrix = _flip_matrix(recording, sites, columns)
    parities = fixed_parities(recording, columns)
    is_read = numpy.isin(columns, experiment.decision_records)
    decision_rows = numpy.flatnonzero(is_read)
    holds_decision = parities[decision_rows]
    flipped = flip_matrix @ parities
    flipped_parities = flipped % 2
    conditions = numpy.vstack([holds_decision, flipped_parities])
    ones = numpy.ones(len(decision_rows), dtype=numpy.uint8)
    zeros = numpy.zeros(len(sites), dtype=numpy.uint8)
    wanted = numpy.concatenate([ones, zeros])
    return is_solvable(conditions, wanted)


def _past_sites(recording, owner) -> list:
    sites = []
    for site in recording.fault_sites:
        if site.owner == owner and site.phase in PAST_PHASES:
            sites.append(site)
    return sites


def _flip_matrix(recording, sites, columns) -> numpy.ndarray:
    rows = []
    for site in sites:
        flipped = flips(recording, site)
        rows.append(flipped[columns])
    return numpy.array(rows)


def main(distance: int) -> None:
    """Print, per case and seam, whether C's past and A's are needed."""
    for case, seam in itertools.product(CASES, SEAMS):
        experiment = build(case, seam, distance)
        is_c_free = has_membrane_avoiding(experiment, "C")
        is_a_free = has_membrane_avoiding(experiment, "A")
        c_words = NEED_WORDS[is_c_free]
        a_words = NEED_WORDS[is_a_free]
        print(
            f"d={distance} {case:15s} {seam:10s} "
            f"v type {experiment.decision_type}: "
            f"C's past {c_words}; A's past {a_words}"
        )


if __name__ == "__main__":
    code_distance = DISTANCE
    if len(sys.argv) > 1:
        code_distance = int(sys.argv[1])
    main(code_distance)
