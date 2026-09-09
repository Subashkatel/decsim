"""The pinned strong input beside the syndrome Bombin's rule defines.

The rule, 2303.04846 lines 775-788: "instead of the syndrome de which
only includes the effect of the physical errors e, the input instance to
decoder j will consist of the syndrome d(e + kappa_Pj), where kappa_Pj
:= sum over i<j of kappa_i includes the corrections committed by all
prior decoders". Written with the detector flip matrix H of the circuit's
detector error model, the input of a strong window whose face is pinned
is raw XOR (H kappa) restricted to that window's own detectors.

The oracle is built outside decsim, from Stim and qLDPC only: the same
rotated surface code memory circuit, its detector error model, and
qLDPC's DetectorErrorModelArrays.detector_flip_matrix (as installed,
qldpc/decoders/dems.py) as H. The committed correction comes off the run
as the fault columns the neighbour's decode committed, and each of those
columns is matched to a column of the external H by the detectors it
flips, so a fault decsim commits that Stim's model does not carry fails
the test before the arithmetic begins.

Tan et al. 2209.09219 lines 943-946 say which detectors change: "when
the next window starts from the (s + 1)-th layer of the current window,
only this layer is updated", so a near pin changes the strong window's
oldest read layer and nothing else, and a far pin its newest.

Bombin's consistency condition, lines 748-749, is
d(kappa_i + kappa_j + e)(sigma) = 0 on the checks two tasks share. Two
exact statements of it hold here: the neighbour and the strong window
commit no fault in common (lines 703-704, each task commits the
restriction to its own commit region), and every seam detector the pair
leaves lit is one an unowned column of the strong window's model
explains, which is decsim's fault-ownership partition and not a second
explanation of the same detector. Measured rates of exact annihilation,
and the same rates on the shipped weak chain, are in design audit
note 21.
"""

import copy
import dataclasses
import pathlib

import numpy
import qldpc.decoders.dems as qldpc_dems
import stim

import decsim.machine as machine_module
import decsim.records.decoding as decoding_records
import decsim.settings as machine_settings
import decsim.windows.decode_requests as decode_requests
import decsim.windows.window_boundaries as window_boundaries
import decsim.windows.window_commits as window_commits
import tests.escalation.test_strong_window_shapes as shape_tests

# One shot per distance, at a round count that is no multiple of the
# stride, on seeds whose run has a nonempty pinned seam (the seeds the
# researcher's runs used, 11 and its prefix, are deliberately not among
# them).
NEAR_CASES = ((3, 14, 5), (5, 23, 3), (7, 32, 1))
FORWARD_CASES = ((3, 14, 1), (5, 23, 2), (7, 32, 1))
PHYSICAL_ERROR_PROBABILITY = 0.008


def _machine(distance: int, rounds: int, seed: int, strong_window: str):
    """The gate's switching card at that distance, on the named row."""
    sections = copy.deepcopy(shape_tests.GATE_SWITCHING_CARD)
    sections["escalation"]["strong_window"] = strong_window
    sections["workload"]["rounds_per_shot"] = rounds
    base_directory = pathlib.Path(".")
    settings = machine_settings.MachineSettings.from_mapping(
        sections, name="seam_pinned", base_directory=base_directory
    )
    qpu = dataclasses.replace(
        settings.qpu, distance=distance, round_period_microseconds=1.0
    )
    workload = dataclasses.replace(
        settings.workload,
        physical_error_probability=PHYSICAL_ERROR_PROBABILITY,
    )
    settings = dataclasses.replace(settings, qpu=qpu, workload=workload)
    return machine_module.Machine.build(settings, seed)


def _external_flip_matrix(distance: int, rounds: int):
    """(H, column by detector set) of the circuit's model, built by qLDPC."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=PHYSICAL_ERROR_PROBABILITY,
        before_measure_flip_probability=PHYSICAL_ERROR_PROBABILITY,
        after_reset_flip_probability=PHYSICAL_ERROR_PROBABILITY,
        before_round_data_depolarization=PHYSICAL_ERROR_PROBABILITY,
    )
    model = circuit.detector_error_model(decompose_errors=True)
    flattened = model.flattened()
    arrays = qldpc_dems.DetectorErrorModelArrays(
        flattened, decompose_errors=True
    )
    flip = arrays.detector_flip_matrix.tocsc()
    return flip, _column_by_detectors(flip)


def _column_by_detectors(flip) -> dict:
    """Which column of H flips exactly that set of detectors."""
    column_by_detectors = {}
    for column in range(flip.shape[1]):
        start = flip.indptr[column]
        stop = flip.indptr[column + 1]
        rows = flip.indices[start:stop]
        detectors = frozenset(int(row) for row in rows)
        column_by_detectors.setdefault(detectors, column)
    return column_by_detectors


class _Capture:
    """What one run of a pinned row shows: its results, pins and inputs.

    The three seams it reads are the two the committer answers on
    (accept_result, accept_strong_result), the courier's pin, and the
    gate that folds the boundary into the landed input.
    """

    def __init__(self) -> None:
        self.result_by_window = {}
        self.pins = []
        self.masked = []

    def install(self, monkeypatch) -> None:
        """Watch the four calls; monkeypatch undoes it after the test.

        The patches are plain functions, so each call still arrives with
        the component it was made on.
        """
        capture = self

        def weak_result(verdict, job, result) -> None:
            capture.note_result((job.operation_id, job.window_id), job, result)
            _WEAK_RESULT(verdict, job, result)

        def strong_result(verdict, job, result) -> None:
            key = (job.request_key.operation_id, job.request_key.window_id)
            capture.note_result(key, job, result)
            _STRONG_RESULT(verdict, job, result)

        def pin(courier, source_key, destination, model, operation, key):
            capture.note_pin(source_key, destination.key)
            _PIN(courier, source_key, destination, model, operation, key)

        def mask(gate, job) -> None:
            raw = _rounds_snapshot(job.decoder_input)
            _MASK(gate, job)
            masked = _rounds_snapshot(job.decoder_input)
            capture.note_input(job, raw, masked)

        verdict_class = window_commits.WindowVerdict
        monkeypatch.setattr(verdict_class, "accept_result", weak_result)
        monkeypatch.setattr(
            verdict_class, "accept_strong_result", strong_result
        )
        courier_class = window_boundaries.BoundaryCourier
        monkeypatch.setattr(courier_class, "pin_strong_face", pin)
        gate_class = decode_requests.WindowInputGate
        monkeypatch.setattr(gate_class, "mask_input", mask)

    def note_result(self, key: tuple, job, result) -> None:
        """The window's latest answer, whichever tier gave it."""
        self.result_by_window[key] = (job, result)

    def note_pin(self, source_key: tuple, destination_key: tuple) -> None:
        """One face pinned, with the result whose boundary it pins on.

        A window that is decoded again ships a second boundary later, so
        the pin holds the result the courier had committed at the moment
        the face was pinned.
        """
        committed = self.result_by_window[source_key]
        self.pins.append((source_key, destination_key, committed))

    def note_input(self, job, raw: tuple, masked: tuple) -> None:
        """One job's input before and after the fold."""
        self.masked.append((job, raw, masked))

    def pinned_jobs(self) -> list:
        """(job, raw input, masked input, pinned sources) per strong job."""
        pinned = []
        for job, raw, masked in self.masked:
            if job.kind is not decoding_records.DecodeJobKind.STRONG_REDECODE:
                continue
            sources = self._sources_for(job)
            if not sources:
                continue
            pinned.append((job, raw, masked, sources))
        return pinned

    def _sources_for(self, job) -> list:
        sources = []
        for source_key, destination_key, committed in self.pins:
            if destination_key == (job.operation_id, job.window_id):
                sources.append((source_key, committed))
        return sources


_WEAK_RESULT = window_commits.WindowVerdict.accept_result
_STRONG_RESULT = window_commits.WindowVerdict.accept_strong_result
_PIN = window_boundaries.BoundaryCourier.pin_strong_face
_MASK = decode_requests.WindowInputGate.mask_input


def _rounds_snapshot(decoder_input) -> tuple:
    """(round index, bits) of every fragment of a job's input."""
    if decoder_input is None:
        return ()
    snapshot = []
    for round_input in decoder_input.rounds:
        for fragment in round_input.fragments:
            bits = fragment.bits or ()
            snapshot.append((round_input.round_index, tuple(bits)))
    return tuple(snapshot)


def _detector_bits(job, snapshot: tuple) -> dict:
    """The input's bit for every detector row of the job's window model."""
    model = job.detector_error_model
    positions = model.defect_positions
    bit_by_place = {}
    for round_index, bits in snapshot:
        for position, bit in enumerate(bits):
            bit_by_place[(round_index, position)] = int(bit)
    bit_by_detector = {}
    for detector_id in model.detector_ids:
        place = positions[detector_id]
        if place in bit_by_place:
            bit_by_detector[detector_id] = bit_by_place[place]
    return bit_by_detector


def _external_correction(job, result, flip, column_by_detectors):
    """The committed correction as a vector over the external columns."""
    placed = _placed_faults(job.detector_error_model)
    correction = numpy.asarray(result.correction, dtype=numpy.uint8)
    nonzero = numpy.nonzero(correction)
    committed = nonzero[0]
    column_count = flip.shape[1]
    kappa = numpy.zeros(column_count, dtype=numpy.uint8)
    for column in committed:
        detectors = frozenset(placed.boundary_flips[int(column)])
        external = column_by_detectors[detectors]
        kappa[external] ^= 1
    return kappa


def _placed_faults(model):
    """The fault columns the decode ran on."""
    if model.graphlike_faults is not None:
        return model.graphlike_faults
    return model.physical_faults


def _flipped_detectors(flip, kappa) -> set:
    """The detectors H kappa flips, anywhere in the circuit."""
    product = flip @ kappa
    reduced = product % 2
    dense = numpy.asarray(reduced)
    flips = dense.ravel()
    nonzero = numpy.nonzero(flips)
    return set(int(detector) for detector in nonzero[0])


# One run and one external model per (row, case), kept: every test below
# reads the same shot, the way tests/links/test_data_through.py keeps its
# runs. A real decode of ten distance rounds at d = 7 is seconds.
_CASE_RUNS: dict = {}


def _run_case(monkeypatch, case: tuple, strong_window: str) -> tuple:
    """One shot of the row, with its capture and its external oracle."""
    kept = _CASE_RUNS.get((strong_window, case))
    if kept is not None:
        return kept
    distance, rounds, seed = case
    capture = _Capture()
    capture.install(monkeypatch)
    machine = _machine(distance, rounds, seed, strong_window)
    machine.run()
    flip, column_by_detectors = _external_flip_matrix(distance, rounds)
    run = (capture, flip, column_by_detectors)
    _CASE_RUNS[(strong_window, case)] = run
    return run


def _pinned_flips(sources, flip, column_by_detectors) -> set:
    """The detectors the pinned neighbours' committed corrections flip."""
    flipped = set()
    for _source_key, committed in sources:
        source_job, source_result = committed
        kappa = _external_correction(
            source_job, source_result, flip, column_by_detectors
        )
        flipped |= _flipped_detectors(flip, kappa)
    return flipped


def _layers_of(job, detector_ids) -> set:
    """The round layers those detectors sit on."""
    positions = job.detector_error_model.defect_positions
    layers = set()
    for detector_id in detector_ids:
        layers.add(positions[detector_id][0])
    return layers


def _differing(raw: dict, masked: dict) -> list:
    """The detectors whose input bit the fold changed."""
    changed = []
    for detector_id, bit in masked.items():
        if raw[detector_id] != bit:
            changed.append(detector_id)
    return changed


def test_the_near_pinned_input_is_the_syndrome_bombin_defines(monkeypatch):
    """The raw input XOR (H kappa) on its own detectors, at d 3, 5, 7.

    The oracle's H is qLDPC's detector flip matrix of the same circuit's
    model; kappa is what the pinned neighbour committed on the run. Only
    the strong window's oldest read layer differs from raw (Tan
    2209.09219 lines 943-946), and at least one job of the three runs
    has a nonempty seam, so a row that silently stopped folding fails.
    """
    nonempty_jobs = 0
    for case in NEAR_CASES:
        capture, flip, columns = _run_case(
            monkeypatch, case, "near_seam_pinned"
        )
        for job, raw_input, masked_input, sources in capture.pinned_jobs():
            raw = _detector_bits(job, raw_input)
            masked = _detector_bits(job, masked_input)
            flipped = _pinned_flips(sources, flip, columns)
            expected = _expected_input(raw, flipped)
            assert masked == expected
            changed = _differing(raw, masked)
            assert _layers_of(job, changed) <= {job.window.start_round}
            nonempty_jobs += min(1, len(changed))
    assert nonempty_jobs


def _expected_input(raw: dict, flipped: set) -> dict:
    """Bombin's input: the raw bit, flipped where H kappa says so."""
    expected = {}
    for detector_id, bit in raw.items():
        is_flipped = detector_id in flipped
        expected[detector_id] = bit ^ int(is_flipped)
    return expected


def test_the_forward_pinned_input_differs_on_its_two_seam_layers(monkeypatch):
    """Both faces pinned: the oldest and the newest read layer, at d 3, 5, 7.

    The far face is the window that restarts the weak chain, whose
    committed correction reaches back one layer and no further, which is
    the same one-layer rule read in the other direction (Tan
    2209.09219 lines 943-946 and 1026-1027).
    """
    nonempty_jobs = 0
    for case in FORWARD_CASES:
        capture, flip, columns = _run_case(
            monkeypatch, case, "forward_seam_pinned"
        )
        for job, raw_input, masked_input, sources in capture.pinned_jobs():
            raw = _detector_bits(job, raw_input)
            masked = _detector_bits(job, masked_input)
            flipped = _pinned_flips(sources, flip, columns)
            expected = _expected_input(raw, flipped)
            assert masked == expected
            changed = _differing(raw, masked)
            edges = {job.window.start_round, job.window.buffer_hi}
            assert _layers_of(job, changed) <= edges
            nonempty_jobs += min(1, len(changed))
    assert nonempty_jobs


def test_a_pinned_pair_commits_no_fault_twice(monkeypatch):
    """Bombin 2303.04846 lines 703-704, on the run's own corrections.

    Only the restriction of a task's estimate to its commit region is
    used, so the neighbour a face is pinned on and the strong window
    that reads it commit disjoint sets of faults: no detector is
    explained twice.
    """
    shared = []
    for case in NEAR_CASES:
        run = _run_case(monkeypatch, case, "near_seam_pinned")
        counts = _shared_fault_counts(run)
        shared.extend(counts)
    assert shared
    assert set(shared) == {0}


def _shared_fault_counts(run: tuple) -> list:
    """How many faults each pinned pair of the run committed in common."""
    capture, flip, columns = run
    counts = []
    for job, _raw, _masked, sources in capture.pinned_jobs():
        strong = _strong_kappa(capture, job, flip, columns)
        if strong is None:
            continue
        neighbour = _neighbour_kappa(sources, flip, columns)
        both = neighbour & strong
        shared = numpy.count_nonzero(both)
        counts.append(int(shared))
    return counts


def test_a_lit_seam_detector_is_one_the_window_does_not_own(monkeypatch):
    """Bombin's consistency condition, and the one way decsim escapes it.

    d(kappa_i + kappa_j + e) = 0 on the shared checks (lines 748-749)
    holds on the seam layer unless the strong decode explained a
    detector with a fault it does not own, which its own model still
    carries as a column and which its correction therefore drops
    (window_placement.py: an excluded fault is never owned). Every seam
    detector the pair leaves lit is such a detector, on the pinned rows
    and on the shipped weak chain alike (design audit note 21).
    """
    unexplained = []
    for case in NEAR_CASES:
        run = _run_case(monkeypatch, case, "near_seam_pinned")
        counts = _lit_seams_with_no_unowned_explanation(run)
        unexplained.extend(counts)
    assert unexplained
    assert set(unexplained) == {0}


def _lit_seams_with_no_unowned_explanation(run: tuple) -> list:
    """Per pinned job, the lit seam detectors no unowned column flips."""
    capture, flip, columns = run
    counts = []
    for job, raw_input, _masked, sources in capture.pinned_jobs():
        strong = _strong_kappa(capture, job, flip, columns)
        if strong is None:
            continue
        neighbour = _neighbour_kappa(sources, flip, columns)
        both = (neighbour + strong) % 2
        flipped = _flipped_detectors(flip, both)
        raw = _detector_bits(job, raw_input)
        lit = _lit_seam_detectors(job, raw, flipped)
        owned_only = _owned_only_count(job, lit)
        counts.append(owned_only)
    return counts


def _strong_kappa(capture, job, flip, columns):
    """The strong window's own committed correction, or None."""
    key = (job.operation_id, job.window_id)
    strong = capture.result_by_window.get(key)
    if strong is None:
        return None
    strong_job, strong_result = strong
    if strong_result.correction is None:
        return None
    return _external_correction(strong_job, strong_result, flip, columns)


def _neighbour_kappa(sources, flip, columns):
    """The pinned neighbours' committed corrections, together."""
    column_count = flip.shape[1]
    kappa = numpy.zeros(column_count, dtype=numpy.uint8)
    for _source_key, committed in sources:
        source_job, source_result = committed
        contribution = _external_correction(
            source_job, source_result, flip, columns
        )
        kappa = (kappa + contribution) % 2
    return kappa


def _lit_seam_detectors(job, raw: dict, flipped: set) -> list:
    """The seam layer's detectors the two corrections leave lit."""
    positions = job.detector_error_model.defect_positions
    seam_round = job.window.start_round
    lit = []
    for detector_id, bit in raw.items():
        place = positions[detector_id]
        if place[0] != seam_round:
            continue
        is_flipped = detector_id in flipped
        residual = bit ^ int(is_flipped)
        if residual:
            lit.append(detector_id)
    return lit


def _owned_only_count(job, lit: list) -> int:
    """How many lit detectors no unowned column of the model flips."""
    placed = _placed_faults(job.detector_error_model)
    owned = numpy.asarray(placed.owned, dtype=bool)
    check = placed.check.tocsr()
    row_of = _row_index(job)
    owned_only = 0
    for detector_id in lit:
        row = check.getrow(row_of[detector_id])
        columns = row.indices
        unowned = numpy.count_nonzero(~owned[columns])
        owned_only += int(not unowned)
    return owned_only


def _row_index(job) -> dict:
    """Which row of the check matrix each detector is."""
    row_of = {}
    for index, detector_id in enumerate(job.detector_error_model.detector_ids):
        row_of[detector_id] = index
    return row_of
