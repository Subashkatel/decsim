"""decsim's per-window fault ownership beside qLDPC's, fault by fault.

This is the projection the 2026-08-22 window validation harness was
built around (archive/2026-09-09-window-harness/harness/references.py,
in the sandbox: "A fault is
identified everywhere by its signature (frozenset of detectors,
frozenset of observables), never by column index"). Two implementations
number their columns differently, so a column index proves nothing; the
signature is the same object in both, and the sets of signatures a
window commits can be compared element by element.

The reference is qLDPC's SlidingWindowDecoder
(qldpc/decoders/sinter.py, class SlidingWindowDecoder and
SequentialWindowDecoder.compile_decoder_for_dem). Its commit rule is
`c_errors = detector_flip_matrix[c_detectors].getnnz(axis=0) != 0` with
`c_errors[addressed_errors] = False`, which is decsim's rule in
decsim/detector_error_model/window_placement.py: a window owns the
faults touching its commit rounds that no earlier window owned. The
last window absorbs the remainder in both.

The installed qldpc is byte-identical to the on-disk clone at
/scratch/gpfs/MARTONOSI/sk2415/qlx-qec-sandbox/tmp/reference-decoders/qLDPC,
so the reference here is the same code the harness ran.

These are referent tests and each builds a real Stim circuit and a real
detector error model at d=3, so they cost seconds rather than
milliseconds. Nothing here decodes: the harness's own reading was that
every per-shot difference against qLDPC is a minimum-weight matching
tie broken by a different column order (VALIDATION.md, "Every
per-window difference against qldpc / the reference loop has equal
total matching weight"), so a shot comparison pins the tie-breaking of
PyMatching and not the window rule. The window rule is what this file
pins, and it is exact.
"""

import numpy
import pytest

import decsim.detector_error_model.detector_chronology as chronology
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.detector_error_model.stim_fault_catalog as fault_catalogs
import decsim.detector_error_model.window_model_builders as builders
import decsim.windows.schemes.sliding as sliding_scheme

stim = pytest.importorskip("stim")

PHYSICAL_ERROR = 0.001
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE


def _circuit(distance, rounds):
    """One real rotated surface code memory, uniform noise."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=PHYSICAL_ERROR,
        before_measure_flip_probability=PHYSICAL_ERROR,
        after_reset_flip_probability=PHYSICAL_ERROR,
        before_round_data_depolarization=PHYSICAL_ERROR,
    )


def _graphlike_catalog(circuit):
    """The whole-circuit graphlike fault catalog decsim builds."""
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    catalogs, _link = fault_catalogs.prepare_fault_catalogs(
        circuit, requirement
    )
    return catalogs[GRAPHLIKE]


def _signature(catalog, fault_id):
    """A fault's identity: which detectors it flips, which observables."""
    detectors = frozenset(catalog.detector_sets[fault_id])
    observables = frozenset(catalog.observable_sets[fault_id])
    return (detectors, observables)


def _catalog_detector_error_model(catalog):
    """The catalog written back out as a Stim model, one error per fault."""
    model = stim.DetectorErrorModel()
    for fault_id in range(len(catalog.priors)):
        detectors = sorted(catalog.detector_sets[fault_id])
        observables = sorted(catalog.observable_sets[fault_id])
        targets = []
        for detector in detectors:
            target = stim.target_relative_detector_id(detector)
            targets.append(target)
        for observable in observables:
            target = stim.target_logical_observable_id(observable)
            targets.append(target)
        prior = catalog.priors[fault_id]
        model.append("error", prior, targets)
    return model


def _plan_entries(rounds, commit, buffer_rounds):
    """The sliding plan as the builder wants it, four bounds per window."""
    scheme = sliding_scheme.SlidingWindowScheme()
    plan = scheme.plan_operation(
        1,
        rounds,
        commit_round_count=commit,
        buffer_round_count=buffer_rounds,
    )
    entries = []
    for geometry in plan.windows:
        entries.append(
            (
                geometry.buffer_lo,
                geometry.commit_lo,
                geometry.commit_hi,
                geometry.buffer_hi,
            )
        )
    return entries


def _our_window_models(circuit, rounds, entries):
    """The window models decsim's builder makes for that plan."""
    requirement = fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    return builders.build_window_error_models(
        circuit,
        entries,
        round_count=rounds,
        fault_model_requirement=requirement,
        fault_exclusion_ranges=(),
    )


def _owned_signatures(catalog, model):
    """The identities of the faults one decsim window commits."""
    placed = model.graphlike_faults
    owned = _flat_boolean(placed.owned)
    source_fault_ids = placed.source_fault_ids
    signatures = set()
    for column, fault_id in enumerate(source_fault_ids):
        if not owned[column]:
            continue
        signature = _signature(catalog, fault_id)
        signatures.add(signature)
    return signatures


def _flat_boolean(mask):
    """A column mask of any shape read as one flat boolean row."""
    array = numpy.asarray(mask)
    flat = array.ravel()
    return flat.astype(bool)


def _compiled_reference(circuit, catalog, rounds, commit, buffer_rounds):
    """The reference SlidingWindowDecoder compiled on the same catalog."""
    qldpc_sinter = pytest.importorskip("qldpc.decoders.sinter")
    round_of_detector = chronology.resolve_detector_rounds(
        circuit, None, rounds
    )

    def time_of_detector(detector):
        return int(round_of_detector[detector])

    window_size = commit + buffer_rounds
    decoder = qldpc_sinter.SlidingWindowDecoder(
        window_size=window_size,
        stride=commit,
        detector_to_time=time_of_detector,
        simplify=False,
        decompose_errors=False,
        with_MWPM=True,
    )
    model = _catalog_detector_error_model(catalog)
    return decoder.compile_decoder_for_dem(model)


def _reference_owned_signatures(catalog, compiled):
    """The identities of the faults each qLDPC window commits, in order."""
    per_window = []
    for commit_mask, _in_detection_region in compiled.window_errors:
        mask = _flat_boolean(commit_mask)
        signatures = set()
        for fault_id in numpy.flatnonzero(mask):
            signature = _signature(catalog, int(fault_id))
            signatures.add(signature)
        per_window.append(signatures)
    return per_window


def _both_sides(distance, rounds, commit, buffer_rounds):
    """The owned identities per window, ours first and the reference's next."""
    circuit = _circuit(distance, rounds)
    catalog = _graphlike_catalog(circuit)
    entries = _plan_entries(rounds, commit, buffer_rounds)
    models = _our_window_models(circuit, rounds, entries)
    ours = []
    for model in models:
        signatures = _owned_signatures(catalog, model)
        ours.append(signatures)
    compiled = _compiled_reference(
        circuit, catalog, rounds, commit, buffer_rounds
    )
    theirs = _reference_owned_signatures(catalog, compiled)
    return ours, theirs


def test_every_window_commits_the_faults_qldpc_commits_at_d3():
    """d=3, 12 rounds, commit 3 buffer 3: an exact multiple of the stride."""
    ours, theirs = _both_sides(3, 12, 3, 3)
    assert len(ours) == 3
    assert ours == theirs


def test_the_absorbing_last_window_commits_what_qldpc_commits():
    """14 rounds under stride 3: the last window takes the remainder."""
    ours, theirs = _both_sides(3, 14, 3, 3)
    assert len(ours) == 3
    assert ours == theirs


def test_an_asymmetric_commit_and_buffer_projects_the_same_way():
    """Commit 2 with buffer 4: the window is wider than twice its stride."""
    ours, theirs = _both_sides(3, 14, 2, 4)
    assert ours == theirs


def test_the_window_rows_are_qldpcs_detection_regions():
    """The rows a window reads are the detectors of its buffer rounds.

    qLDPC keeps the detection region as a detector list per window
    (CompiledSequentialWindowDecoder.window_detectors); decsim keeps it
    as WindowErrorModel.detector_ids. Same detectors, same windows.
    """
    circuit = _circuit(3, 12)
    catalog = _graphlike_catalog(circuit)
    entries = _plan_entries(12, 3, 3)
    models = _our_window_models(circuit, 12, entries)
    compiled = _compiled_reference(circuit, catalog, 12, 3, 3)
    ours = []
    for model in models:
        rows = set(model.detector_ids)
        ours.append(rows)
    theirs = []
    for detectors in compiled.window_detectors:
        rows = {int(detector) for detector in detectors}
        theirs.append(rows)
    assert ours == theirs


def test_the_projection_gives_every_catalogued_fault_exactly_one_owner():
    """Skoric 2209.08552 section I.B: the commit regions partition the faults.

    Ownership is what makes a windowed decode add up to one correction:
    every fault of the whole-circuit catalog is committed by exactly one
    window, so no fault is corrected twice and none is dropped.
    """
    circuit = _circuit(3, 12)
    catalog = _graphlike_catalog(circuit)
    entries = _plan_entries(12, 3, 3)
    models = _our_window_models(circuit, 12, entries)
    seen = []
    for model in models:
        signatures = _owned_signatures(catalog, model)
        ordered = sorted(signatures)
        seen.extend(ordered)
    committed = set(seen)
    assert len(seen) == len(committed)
    catalogued = set()
    for fault_id in range(len(catalog.priors)):
        signature = _signature(catalog, fault_id)
        catalogued.add(signature)
    assert committed == catalogued
