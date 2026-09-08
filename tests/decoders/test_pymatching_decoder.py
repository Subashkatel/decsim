"""The pymatching row against PyMatching itself and sinter's pymatching row.

One d=3 memory circuit as a single window: decsim's row, a
pymatching.Matching built directly from the window's graph, and sinter's
pymatching row on the whole-circuit detector error model
(.pydeps/sinter/_decoding/_decoding_pymatching.py) predict the same
observable per shot, or tie at the same weight. The hand-written graphs
below check the row's own contract where the circuit cannot reach it: a
boundaryless (toric-like) component, which PyMatching refuses to match
at odd parity, and parallel fault columns, which Stim's detector error
models and PyMatching's model loader merge as independent errors.
"""

import numpy
import pymatching
import pytest
import sinter

import decsim.decoders.decoder as decoder_module
import decsim.decoders.minimum_weight_perfect_matching.decoder as adapter
import decsim.decoders.minimum_weight_perfect_matching.weights as weights
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
from tests.decoders import windows

ROUNDS = 3
SHOTS = 100
GRAPHLIKE = fault_models.FaultRepresentation.GRAPHLIKE


def _window_and_shots():
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.GRAPHLIKE_FAULT_MODEL_REQUIRED
    )
    detection_events, observables = windows.sampled_shots(circuit, SHOTS, 5)
    return circuit, model, detection_events, observables


def placed_faults(check, priors, observables):
    """One window's graphlike faults from a hand-written check matrix."""
    matrix = numpy.asarray(check, dtype=numpy.uint8)
    prior_array = numpy.asarray(priors, dtype=float)
    observable_matrix = numpy.asarray(observables, dtype=numpy.uint8)
    fault_count = matrix.shape[1]
    owned = numpy.ones(fault_count, dtype=bool)
    source_fault_ids = tuple(range(fault_count))
    return fault_models.PlacedFaultModel(
        representation=GRAPHLIKE,
        check=matrix,
        priors=prior_array,
        observables=observable_matrix,
        owned=owned,
        source_fault_ids=source_fault_ids,
        boundary_flips={},
    )


def window_of(faults, detector_count):
    """A one-window model over the detector rows in row order."""
    detector_ids = tuple(range(detector_count))
    defect_positions = {}
    for detector_id in detector_ids:
        defect_positions[detector_id] = (1, detector_id)
    return fault_models.WindowErrorModel(
        detector_ids=detector_ids,
        detector_coordinates=None,
        defect_positions=defect_positions,
        graphlike_faults=faults,
        physical_faults=None,
        physical_to_graphlike_detector_projection=None,
    )


def _direct_matching(faults):
    check = faults.check.copy()
    edge_weights = weights.matching_weights(faults.priors)
    error_probabilities = weights.finite_priors(faults.priors)
    return pymatching.Matching.from_check_matrix(
        check,
        weights=edge_weights,
        error_probabilities=error_probabilities,
        merge_strategy="independent",
    )


def test_the_row_matches_pymatching_on_the_same_graph():
    _, model, detection_events, _ = _window_and_shots()
    faults = model.require_faults(GRAPHLIKE)
    row = adapter.PyMatchingDecoder()
    direct = _direct_matching(faults)
    for shot in detection_events:
        job = windows.job_for(model, shot)
        result = row.decode(job)
        syndrome = windows.row_syndrome(model, shot)
        expected = direct.decode(syndrome)
        assert result.correction.tolist() == expected.tolist()


def test_the_row_predicts_what_sinters_pymatching_row_predicts():
    circuit, model, detection_events, _ = _window_and_shots()
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    sinter_row = sinter.BUILT_IN_DECODERS["pymatching"]
    compiled = sinter_row.compile_decoder_for_dem(dem=detector_error_model)
    packed = numpy.packbits(detection_events, axis=1, bitorder="little")
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    predictions = numpy.unpackbits(predictions, axis=1, bitorder="little")
    whole = pymatching.Matching.from_detector_error_model(detector_error_model)
    row = adapter.PyMatchingDecoder()
    faults = model.require_faults(GRAPHLIKE)
    graphs = row.compiled_for(faults, model)
    matching = graphs.plain
    ties = 0
    for shot, predicted in zip(detection_events, predictions):
        job = windows.job_for(model, shot)
        result = row.decode(job)
        prediction = int(predicted[0])
        if result.logical_observables == (prediction,):
            continue
        # the two graphs may pick different minimum-weight matchings of
        # the same weight; that is a tie, not a disagreement
        syndrome = windows.row_syndrome(model, shot)
        _, row_weight = matching.decode(syndrome, return_weight=True)
        _, whole_weight = whole.decode(shot, return_weight=True)
        difference = row_weight - whole_weight
        weight_gap = abs(difference)
        assert weight_gap < 1e-6
        ties += 1
    assert ties < SHOTS / 10


def test_an_unmatchable_syndrome_is_reported_and_not_raised():
    """PyMatching refuses odd parity in a boundaryless component.

    Matching.decode raises ValueError there rather than answering,
    because no perfect matching exists. No plan produces such a
    syndrome, so the row keeps the run going: an all-zero correction
    marked INVALID_CORRECTION says the answer is not trustworthy
    without stopping the machine (backend_outcome.py's policy).
    """
    # two boundaryless edges, (0, 2) and (1, 3): one defect on each
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    faults = placed_faults(check, [0.1, 0.1], [[0, 0]])
    model = window_of(faults, 4)
    syndrome = numpy.asarray([1, 1, 0, 0], dtype=numpy.uint8)
    job = windows.job_for(model, syndrome)
    row = adapter.PyMatchingDecoder()
    result = row.decode(job)
    invalid = decoder_module.BackendDecodeStatus.INVALID_CORRECTION
    assert result.decode_status is invalid
    assert result.correction.tolist() == [0, 0]


def test_a_satisfiable_syndrome_on_the_same_graph_has_no_status():
    """Both defects on one boundaryless edge are that edge's fault."""
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    faults = placed_faults(check, [0.1, 0.1], [[0, 0]])
    model = window_of(faults, 4)
    syndrome = numpy.asarray([1, 0, 1, 0], dtype=numpy.uint8)
    job = windows.job_for(model, syndrome)
    row = adapter.PyMatchingDecoder()
    result = row.decode(job)
    assert result.decode_status is None
    assert result.correction.tolist() == [1, 0]


def test_the_warm_up_survives_a_boundaryless_component():
    """The compile step warms each column on its own detector set.

    PyMatching builds its graph lazily and finishes warming only once
    it has matched real defects, so compile decodes a few syndromes
    before the first timed call. An arbitrary detector pair has no
    perfect matching on a toric-like graph, while a column's own
    detector set is explained by that column alone and is therefore
    satisfiable on any graph, so the window still compiles and decodes.
    """
    check = [[1, 0], [0, 1], [1, 0], [0, 1]]
    faults = placed_faults(check, [0.1, 0.1], [[0, 0]])
    row = adapter.PyMatchingDecoder()
    graphs = row.compile(faults)
    matching = graphs.plain
    syndrome = numpy.asarray([1, 0, 1, 0], dtype=numpy.uint8)
    selected = matching.decode(syndrome)
    assert selected.tolist() == [1, 0]


def test_two_placed_columns_with_the_same_endpoints_become_one_edge():
    """Parallel faults merge at p1(1 - p2) + p2(1 - p1).

    The convention of Stim's detector error models and of PyMatching's
    model loader, and what a window restriction produces when distinct
    circuit faults fold onto the same in-window endpoints. Two
    parallel 0.05 columns between detectors 0 and 1 combine to 0.095
    (weight 2.254) and beat the path through detector 2 at 0.2 per
    edge (weight 2.773); the lighter column kept alone weighs 2.944,
    so a row that did not merge them would take the path instead.
    """
    check = [[1, 1, 1, 0], [1, 1, 0, 1], [0, 0, 1, 1]]
    priors = [0.05, 0.05, 0.2, 0.2]
    faults = placed_faults(check, priors, [[1, 1, 0, 0]])
    row = adapter.PyMatchingDecoder()
    graphs = row.compile(faults)
    matching = graphs.plain
    syndrome = numpy.asarray([1, 1, 0], dtype=numpy.uint8)
    selected, weight = matching.decode(syndrome, return_weight=True)
    selected_columns = selected.tolist()
    parallel_count = selected_columns[0] + selected_columns[1]
    assert parallel_count == 1
    assert selected_columns[2] == 0
    assert selected_columns[3] == 0
    assert weight == pytest.approx(2.2540580520993854, abs=1e-6)


def test_the_lighter_forced_class_is_the_row_s_own_unforced_answer():
    """A forced solve pins the observable; the lighter class is the decode.

    The complementary gap's two solves (Gidney et al. arXiv:2312.04522
    Sec. "Complementary gap") are the same matching on the graph with
    the observable row appended as one more check. The minimum over the
    two classes is the unconstrained minimum, so the lighter forced
    solve reproduces the row's own correction, and the two weights are
    what a gap subtracts.
    """
    _circuit, model, detection_events, _observables = _window_and_shots()
    row = adapter.PyMatchingDecoder()
    for shot in detection_events:
        plain_job = windows.job_for(model, shot)
        plain = row.decode(plain_job)
        forced_results = []
        for forced_class in (0, 1):
            job = windows.job_for(model, shot)
            job.forced_logical_class = forced_class
            forced = row.decode(job)
            forced_results.append(forced)
        weight_zero = forced_results[0].forced_class_weight
        weight_one = forced_results[1].forced_class_weight
        lighter = forced_results[0]
        if weight_one < weight_zero:
            lighter = forced_results[1]
        assert lighter.correction.tolist() == plain.correction.tolist()


def test_a_row_that_declares_no_forced_solve_refuses_a_forced_job():
    """The declaration is data on the row; the call is a caller's bug."""
    row = adapter.PyMatchingDecoder()
    assert row.decoder_evidence == decoding_records.FORCED_CLASS_SOLVES
    faults = placed_faults([[1, 0], [0, 1]], [0.1, 0.1], [[1, 0]])
    model = window_of(faults, 2)
    syndrome = numpy.zeros(2, dtype=numpy.uint8)
    with pytest.raises(RuntimeError, match="forced-class solve"):
        decoder_module.WindowDecoderBase.decode_forced_window(
            row, None, model, faults, syndrome, 0
        )
