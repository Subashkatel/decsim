"""The tesseract row against tesseract_decoder on the same model and seed.

decsim rebuilds a Stim detector error model out of the window's physical
faults (tesseract/window_decoder.py, detector_error_model_of) and
compiles the official backend with a detector-order seed; the same
backend built here from that model with the row's settings and the same
seed returns the same error indices, and the backend the row built
reads back those settings. With merge_errors on and the order seed
2384753 the row predicts what the package's own tesseract-short-beam
profile (tesseract-decoder src/tesseract_sinter_compat.pybind.h:466-472)
predicts on the model sinter hands a decoder
(.pydeps/sinter/_collection/_collection_worker_state.py:28). The wheel
is the bb-decoders extra; the reference tests skip until it is
installed. The yaml refusals need no wheel.
"""

import numpy
import pytest
import stim

import decsim.decoders.decoder as decoder_module
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.tesseract.window_decoder as tesseract_window
import decsim.detector_error_model.fault_model_contracts as fault_models
import decsim.records.decoding as decoding_records
from tests.decoders import windows

ROUNDS = 3
SHOTS = 25
SEED = 11
PHYSICAL = fault_models.FaultRepresentation.PHYSICAL
# every key away from its default, so a key that failed to reach the
# backend would show
SETTINGS = tesseract.TesseractDecoder.Settings(
    detector_beam=7,
    beam_climbing=False,
    no_revisit_detectors=False,
    priority_queue_limit=50_000,
    detector_order_method="breadth_first",
    detector_order_count=3,
    merge_errors=True,
)
PROFILE_SHOTS = 100
# the least seed build_det_orders cannot take as a uint64
ONE_PAST_UINT64 = 2**64


def _seeded_row():
    """The row with its detector orders seeded as a run seeds them."""
    row = tesseract.TesseractDecoder(settings=SETTINGS)
    reservation = row.window_decoder.reserve_run_seed(SEED)
    row.window_decoder.commit_run_seed(reservation)
    return row


def _direct_backend(backend, detector_error_model):
    """The official backend compiled with the settings' own arguments."""
    order_method = backend.utils.DetOrder.DetBFS
    orders = backend.utils.build_det_orders(
        detector_error_model, 3, order_method, SEED
    )
    configuration = backend.tesseract.TesseractConfig(
        dem=detector_error_model,
        det_beam=7,
        beam_climbing=False,
        no_revisit_dets=False,
        verbose=False,
        merge_errors=True,
        pqlimit=50_000,
        det_orders=orders,
        det_penalty=0.0,
        create_visualization=False,
        sparsify_errors=False,
        sparsify_base_degree=-1,
        sparsify_max_degree=-1,
        sparsify_reactivate_limit=-1,
    )
    return configuration.compile_decoder()


def test_the_row_returns_the_backends_error_indices_property():
    backend = pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, SHOTS, 3)
    row = _seeded_row()
    detector_error_model, _coordinates = (
        tesseract_window.detector_error_model_of(model, physical)
    )
    direct = _direct_backend(backend, detector_error_model)
    for shot in detection_events:
        syndrome = windows.row_syndrome(model, shot)
        bits = syndrome.astype(bool)
        expected = direct.decode_to_errors(bits)
        expected = sorted(expected)
        job = windows.job_for(model, shot)
        result = row.decode(job)
        flipped = numpy.flatnonzero(result.correction)
        selected = flipped.tolist()
        assert selected == expected


def test_the_backend_the_row_built_reads_back_its_settings():
    backend = pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 1, 3)
    row = _seeded_row()
    job = windows.job_for(model, detection_events[0])
    row.decode(job)
    _reference, built = row.window_decoder.compiled_by_model[id(model)]
    detector_error_model, _coordinates = (
        tesseract_window.detector_error_model_of(model, physical)
    )
    order_method = backend.utils.DetOrder.DetBFS
    orders = backend.utils.build_det_orders(
        detector_error_model, 3, order_method, SEED
    )
    assert built.config.det_beam == 7
    assert built.config.beam_climbing is False
    assert built.config.no_revisit_dets is False
    assert built.config.pqlimit == 50_000
    assert built.config.merge_errors is True
    assert built.config.det_orders == orders


def test_the_default_row_decodes_unmerged_with_orders_from_the_run_seed():
    backend = pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 1, 3)
    row = tesseract.TesseractDecoder()
    reservation = row.window_decoder.reserve_run_seed(SEED)
    row.window_decoder.commit_run_seed(reservation)
    job = windows.job_for(model, detection_events[0])
    row.decode(job)
    _reference, built = row.window_decoder.compiled_by_model[id(model)]
    detector_error_model, _coordinates = (
        tesseract_window.detector_error_model_of(model, physical)
    )
    order_method = backend.utils.DetOrder.DetIndex
    orders = backend.utils.build_det_orders(
        detector_error_model, 16, order_method, SEED
    )
    assert built.config.merge_errors is False
    assert built.config.det_orders == orders


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize(
    "code_task",
    ["surface_code:rotated_memory_x", "surface_code:rotated_memory_z"],
)
def test_the_short_beam_row_predicts_the_packages_profile_property(
    distance: int, code_task: str
):
    backend = pytest.importorskip("tesseract_decoder")
    circuit = stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.003,
        before_measure_flip_probability=0.003,
        after_reset_flip_probability=0.003,
        before_round_data_depolarization=0.003,
    )
    model = windows.whole_circuit_window(
        circuit, distance, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, PROFILE_SHOTS, 3)
    sinter_model = circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    profiles = backend.make_tesseract_sinter_decoders_dict()
    profile = profiles["tesseract-short-beam"]
    compiled = profile.compile_decoder_for_dem(dem=sinter_model)
    packed = numpy.packbits(detection_events, axis=1, bitorder="little")
    predicted = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed
    )
    predictions = numpy.unpackbits(predicted, axis=1, bitorder="little")
    row = tesseract.TesseractDecoder(settings=tesseract.TESSERACT_SHORT_BEAM)
    for shot, prediction in zip(detection_events, predictions, strict=True):
        job = windows.job_for(model, shot)
        result = row.decode(job)
        assert result.logical_observables == (prediction[0],)


def test_a_fixed_detector_order_seed_replaces_the_run_seed():
    """The orders come from the key's seed, whatever the run seed is."""
    backend = pytest.importorskip("tesseract_decoder")
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    physical = model.require_faults(PHYSICAL)
    detection_events, _ = windows.sampled_shots(circuit, 1, 3)
    settings = tesseract.TesseractDecoder.Settings(detector_order_seed=2384753)
    row = tesseract.TesseractDecoder(settings=settings)
    reservation = row.window_decoder.reserve_run_seed(SEED)
    row.window_decoder.commit_run_seed(reservation)
    job = windows.job_for(model, detection_events[0])
    row.decode(job)
    _reference, built = row.window_decoder.compiled_by_model[id(model)]
    detector_error_model, _coordinates = (
        tesseract_window.detector_error_model_of(model, physical)
    )
    order_method = backend.utils.DetOrder.DetIndex
    orders = backend.utils.build_det_orders(
        detector_error_model, 16, order_method, 2384753
    )
    assert built.config.det_orders == orders


@pytest.mark.parametrize(
    ("key", "value", "sentence"),
    [
        (
            "detector_beam",
            15.0,
            "must be a whole number of detection events, at least 0",
        ),
        ("detector_beam", -1, "at least 0"),
        ("priority_queue_limit", 0, "search states, at least 1"),
        ("detector_order_count", True, "orders, at least 1"),
        ("detector_order_method", "random", "is not a row of its table"),
        ("detector_order_seed", -1, "must be None or a whole number from 0"),
        ("detector_order_seed", 1.5, "must be None or a whole number from 0"),
        ("detector_order_seed", True, "must be None or a whole number from 0"),
        ("detector_order_seed", ONE_PAST_UINT64, "to 18446744073709551615"),
    ],
)
def test_a_search_key_of_the_wrong_type_or_range_is_refused(
    key: str, value, sentence: str
) -> None:
    """A count is a whole number, a method a row.

    A detector-order seed is the uint64 build_det_orders takes
    (tesseract-decoder src/utils.h:42-45).
    """
    fields = {key: value}
    with pytest.raises(ValueError, match=f"{key} .*{sentence}"):
        tesseract.TesseractDecoder.Settings(**fields)


def test_a_row_with_a_fixed_order_seed_names_no_run_seed_owner_but_timing():
    row = tesseract.TesseractDecoder(settings=tesseract.TESSERACT_SHORT_BEAM)

    (timing,) = row.run_seed_children()

    assert timing.child is row.latency_model


def test_a_row_that_draws_its_orders_names_its_window_decoder():
    row = tesseract.TesseractDecoder(settings=SETTINGS)

    (_timing, orders) = row.run_seed_children()

    assert orders.child is row.window_decoder


class _FakeClock:
    """A perf_counter_ns that moves only when the backend works."""

    def __init__(self) -> None:
        self.now_ns = 0

    def perf_counter_ns(self) -> int:
        return self.now_ns


class _TimedBackend:
    """The compiled backend, advancing the fake clock 10 ns a decode."""

    def __init__(self, backend, clock: _FakeClock) -> None:
        self.backend = backend
        self.clock = clock

    def decode_to_errors(self, bits):
        self.clock.now_ns += 10
        return self.backend.decode_to_errors(bits)

    def __getattr__(self, name):
        return getattr(self.backend, name)


def test_a_measured_decode_charges_the_search_and_not_the_build(monkeypatch):
    """Tesseract's benchmark builds its decoder outside the timer.

    tesseract-decoder src/tesseract_main.cc:568-579 constructs the
    decoder, then times decode_to_errors alone. The build here takes
    1000 ns of a fake clock and each search 10 ns, so every measured
    decode, the first included, is 10 ns.
    """
    pytest.importorskip("tesseract_decoder")
    clock = _FakeClock()
    build = tesseract_window._compile_backend

    def slow_build(settings, detector_error_model, seed):
        clock.now_ns += 1000
        backend = build(settings, detector_error_model, seed)
        return _TimedBackend(backend, clock)

    monkeypatch.setattr(tesseract_window, "_compile_backend", slow_build)
    monkeypatch.setattr(decoder_module, "time", clock)
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, 2, 3)
    row = _seeded_row()
    first_job = windows.job_for(model, detection_events[0])
    second_job = windows.job_for(model, detection_events[1])

    _first, first_ns = row.decode_timed(first_job)
    _second, second_ns = row.decode_timed(second_job)

    assert (first_ns, second_ns) == (10, 10)
    assert clock.now_ns == 1020


def test_a_beam_the_backend_cannot_build_leaves_the_shot_unscored():
    """The record takes a beam of 2**31, which the backend's build refuses.

    TesseractConfig takes det_beam as a C++ int (tesseract-decoder
    src/tesseract.pybind.h:57), so the build raises at compile and again
    at the decode, which answers with no correction and the upstream
    exception as its reason, as a search that raises does.
    """
    pytest.importorskip("tesseract_decoder")
    settings = tesseract.TesseractDecoder.Settings(detector_beam=2**31)
    row = tesseract.TesseractDecoder(settings=settings)
    circuit = windows.memory_circuit(3, ROUNDS, 0.005)
    model = windows.whole_circuit_window(
        circuit, ROUNDS, fault_models.PHYSICAL_FAULT_MODEL_REQUIRED
    )
    detection_events, _ = windows.sampled_shots(circuit, 1, 3)
    job = windows.job_for(model, detection_events[0])

    result, _elapsed_ns = row.decode_timed(job)

    reasons = decoding_records.BackendFailureReason
    assert result.no_correction_reason is reasons.UPSTREAM_EXCEPTION
