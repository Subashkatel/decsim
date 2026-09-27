"""The tesseract row against tesseract_decoder on the same model and seed.

decsim rebuilds a Stim detector error model out of the window's physical
faults (tesseract/window_decoder.py, detector_error_model_of) and
compiles the official backend with a detector-order seed; the same
backend built here from that model with the row's settings and the same
seed returns the same error indices, and the backend the row built
reads back those settings. The wheel is the bb-decoders extra; the
reference tests skip until it is installed. The yaml refusals need no
wheel.
"""

import numpy
import pytest

import decsim.config as config
import decsim.decoders.settings as decoder_settings
import decsim.decoders.tesseract.decoder as tesseract
import decsim.decoders.tesseract.window_decoder as tesseract_window
import decsim.detector_error_model.fault_model_contracts as fault_models
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
)


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
        merge_errors=False,
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
    assert built.config.merge_errors is False
    assert built.config.det_orders == orders


def test_the_tier_sections_keys_reach_the_rows_settings():
    section = {
        "kind": "tesseract",
        "units": 1,
        "unit_memory": {"bits": None},
        "engine": {
            "clock": "decoder",
            "fetch_cycles_per_round": 1,
            "fetch_cycles_per_job": 0,
            "release_cycles_per_job": 1,
            "release_cycles_per_round": 0,
        },
        "detector_beam": 7,
        "beam_climbing": False,
        "no_revisit_detectors": False,
        "priority_queue_limit": 50_000,
        "detector_order_method": "breadth_first",
        "detector_order_count": 3,
    }
    clocks = config.ClockSettings({"decoder": 250.0})
    tier = decoder_settings.DecoderSettings.from_yaml(
        section, clocks, "strong_decoder"
    )
    assert tier.row_settings == SETTINGS


def test_a_section_with_no_keys_keeps_the_short_beam_profile():
    """The defaults are the values the row was hard-wired to before."""
    settings = tesseract.TesseractDecoder.Settings.from_yaml(
        {}, None, "weak_decoder"
    )
    assert settings == tesseract.TesseractDecoder.Settings(
        detector_beam=15,
        beam_climbing=True,
        no_revisit_detectors=True,
        priority_queue_limit=200_000,
        detector_order_method="index",
        detector_order_count=16,
    )


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
        ("beam_climbing", 1, "must be true or false"),
        ("no_revisit_detectors", None, "must be true or false"),
        ("detector_order_method", "random", "is not a row of its table"),
    ],
)
def test_a_search_key_of_the_wrong_type_or_range_is_refused(
    key: str, value, sentence: str
) -> None:
    """A count is a whole number, a switch a boolean, a method a row."""
    section = {key: value}
    with pytest.raises(ValueError, match=f"weak_decoder.{key} .*{sentence}"):
        tesseract.TesseractDecoder.Settings.from_yaml(
            section, None, "weak_decoder"
        )
