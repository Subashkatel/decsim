"""The paired-core gap engine: same gap, slower-core timing, same loop.

The parallel_pair computation exists so the weak unit's soft output has
an honest hardware story: two forced-class solves on two cores joined
by a subtract-compare. Its contract has three legs: the gap it reports
is the serial metric's gap exactly, its modelled time is the slower
forced solve plus the join and never the serial sum, and the closed
loop behaves identically because the committed answer never changes.
"""

import numpy
import pytest
from scipy.sparse import csr_matrix

import decsim.records.decoding as decoding_records
from decsim.confidence.complementary import (
    COMPLEMENTARY_GAP_SOURCE,
    ComplementaryGapMetric,
    PairedGapEvaluation,
)
from decsim.confidence.decoder import ParallelGapDecoder, SoftOutputDecoder
from decsim.decoders.decoder import DecoderBase
from decsim.detector_error_model.fault_model_contracts import (
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from decsim.front.experiment import load_experiment
from decsim.machine import build_decoder_unit
from tests.escalation.test_switching_mode import (
    measured_shot,
    switching_config,
)
from tests.front.yaml_configs import write_config

PARITY_DISTANCE = 5
PARITY_P = 0.008
PARITY_SHOTS = 300
PARITY_SEED = 11


def surface_code_metric():
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=PARITY_DISTANCE,
        rounds=PARITY_DISTANCE,
        after_clifford_depolarization=PARITY_P,
        before_measure_flip_probability=PARITY_P,
        after_reset_flip_probability=PARITY_P,
    )
    model = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=PARITY_SEED)
    detection_events, _ = sampler.sample(
        PARITY_SHOTS, separate_observables=True
    )
    return ComplementaryGapMetric.from_detector_error_model(
        model
    ), detection_events


def test_paired_evaluate_matches_serial_evaluate_shot_for_shot():
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events:
        serial = metric.evaluate(shot_events)
        paired = metric.paired_evaluate(shot_events)
        assert paired.soft_output.gap == pytest.approx(serial.gap, abs=1e-9)
        assert paired.soft_output.decoded_class_weight == pytest.approx(
            serial.decoded_class_weight, abs=1e-9
        )
        assert len(paired.forced_solve_nanoseconds) == 2
        assert all(t > 0 for t in paired.forced_solve_nanoseconds)


def test_the_forced_pair_reproduces_the_unconstrained_solve():
    # Min over the forced classes IS the plain decode: same minimum
    # weight, and the winning class is the plain correction's class
    # whenever the gap can break the tie.
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events:
        shot_array = numpy.asarray(shot_events, dtype=numpy.uint8)
        bits = shot_array.ravel()
        correction, plain_weight = metric._matching.decode(
            bits, return_weight=True
        )
        observable_bits = metric.observable_matrix @ correction
        observable_parity = observable_bits % 2
        plain_class = int(observable_parity[0])
        paired = metric.paired_evaluate(shot_events)
        assert paired.soft_output.decoded_class_weight == pytest.approx(
            float(plain_weight), abs=1e-9
        )
        if paired.soft_output.gap > 1e-9:
            assert paired.predicted_class == plain_class


ONE_OBSERVABLE_MATRIX = numpy.array([[1]], dtype=numpy.uint8)


class OneObservableModel:
    """The minimal window model the wrapper's metric gate accepts."""

    class _Faults:
        observables = csr_matrix(ONE_OBSERVABLE_MATRIX)

    def require_faults(self, _representation):
        return self._Faults()


class StubWeakDecoder(DecoderBase):
    """A measured row that answers one prediction in no time at all."""

    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, prediction: int):
        self.prediction = prediction

    def latency(self, _job) -> int:
        return 1

    def occupancy(self, _job):
        return None

    def decode(self, job) -> decoding_records.DecodeResult:
        return decoding_records.DecodeResult(
            operation_id=job.operation_id,
            window_id=job.window_id,
            logical_observables=(self.prediction,),
        )

    def decode_timed(self, job):
        return self.decode(job), 0


class StubPairedMetric:
    def __init__(self, predicted_class: int, gap: float, solve_ns: tuple):
        soft_output = decoding_records.SoftOutput(
            gap=gap, source=COMPLEMENTARY_GAP_SOURCE
        )
        self.evaluation = PairedGapEvaluation(
            soft_output=soft_output,
            predicted_class=predicted_class,
            forced_solve_nanoseconds=solve_ns,
        )

    def paired_evaluate(self, _syndrome) -> PairedGapEvaluation:
        return self.evaluation


class StubSignal:
    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, metric: StubPairedMetric):
        self.metric = metric

    def metric_for(self, _model) -> StubPairedMetric:
        return self.metric


def paired_job() -> decoding_records.DecodeJob:
    window_model = OneObservableModel()
    job = decoding_records.DecodeJob(
        operation_id=0, window_id=0, round_count=1, dem=window_model
    )
    job.payloads = []
    return job


def test_pair_timing_charges_the_slower_core_plus_the_join():
    metric = StubPairedMetric(
        predicted_class=0, gap=4.0, solve_ns=(5_000_000, 3_000_000)
    )
    weak_decoder = StubWeakDecoder(prediction=0)
    signal = StubSignal(metric)
    wrapper = ParallelGapDecoder(weak_decoder, signal, combine_nanoseconds=250)
    job = paired_job()
    result, elapsed_ns = wrapper.decode_timed(job)
    assert elapsed_ns == 5_000_000 + 250
    assert result.soft_output.gap == 4.0


def test_the_committed_result_is_the_base_decoders_result():
    # The pair supplies soft output only: correction and observables
    # stay the base decode's, whatever class the pair preferred (the pair
    # speaks about the whole window, the result about the owned slice).
    metric = StubPairedMetric(predicted_class=1, gap=4.0, solve_ns=(1, 1))
    weak_decoder = StubWeakDecoder(prediction=0)
    signal = StubSignal(metric)
    wrapper = ParallelGapDecoder(weak_decoder, signal)
    job = paired_job()
    result = wrapper.decode(job)
    assert result.logical_observables == (0,)
    assert result.soft_output.gap == 4.0


def pair_config(tmp_path, gap_threshold_db: float):
    import yaml

    path = switching_config(tmp_path, gap_threshold_db)
    config_text = path.read_text()
    raw = yaml.safe_load(config_text)
    raw["escalation"]["gap_computation"] = "parallel_pair"
    return write_config(tmp_path, raw)


def test_gap_computation_selects_the_pair_engine(tmp_path):
    serial_path = switching_config(tmp_path, 20.0)
    serial = load_experiment(serial_path)
    serial_engine = build_decoder_unit(serial.settings, "weak")
    assert type(serial_engine.decoder) is SoftOutputDecoder
    paired_path = pair_config(tmp_path, 20.0)
    paired = load_experiment(paired_path)
    pair_engine = build_decoder_unit(paired.settings, "weak")
    assert type(pair_engine.decoder) is ParallelGapDecoder


def test_the_pair_closed_loop_matches_the_serial_closed_loop(tmp_path):
    serial_path = switching_config(tmp_path, 20.0)
    serial = load_experiment(serial_path)
    paired_path = pair_config(tmp_path, 20.0)
    paired = load_experiment(paired_path)
    for seed in range(4):
        serial_shot = measured_shot(serial, seed)
        paired_shot = measured_shot(paired, seed)
        assert paired_shot.logical_failure == serial_shot.logical_failure
        assert (
            paired_shot.link_totals["weak_decoder_to_strong_decoder"][
                "transfers"
            ]
            == serial_shot.link_totals["weak_decoder_to_strong_decoder"][
                "transfers"
            ]
        )
