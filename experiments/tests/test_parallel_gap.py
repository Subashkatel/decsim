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

from decsim.confidence.complementary import (
    COMPLEMENTARY_GAP_SOURCE,
    ComplementaryGapMetric,
    PairedGapEvaluation,
)
from decsim.confidence.decoder import ParallelGapDecoder, SoftOutputDecoder
from decsim.detector_error_model.fault_model_contracts import (
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
)
from decsim.message import DecodeJob, DecodeResult, SoftOutput

from experiments.experiment_config import load_experiment
from experiments.build_run import decoder_engine
from experiments.measure_shot import measure_shot

from test_decoder_units import write_config
from test_switching_mode import (
    NEAR_THRESHOLD_P,
    measured_shot,
    switching_config,
)

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
        after_reset_flip_probability=PARITY_P)
    model = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=PARITY_SEED)
    detection_events, _ = sampler.sample(PARITY_SHOTS,
                                         separate_observables=True)
    return ComplementaryGapMetric.from_dem(model), detection_events


def test_paired_evaluate_matches_serial_evaluate_shot_for_shot():
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events:
        serial = metric.evaluate(shot_events)
        paired = metric.paired_evaluate(shot_events)
        assert paired.soft_output.gap == pytest.approx(serial.gap,
                                                       abs=1e-9)
        assert paired.soft_output.w_min == pytest.approx(serial.w_min,
                                                         abs=1e-9)
        assert len(paired.forced_solve_ns) == 2
        assert all(t > 0 for t in paired.forced_solve_ns)


def test_the_forced_pair_reproduces_the_unconstrained_solve():
    """min over the forced classes IS the plain decode: same minimum
    weight, and the winning class is the plain correction's class
    whenever the gap can break the tie."""
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events:
        bits = numpy.asarray(shot_events, dtype=numpy.uint8).ravel()
        correction, plain_weight = metric._base.decode(
            bits, return_weight=True)
        plain_class = int(((metric.obs @ correction) % 2)[0])
        paired = metric.paired_evaluate(shot_events)
        assert paired.soft_output.w_min == pytest.approx(
            float(plain_weight), abs=1e-9)
        if paired.soft_output.gap > 1e-9:
            assert paired.predicted_class == plain_class


class OneObservableModel:
    """The minimal window model the wrapper's metric gate accepts."""

    class _Faults:
        observables = csr_matrix(numpy.array([[1]], dtype=numpy.uint8))

    def require_faults(self, representation):
        return self._Faults()


class StubWeakDecoder:
    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED
    measures_wall_clock = True

    def __init__(self, prediction: int):
        self.prediction = prediction
        self.last_decode_ns = 0

    def latency(self, job) -> int:
        return 1

    def decode(self, job) -> DecodeResult:
        return DecodeResult(op_id=job.op_id, window_id=job.window_id,
                            logical_observables=(self.prediction,))


class StubPairedMetric:
    def __init__(self, predicted_class: int, gap: float, solve_ns: tuple):
        self.evaluation = PairedGapEvaluation(
            soft_output=SoftOutput(gap=gap,
                                   source=COMPLEMENTARY_GAP_SOURCE),
            predicted_class=predicted_class,
            forced_solve_ns=solve_ns)

    def paired_evaluate(self, syndrome) -> PairedGapEvaluation:
        return self.evaluation


class StubMetricFactory:
    source = COMPLEMENTARY_GAP_SOURCE
    fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

    def __init__(self, metric: StubPairedMetric):
        self.metric = metric

    def from_window_model(self, model) -> StubPairedMetric:
        return self.metric


def paired_job() -> DecodeJob:
    job = DecodeJob(op_id=0, window_id=0, n_rounds=1,
                    dem=OneObservableModel())
    job.payloads = []
    return job


def test_pair_timing_charges_the_slower_core_plus_the_join():
    metric = StubPairedMetric(predicted_class=0, gap=4.0,
                              solve_ns=(5_000_000, 3_000_000))
    wrapper = ParallelGapDecoder(StubWeakDecoder(prediction=0),
                                 StubMetricFactory(metric),
                                 combine_ns=250)
    result = wrapper.decode(paired_job())
    assert wrapper.last_decode_ns == 5_000_000 + 250
    assert result.soft_output.gap == 4.0


def test_the_committed_result_is_the_base_decoders_result():
    """The pair supplies soft output only: correction and observables
    stay the base decode's, whatever class the pair preferred (the pair
    speaks about the whole window, the result about the owned slice)."""
    metric = StubPairedMetric(predicted_class=1, gap=4.0,
                              solve_ns=(1, 1))
    wrapper = ParallelGapDecoder(StubWeakDecoder(prediction=0),
                                 StubMetricFactory(metric))
    result = wrapper.decode(paired_job())
    assert result.logical_observables == (0,)
    assert result.soft_output.gap == 4.0


def pair_config(tmp_path, gap_threshold_db: float):
    import yaml
    path = switching_config(tmp_path, gap_threshold_db)
    raw = yaml.safe_load(path.read_text())
    raw["switching"]["gap_computation"] = "parallel_pair"
    return write_config(tmp_path, raw)


def test_gap_computation_selects_the_pair_engine(tmp_path):
    serial_engine = decoder_engine(
        load_experiment(switching_config(tmp_path, 20.0)))
    assert type(serial_engine.decoder) is SoftOutputDecoder
    pair_engine = decoder_engine(
        load_experiment(pair_config(tmp_path, 20.0)))
    assert type(pair_engine.decoder) is ParallelGapDecoder


def test_the_pair_closed_loop_matches_the_serial_closed_loop(tmp_path):
    serial = load_experiment(switching_config(tmp_path, 20.0))
    paired = load_experiment(pair_config(tmp_path, 20.0))
    for seed in range(4):
        serial_shot = measured_shot(serial, seed)
        paired_shot = measured_shot(paired, seed)
        assert (paired_shot.logical_failure
                == serial_shot.logical_failure)
        assert (paired_shot.link_totals["wsd"]["transfers"]
                == serial_shot.link_totals["wsd"]["transfers"])
