"""The data-path decoder emits a REAL soft output, not the 0/1 Bernoulli flag.

`SoftOutputDecoder` wraps any base decoder and attaches typed complementary
confidence to `DecodeResult.soft_output`, computed on the SAME window decode graph the
hard decoder uses. Validated in naive (single-window) mode, where the window
carries the observable (Toshio et al. 2510.25222; see decsim.soft_output).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gc
import weakref

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.message import Operation
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.schemes import NaiveOnlineScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import FixedRounds
from decsim.soft_output import (
    ComplementaryGapMetric,
    ComplementaryGapMetricFactory,
    SoftOutputDecoder,
)
from decsim.run_spec import RunSpec, simulate


def test_soft_output_decoder_requires_a_configured_metric_factory():
    from decsim.soft_output import (
        COMPLEMENTARY_GAP_SOURCE,
        ComplementaryGapMetricFactory,
    )

    metric_factory = ComplementaryGapMetricFactory()
    decoder = SoftOutputDecoder(
        PyMatchingDecoder(_ZeroLatency()),
        metric_factory,
    )

    assert decoder.metric_cls is metric_factory
    assert metric_factory.source is COMPLEMENTARY_GAP_SOURCE
    with pytest.raises(TypeError, match="configured metric factory instance"):
        SoftOutputDecoder(
            PyMatchingDecoder(_ZeroLatency()),
            ComplementaryGapMetric,
        )


def test_soft_output_decoder_releases_metrics_for_dead_window_models():
    from decsim.detector_error_model import (
        FaultRepresentation,
        GRAPHLIKE_FAULT_MODEL_REQUIRED,
    )
    from decsim.message import DecodeJob, DecodeResult, SoftOutput
    from decsim.soft_output import COMPLEMENTARY_GAP_SOURCE

    class Faults:
        observables = np.array([[1]], dtype=np.uint8)

    class Model:
        def __init__(self):
            self.faults = Faults()

        def require_faults(self, representation):
            assert representation is FaultRepresentation.GRAPHLIKE
            return self.faults

    class Metric:
        def __init__(self, model):
            # A configured metric may retain what its factory receives. The
            # wrapper must not extend that ownership to the decoder lifetime.
            self.model = model

        def evaluate(self, syndrome):
            return SoftOutput(
                gap=1.0,
                source=COMPLEMENTARY_GAP_SOURCE,
            )

    class MetricFactory:
        source = COMPLEMENTARY_GAP_SOURCE
        fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

        def __init__(self):
            self.metric_references = []

        def from_window_model(self, model):
            metric = Metric(model)
            self.metric_references.append(weakref.ref(metric))
            return metric

    class BaseDecoder:
        fault_model_requirement = GRAPHLIKE_FAULT_MODEL_REQUIRED

        def decode(self, job):
            return DecodeResult(job.op_id, job.window_id)

    metric_factory = MetricFactory()
    decoder = SoftOutputDecoder(BaseDecoder(), metric_factory)
    models = [Model() for _ in range(256)]
    model_references = [weakref.ref(model) for model in models]

    for operation_id, model in enumerate(models):
        job = DecodeJob(
            op_id=operation_id,
            window_id=0,
            n_rounds=1,
            dem=model,
        )
        result = decoder.decode(job)
        assert result.soft_output is not None

    del job, model, result
    models.clear()
    gc.collect()

    assert all(reference() is None for reference in model_references)
    assert all(
        reference() is None
        for reference in metric_factory.metric_references
    )
    assert not hasattr(decoder, "_metrics")


class _ZeroLatency:
    def latency(self, job):
        return 1


def _circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _naive_shot(circuit, device, decoder, d, rounds, *, seed):
    op = Operation(id=1, name="memory", qubits=(0,), clifford=True, circuit=circuit)
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(rounds),
              code=SurfaceCodeModel(d=d),
              scheme=NaiveOnlineScheme(),
              device=device,
              decoder=decoder,
              seed=seed,
          ), verbose=False)
    return res.window_manager.op_results[1]


class _Capturing(SoftOutputDecoder):
    """Capture the soft output of the last decoded window for assertions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_soft = None

    def decode(self, job):
        result = super().decode(job)
        if result.soft_output is not None:
            self.last_soft = result.soft_output
        return result


def test_engine_emits_real_soft_output_not_flag():
    """Through the real engine, soft_output is a continuous gap, never the {0,1} flag."""
    d, rounds, p = 3, 6, 5e-3
    circuit = _circuit(d, rounds, p)
    decoder = _Capturing(
        PyMatchingDecoder(_ZeroLatency()),
        ComplementaryGapMetricFactory(),
    )
    softs = []
    for shot in range(40):
        _naive_shot(
            circuit,
            StimDevice(),
            decoder,
            d,
            rounds,
            seed=4 + shot,
        )
        assert decoder.last_soft is not None
        assert decoder.last_soft.gap >= 0.0
        softs.append(decoder.last_soft.gap)
    assert set(np.round(softs, 6)) - {0.0, 1.0}          # not just the old path flag
    assert len(np.unique(np.round(softs, 6))) > 5


def test_soft_output_decoder_preserves_hard_decode():
    """Attaching soft output must not change the committed logical value."""
    d, rounds, p = 3, 6, 5e-3
    circuit = _circuit(d, rounds, p)
    plain = PyMatchingDecoder(_ZeroLatency())
    soft = SoftOutputDecoder(
        PyMatchingDecoder(_ZeroLatency()),
        ComplementaryGapMetricFactory(),
    )
    for shot in range(30):
        a = _naive_shot(
            circuit,
            StimDevice(),
            plain,
            d,
            rounds,
            seed=8 + shot,
        )
        b = _naive_shot(
            circuit,
            StimDevice(),
            soft,
            d,
            rounds,
            seed=8 + shot,
        )
        assert a == b


def test_engine_soft_output_is_error_predictive():
    """corr(real soft output, logical error) < 0 through the engine (low g = error-prone)."""
    d, rounds, p = 3, 6, 8e-3
    circuit = _circuit(d, rounds, p)
    decoder = _Capturing(
        PyMatchingDecoder(_ZeroLatency()),
        ComplementaryGapMetricFactory(),
    )
    gaps, errs = [], []
    shots = 600
    for shot in range(shots):
        device = StimDevice()
        decoded = _naive_shot(
            circuit,
            device,
            decoder,
            d,
            rounds,
            seed=15 + shot,
        )
        truth = int(device._truth[1][0])
        gaps.append(decoder.last_soft.gap)
        errs.append(decoded[0] ^ truth)
    gaps, errs = np.asarray(gaps), np.asarray(errs)
    assert errs.sum() > 0
    assert np.corrcoef(gaps, errs)[0, 1] < 0


def test_hard_union_find_engine_path_reproduces_a_deterministic_logical_fault():
    from decsim.union_find_decoder import UnionFindDecoder

    circuit = stim.Circuit(
        """
        X_ERROR(1) 0
        M 0
        DETECTOR(0, 0, 1) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    class CapturingUnionFind(UnionFindDecoder):
        def decode(self, job):
            result = super().decode(job)
            self.last_result = result
            return result

    decoder = CapturingUnionFind(_ZeroLatency())
    decoded = _naive_shot(
        circuit,
        StimDevice(),
        decoder,
        d=3,
        rounds=1,
        seed=8801,
    )

    assert decoded == (1,)
    assert decoder.last_result.correction.tolist() == [1]
    assert decoder.last_result.logical_observables == (1,)
    assert decoder.last_result.soft_output is None


@pytest.mark.parametrize("distance", (3, 5))
def test_real_surface_memory_cluster_wrapper_preserves_hard_result(distance):
    from decsim.soft_output import (
        UnionFindClusterGapDecoder,
        union_find_cluster_gap_source,
    )
    from decsim.union_find_decoder import UnionFindDecoder

    class CapturingClusterDecoder(UnionFindClusterGapDecoder):
        def decode(self, job):
            result = super().decode(job)
            self.last_result = result
            return result

    circuit = _circuit(distance, distance, 2e-3)
    for seed in (8803, 8805, 8807, 8809):
        hard_prediction = _naive_shot(
            circuit,
            StimDevice(),
            UnionFindDecoder(_ZeroLatency()),
            distance,
            distance,
            seed=seed,
        )
        cluster_decoder = CapturingClusterDecoder(
            UnionFindDecoder(_ZeroLatency())
        )
        cluster_prediction = _naive_shot(
            circuit,
            StimDevice(),
            cluster_decoder,
            distance,
            distance,
            seed=seed,
        )

        assert cluster_prediction == hard_prediction
        assert cluster_decoder.last_result.soft_output.source == (
            union_find_cluster_gap_source()
        )
        assert cluster_decoder.last_result.soft_output.gap >= 0.0
