"""Real complementary-gap soft output for decsim's matching decoders.

These tests pin the behaviour the decoder-switching paper (Toshio et al.,
arXiv:2510.25222, Sec. II.C / Fig. 3a-b, Fig. 5, Fig. 8) requires of a real soft
output g = |w_comp - w_min|, replacing decsim's old 0/1 Bernoulli path flag:

  * g is a real, non-negative confidence (NOT a {0,1} flag);
  * small g <-> low confidence <-> error-prone (corr(g, error) < 0);
  * the switch rate gamma(g_th) = P(g < g_th) falls with code distance d.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.soft_output import ComplementaryGapMetric, SoftOutput


def test_complementary_gap_uses_canonical_matching_weights_at_endpoints():
    from decsim.mwpm_decoder.weights import matching_weights
    from decsim.soft_output.complementary import (
        _weights_from_priors,
        dem_to_matrices,
    )

    endpoint_priors = np.array([0.0, 1.0])
    detector_error_model = stim.DetectorErrorModel(
        "error(0) D0 L0\nerror(1) D1"
    )
    _, _, detector_model_weights = dem_to_matrices(detector_error_model)
    window_model_weights = _weights_from_priors(endpoint_priors)
    expected_weights = matching_weights(endpoint_priors)

    assert np.array_equal(detector_model_weights, expected_weights)
    assert np.array_equal(window_model_weights, expected_weights)
    assert np.isfinite(expected_weights).all()
    assert expected_weights[0] > 0
    assert expected_weights[1] < 0


def test_matching_weights_preserve_every_strict_interior_probability():
    from decsim.mwpm_decoder.weights import matching_weights

    strict_interior_priors = np.array([
        5e-13,
        1e-12,
        0.25,
        1 - 1e-12,
        1 - 5e-13,
    ])
    expected_weights = np.log(
        (1 - strict_interior_priors) / strict_interior_priors
    )

    assert np.array_equal(
        matching_weights(strict_interior_priors),
        expected_weights,
    )


def _memory_circuit(d, p, rounds=None):
    rounds = rounds or d
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=rounds, distance=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _sample(circ, shots, seed):
    dem = circ.detector_error_model(decompose_errors=True)
    det, obs = circ.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    return dem, det.astype(np.uint8), obs.astype(np.uint8)


def test_complementary_gap_is_real_valued_not_a_flag():
    """The gap is a real non-negative confidence, not the old 0/1 path flag."""
    dem, det, obs = _sample(_memory_circuit(3, 5e-3), 1500, seed=2)
    metric = ComplementaryGapMetric.from_dem(dem)
    gaps = []
    for i in range(len(det)):
        so = metric.evaluate(det[i])
        assert isinstance(so, SoftOutput)
        assert so.gap >= 0.0
        assert not hasattr(so, "logical_value")
        gaps.append(so.gap)
    gaps = np.asarray(gaps)
    assert len(np.unique(np.round(gaps, 6))) > 10   # many distinct values, not {0,1}
    assert gaps.max() > 1.0


def test_low_gap_shots_are_error_prone():
    """corr(g, error) < 0: small gap = low confidence = more logical errors."""
    dem, det, obs = _sample(_memory_circuit(3, 5e-3), 5000, seed=5)
    metric = ComplementaryGapMetric.from_dem(dem)
    matching = pymatching.Matching.from_detector_error_model(dem)
    gaps = np.empty(len(det)); errs = np.empty(len(det))
    for i in range(len(det)):
        so = metric.evaluate(det[i])
        prediction = matching.decode(det[i])
        gaps[i] = so.gap
        errs[i] = int(prediction[0]) ^ int(obs[i, 0])
    assert errs.sum() > 0
    assert np.corrcoef(gaps, errs)[0, 1] < 0


def test_switch_rate_falls_with_distance():
    """gamma(g_th)=P(g<g_th) decreases from d=3 to d=5 at fixed threshold (Fig. 8)."""
    g_th = 8.0
    rates = {}
    for d in (3, 5):
        dem, det, obs = _sample(_memory_circuit(d, 1e-3), 5000, seed=10 + d)
        metric = ComplementaryGapMetric.from_dem(dem)
        gaps = np.array([metric.evaluate(det[i]).gap for i in range(len(det))])
        rates[d] = float((gaps < g_th).mean())
    assert rates[5] < rates[3]


def test_gap_matches_independent_weight_difference():
    """g equals |w_comp - w_min| computed from the two matchings' own weights."""
    dem, det, obs = _sample(_memory_circuit(3, 5e-3), 200, seed=7)
    metric = ComplementaryGapMetric.from_dem(dem)
    for i in range(len(det)):
        so = metric.evaluate(det[i])
        assert so.gap == pytest.approx(abs(so.w_comp - so.w_min), abs=1e-9)
