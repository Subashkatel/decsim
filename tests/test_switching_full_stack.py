"""E1 pre-run integration test (Codex checkpoint-5 required change 1).

Exercises the FULL real-soft-output switching stack that experiment E1
will sweep — StimDevice syndromes -> SlidingWindowScheme windows ->
UnweightedPyMatchingDecoder weak tier wrapped in SoftOutputDecoder (real
complementary-gap confidence) -> Switching escalation -> weighted
PyMatchingDecoder strong re-decode on the two-sided context model —
end to end through build_and_run. Until this file is green, E1 does not
run (docs/validation/2026-07-02-gate3-decision-and-gate4-readiness.md,
design v2 item 2).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pytest.importorskip("pymatching")

from decsim.adapters.stim_device import StimDevice
from decsim.codes import SurfaceCodeModel
from decsim.decoders import SwitchingRouter
from decsim.message import Operation
from decsim.mwpm_decoder import PyMatchingDecoder, UnweightedPyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.soft_output import ComplementaryGapMetric, SoftOutputDecoder
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate


class _Latency:
    def __init__(self, ticks=1):
        self._t = ticks

    def latency(self, job):
        return self._t


class _Recording:
    """Wrap a decoder, recording every job it decodes."""

    def __init__(self, inner):
        self.inner = inner
        self.jobs = []
        self.results = []

    def latency(self, job):
        return self.inner.latency(job)

    def decode(self, job):
        self.jobs.append(job)
        result = self.inner.decode(job)
        self.results.append(result)
        return result


def _run(threshold, d=3, rounds=9, seed=7):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=0.008,
        after_reset_flip_probability=0.008,
        before_measure_flip_probability=0.008,
        before_round_data_depolarization=0.008)
    op = Operation(0, "memory", (0,), clifford=True, circuit=circuit)
    weak = _Recording(SoftOutputDecoder(
        UnweightedPyMatchingDecoder(_Latency()), ComplementaryGapMetric))
    strong = _Recording(PyMatchingDecoder(_Latency()))
    res = simulate(RunSpec(
              ops=[op],
              num_units=1,
              d=d,
              rounds_policy=FixedRounds(rounds),
              code=SurfaceCodeModel(d=d),
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=threshold),
              device=StimDevice(seed=seed),
              decoder=weak,
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
          ), verbose=False)
    return res, weak, strong


def test_real_soft_output_escalation_reaches_the_strong_path():
    """Mid threshold: some (not all) windows escalate; strong jobs carry the
    enlarged two-sided context model, not the weak slice. The threshold is
    chosen INSIDE the range of gaps this workload actually produces (a
    calibration run on the same seed), so decoder updates that shift the
    absolute gap values cannot silently turn the test into all-or-nothing."""
    _, calibration, _ = _run(threshold=0.0)
    gaps = sorted({result.soft_output for result in calibration.results})
    assert len(gaps) >= 2, f"soft output did not distinguish windows: {gaps}"
    threshold = (gaps[0] + gaps[-1]) / 2

    res, weak, strong = _run(threshold=threshold)
    assert strong.jobs, "no window escalated at a mid threshold"
    assert len(strong.jobs) < len(weak.jobs), "everything escalated"
    weak_by_win = {j.window_id: j for j in weak.jobs}
    for job in strong.jobs:
        assert job.hint == "strong"
        wj = weak_by_win[job.window_id]
        assert len(job.dem.detector_ids) >= len(wj.dem.detector_ids)
    assert 0 in res["cluster"].op_results   # a real logical verdict exists


def test_escalation_rate_is_monotone_in_the_threshold():
    """Higher confidence threshold => more windows fail it => more strong
    jobs (real complementary-gap outputs, real syndromes, same seed)."""
    counts = []
    for theta in (0.5, 3.0, 12.0):
        _, _, strong = _run(threshold=theta)
        counts.append(len(strong.jobs))
    assert counts[0] <= counts[1] <= counts[2], counts
    assert counts[2] > counts[0], f"threshold has no effect: {counts}"


def test_never_escalating_matches_weak_only():
    """threshold below every gap -> no strong jobs, and the final logical
    value equals a plain weak-only (no switching) run on the same seed."""
    res_sw, _, strong = _run(threshold=0.0)
    assert not strong.jobs
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=9,
        after_clifford_depolarization=0.008,
        after_reset_flip_probability=0.008,
        before_measure_flip_probability=0.008,
        before_round_data_depolarization=0.008)
    op = Operation(0, "memory", (0,), clifford=True, circuit=circuit)
    res_weak = simulate(RunSpec(
                   ops=[op],
                   num_units=1,
                   d=3,
                   rounds_policy=FixedRounds(9),
                   code=SurfaceCodeModel(d=3),
                   scheme=SlidingWindowScheme(),
                   device=StimDevice(seed=7),
                   decoder=SoftOutputDecoder(UnweightedPyMatchingDecoder(_Latency()),
                                  ComplementaryGapMetric),
               ), verbose=False)
    assert res_sw["cluster"].op_results[0] == res_weak["cluster"].op_results[0]


def test_weak_strong_pair_has_real_accuracy_separation():
    """The E1 weak/strong pair must differ in accuracy (checkpoint-5 fatal
    finding on the v1 pair): unweighted vs weighted MWPM on the same 2000
    global shots at d=3, p=0.008 — measured ~3.1x in the design doc;
    assert >= 1.5x (the pre-registered minimum separation)."""
    import pymatching
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=9,
        after_clifford_depolarization=0.008,
        after_reset_flip_probability=0.008,
        before_measure_flip_probability=0.008,
        before_round_data_depolarization=0.008)
    dem = circuit.detector_error_model(decompose_errors=True)
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        2000, separate_observables=True)
    weighted = pymatching.Matching.from_detector_error_model(dem)
    uniform = stim.DetectorErrorModel("\n".join(
        (ln.split("(")[0] + "(0.1)" + ln.split(")", 1)[1])
        if ln.strip().startswith("error") else ln
        for ln in str(dem).splitlines()))
    unweighted = pymatching.Matching.from_detector_error_model(uniform)
    lw = float((weighted.decode_batch(dets)[:, 0] != obs[:, 0]).mean())
    lu = float((unweighted.decode_batch(dets)[:, 0] != obs[:, 0]).mean())
    assert lu >= 1.5 * lw, f"weak tier not weak enough: {lu} vs {lw}"
