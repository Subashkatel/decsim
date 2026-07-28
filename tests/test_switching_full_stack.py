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
from dataclasses import replace
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pytest.importorskip("pymatching")

from decsim.adapters.stim_device import StimDevice
from decsim.codes import SurfaceCodeModel
from decsim.decoders import SAMPLED_CONFIDENCE_SOURCE, SwitchingRouter
from decsim.message import Operation, SeamFaultOwner
from decsim.mwpm_decoder import PyMatchingDecoder, UnweightedPyMatchingDecoder
from decsim.planner import FixedRounds
from decsim.schemes import SlidingWindowScheme
from decsim.soft_output import (
    COMPLEMENTARY_GAP_SOURCE,
    ComplementaryGapMetric,
    SoftOutputDecoder,
)
from decsim.switching import Switching
from decsim.run_spec import RunSpec, simulate
from decsim.window_interactions import DefaultWindowInteraction


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


def _run(threshold, d=3, rounds=9, seed=7, double_window=False, device=None):
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
              rounds_policy=FixedRounds(rounds),
              code=SurfaceCodeModel(d=d),
              scheme=SlidingWindowScheme(),
              strategy=Switching(confidence_threshold=threshold,
                                 expected_source=COMPLEMENTARY_GAP_SOURCE,
                                 double_window=double_window),
              device=device if device is not None else StimDevice(),
              router=SwitchingRouter(weak, strong),
              unit_pools={"default": 1, "strong": 1},
              seed=seed,
          ), verbose=False)
    return res, weak, strong


def test_real_soft_output_escalation_reaches_the_strong_path():
    """Mid threshold: some (not all) windows escalate; strong jobs carry the
    enlarged two-sided context model, not the weak slice. The threshold is
    chosen INSIDE the range of gaps this workload actually produces (a
    calibration run on the same seed), so decoder updates that shift the
    absolute gap values cannot silently turn the test into all-or-nothing."""
    _, calibration, _ = _run(threshold=0.0)
    gaps = sorted({result.soft_output.gap for result in calibration.results})
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
    assert 0 in res.cluster.op_results   # a real logical verdict exists


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
                   rounds_policy=FixedRounds(9),
                   code=SurfaceCodeModel(d=3),
                   scheme=SlidingWindowScheme(),
                   device=StimDevice(),
                   decoder=SoftOutputDecoder(UnweightedPyMatchingDecoder(_Latency()),
                                  ComplementaryGapMetric),
                   seed=7,
               ), verbose=False)
    assert res_sw.cluster.op_results[0] == res_weak.cluster.op_results[0]


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


def test_double_window_full_stack_faithful_start_and_same_shot_truth():
    """Double window through the real path: Stim syndromes, real
    complementary-gap confidence, forward slab with weak-chain skip,
    weighted-MWPM slab decode. One deterministic shot proves mechanics,
    not statistics: slabs start at the suspicious commit, extend forward,
    and dispatch only after the restart window's weak commit; the verdict
    is recomputed from the same shot's per-window results; error labels
    come from the Stim observable, never the strong decoder; a
    never-escalate threshold matches the plain serial weak result."""
    rounds = 15
    _, calibration, _ = _run(threshold=0.0, rounds=rounds, double_window=True)
    gaps = sorted({result.soft_output.gap for result in calibration.results})
    assert len(gaps) >= 2, f"soft output did not distinguish windows: {gaps}"
    threshold = (gaps[0] + gaps[-1]) / 2

    device = StimDevice()
    res, weak, strong = _run(threshold=threshold, rounds=rounds,
                             double_window=True, device=device)
    cluster = res.cluster
    runtime = cluster.window_manager
    assert strong.jobs, "no window escalated at the calibrated mid threshold"

    escalated = set()
    for job in strong.jobs:
        key = job.strong_decode_for
        escalated.add(key)
        weak_window = cluster.windows[key]
        buffer_rounds = max(0, weak_window.buffer_hi - weak_window.commit_hi)
        assert job.window.commit_lo == weak_window.commit_lo
        assert job.window.commit_hi == min(
            weak_window.commit_hi + 2 * buffer_rounds, rounds)
        assert job.window.boundary_in == {}   # raw context, no folded defects
        restart = next(
            (cluster.windows[(key[0], j)]
             for j in sorted(k for o, k in cluster.windows if o == key[0])
             if cluster.windows[(key[0], j)].commit_lo > job.window.commit_hi),
            None)
        if restart is not None:
            assert restart.t_done is not None
            assert job.window.t_dispatch > restart.t_done, (
                "strong slab started before the restart window's weak commit")

    # same-shot provenance: recompute the op verdict from this shot's
    # per-window results; absorbed windows contribute nothing
    weak_values = {(j.op_id, j.window_id): r.logical_observables[0]
                   for j, r in zip(weak.jobs, weak.results)}
    strong_values = {j.strong_decode_for: r.logical_observables[0]
                     for j, r in zip(strong.jobs, strong.results)}
    expected = 0
    for key, value in weak_values.items():
        if key not in escalated and key not in runtime.absorbed_windows:
            expected ^= value
    for value in strong_values.values():
        expected ^= value
    assert cluster.op_results[0] == (expected,)
    # the far side feeds the slab through its raw trailing context: every
    # context round past the slab must arrive with real sampled bits (the
    # repo's validated two-sided formalism; decoded defects from the restart
    # window would double-count the boundary, see
    # test_parallel_two_sided_windows_match_global_decoding)
    for job in strong.jobs:
        context_rounds = [payload for payload in job.payloads
                          if payload.round_index > job.window.commit_hi]
        if job.window.buffer_hi > job.window.commit_hi:
            assert context_rounds, "slab missing its far-side context data"
            assert all(payload.bits is not None for payload in context_rounds)
    # physical truth comes from the SAME shot's Stim observable, never the
    # strong decoder; both the switched verdict and an independent global
    # decode are labeled against it, and flipping the truth bit must flip
    # exactly those error labels
    import pymatching
    truth = int(device._truth[0][0])
    verdict = cluster.op_results[0][0]
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=rounds,
        after_clifford_depolarization=0.008,
        after_reset_flip_probability=0.008,
        before_measure_flip_probability=0.008,
        before_round_data_depolarization=0.008)
    global_prediction = int(pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)).decode(
            device._dets[0])[0])
    switched_error = verdict ^ truth
    global_error = global_prediction ^ truth
    assert {switched_error, global_error} <= {0, 1}
    assert verdict ^ (1 - truth) == 1 - switched_error
    assert global_prediction ^ (1 - truth) == 1 - global_error

    res_calm, _, strong_calm = _run(threshold=0.0, rounds=rounds,
                                    double_window=True)
    assert not strong_calm.jobs
    res_serial, _, strong_serial = _run(threshold=0.0, rounds=rounds)
    assert not strong_serial.jobs
    assert res_calm.cluster.op_results[0] \
        == res_serial.cluster.op_results[0]


@pytest.mark.parametrize(
    "seam_owner",
    [SeamFaultOwner.STRONG_REGION, SeamFaultOwner.RESTART_WINDOW],
)
@pytest.mark.parametrize("seed", [0, 5, 31, 127])
def test_double_window_seam_models_are_decodable_and_partition_ownership(
    seam_owner, seed,
):
    """Both seam choices decode while the selected side owns crossing faults."""
    import numpy as np
    from decsim.decoders import SampledConfidenceDecoder

    rounds = 21
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=rounds,
        after_clifford_depolarization=0.006,
        after_reset_flip_probability=0.006,
        before_measure_flip_probability=0.006,
        before_round_data_depolarization=0.006)
    op = Operation(0, "memory", (0,), clifford=True, circuit=circuit)
    weak = _Recording(SampledConfidenceDecoder(
        UnweightedPyMatchingDecoder(_Latency()), 0.0,
        probability_for=lambda job: 1.0 if job.window_id == 2 else 0.0))
    strong = _Recording(PyMatchingDecoder(_Latency()))

    class SelectedSeamOwner(DefaultWindowInteraction):
        def plan_strong_region(self, *args, **kwargs):
            plan = super().plan_strong_region(*args, **kwargs)
            return replace(plan, restart_seam_fault_owner=seam_owner)

    result = simulate(RunSpec(
        ops=[op],
        num_units=1,
        rounds_policy=FixedRounds(rounds),
        code=SurfaceCodeModel(d=3),
        scheme=SlidingWindowScheme(),
        strategy=Switching(
            confidence_threshold=0.5,
            expected_source=SAMPLED_CONFIDENCE_SOURCE,
            double_window=True,
        ),
        device=StimDevice(),
        router=SwitchingRouter(weak, strong),
        window_interaction=SelectedSeamOwner(),
        unit_pools={"default": 1, "strong": 1},
        seed=seed,
    ), verbose=False)

    (slab_job,) = strong.jobs
    assert slab_job.strong_decode_for == (0, 2)
    slab_lo, slab_hi = slab_job.window.commit_lo, slab_job.window.commit_hi
    assert (slab_lo, slab_hi) == (7, 15)
    restart_job = next(j for j in weak.jobs if j.window_id == 5)
    assert restart_job.payloads[0].round_index == 13   # leading buffer read

    coords = circuit.get_detector_coordinates()

    def column_rounds(dem, column):
        rows = np.nonzero(dem.check[:, column])[0]
        return {int(coords[dem.detector_ids[row]][-1]) + 1 for row in rows}

    restart_dem, slab_dem = restart_job.dem, slab_job.dem
    restart_owned = np.nonzero(restart_dem.owned)[0]
    assert restart_owned.size
    restart_seam_owned = any(
        {slab_hi, slab_hi + 1} <= column_rounds(restart_dem, column)
        for column in restart_owned
    )

    slab_seam_owned = False
    for column in np.nonzero(slab_dem.owned)[0]:
        fault_rounds = column_rounds(slab_dem, column)
        assert min(fault_rounds) >= slab_lo      # owns nothing pre-slab
        if {slab_hi, slab_hi + 1} <= fault_rounds:
            slab_seam_owned = True

    assert restart_seam_owned != slab_seam_owned
    assert slab_seam_owned == (
        seam_owner is SeamFaultOwner.STRONG_REGION)

    weak_values = {
        (job.op_id, job.window_id): decode_result.logical_observables[0]
        for job, decode_result in zip(weak.jobs, weak.results)
    }
    expected = strong.results[0].logical_observables[0]
    for key, value in weak_values.items():
        if key not in {(0, 2), (0, 3), (0, 4)}:
            expected ^= value
    runtime = result.cluster.window_manager
    assert runtime.op_results[0] == (expected,)
    assert runtime.payloads_held == 0
