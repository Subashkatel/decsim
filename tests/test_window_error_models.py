"""Tests for slicing global detector error models into window models.

Paper contract: docs/PAPER_MODEL_MAP.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")

from decsim.detector_error_model import (build_window_error_models,
                                             decode_windowed,
                                             detector_error_model_to_faults)
from decsim.mwpm_decoder import matching_window_decoder
from decsim.codes import SurfaceCodeModel
from decsim.schemes import SlidingWindowScheme, ParallelWindowScheme


def _memory_circuit(d=3, rounds=12, p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def _plan(circuit, d=3):
    """The REAL scheme's window plan over the circuit's detector layers
    (layer t = round t+1), exactly as the cluster would plan it."""
    n_layers = 1 + max(int(c[-1]) for c in
                       circuit.get_detector_coordinates().values())
    return SlidingWindowScheme().plan_windows(0, n_layers, SurfaceCodeModel(d=d))


def test_fault_conversion_merges_duplicates_with_the_standard_rule():
    """p (+) q = p(1-q) + q(1-p), the BeliefMatching/QUITS convention."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D1
        error(0.2) D0 D1
        error(0.3) D1 D2 L0
    """)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert len(det_sets) == 2
    i = det_sets.index((0, 1))
    assert priors[i] == pytest.approx(0.1 * 0.8 + 0.2 * 0.9)
    j = det_sets.index((1, 2))
    assert obs_sets[j] == (0,)


def test_composite_errors_split_into_matchable_components():
    """Stim's `^`-separated decompositions become separate <=2-detector faults, each
    carrying the parent probability (PyMatching's own convention)."""
    dem = stim.DetectorErrorModel("error(0.25) D0 D1 ^ D2 D3 L0")
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    assert sorted(det_sets) == [(0, 1), (2, 3)]
    assert priors == [0.25, 0.25]
    # and on the real circuit, EVERY column is matchable
    circuit = _memory_circuit()
    sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    assert max(len(s) for s in sets) <= 2


def test_every_fault_is_owned_by_exactly_one_window():
    """The commit partition: each fault decided once, none lost (Skoric's rule)."""
    circuit = _memory_circuit()
    models = build_window_error_models(circuit, _plan(circuit))
    det_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    owned_total = sum(int(p.owned.sum()) for p in models)
    assert owned_total == len(det_sets)


def test_interior_windows_have_open_time_boundaries():
    """A fault straddling a window's edge appears as a single-detector column -- the
    boundary edge Tan's imaginary detectors formalize."""
    circuit = _memory_circuit()
    models = build_window_error_models(circuit, _plan(circuit))
    interior = models[1]
    assert (interior.check.sum(axis=0) == 1).any()


def test_single_crossing_fault_round_trips_exactly():
    """THE mechanics test: a single fault that crosses a commit boundary must be
    committed by its owning window, hand its beyond-commit flips forward as
    artificial defects, and the windowed pass must reproduce the fault's observable
    flips exactly -- with every handed-forward defect consumed."""
    circuit = _memory_circuit()
    models = build_window_error_models(circuit, _plan(circuit))
    det_sets, obs_sets, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=True))
    w0 = models[0]
    crossing_cols = [c for c in w0.future_flips if w0.owned[c]]
    assert crossing_cols, "no boundary-crossing fault found in window 0"
    decode = matching_window_decoder()
    n_dets = circuit.num_detectors
    checked = 0
    for col in crossing_cols[:5]:
        # rebuild the GLOBAL detection events of exactly this fault
        in_window = set(np.asarray(w0.detector_ids)[w0.check[:, col] > 0])
        beyond = set(w0.future_flips[col])
        events = np.zeros(n_dets, dtype=np.uint8)
        for d in in_window | beyond:
            events[d] = 1
        predicted = decode_windowed(models, events, decode)
        # which fault is this, globally? find it by its full detector set
        full = tuple(sorted(in_window | beyond))
        expected = np.zeros(circuit.num_observables, dtype=np.uint8)
        for o in obs_sets[det_sets.index(full)]:
            expected[o] = 1
        assert (predicted == expected).all(), f"fault {full} mis-roundtripped"
        checked += 1
    assert checked > 0


def test_windowed_accuracy_matches_global_decoding():
    """Skoric Appendix C, the published anchor: with buffer = d, sliding-window
    decoding shows 'no noticeable increase in logical error rate' over decoding the
    whole history at once. Fixed seed -> deterministic counts."""
    pymatching = pytest.importorskip("pymatching")
    circuit = _memory_circuit(d=3, rounds=12, p=0.003)
    models = build_window_error_models(circuit, _plan(circuit))
    shots = 2000
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    global_pred = global_m.decode_batch(dets)
    decode = matching_window_decoder()
    windowed_pred = np.array([decode_windowed(models, dets[i], decode)
                              for i in range(shots)])
    agree = float((windowed_pred == global_pred).all(axis=1).mean())
    ler_global = float((global_pred != obs).any(axis=1).mean())
    ler_windowed = float((windowed_pred != obs).any(axis=1).mean())
    assert agree > 0.97, f"windowed disagrees with global too often: {agree}"
    # 'no noticeable increase': allow binomial wiggle on 2000 shots, nothing more
    assert ler_windowed <= ler_global + 2 * (ler_global / shots) ** 0.5 + 0.005, \
        f"windowed LER {ler_windowed} vs global {ler_global}"


def test_empty_syndrome_decodes_to_no_correction_and_no_flip():
    """Validation-matrix row 6, made explicit: an all-zero detection-event
    vector must produce a zero correction in every window, a zero observable
    from the windowed pipeline, and a zero prediction from global MWPM."""
    pymatching = pytest.importorskip("pymatching")
    circuit = _memory_circuit(d=3, rounds=9, p=0.003)
    models = build_window_error_models(circuit, _plan(circuit))
    decode = matching_window_decoder()
    empty = np.zeros(circuit.num_detectors, dtype=np.uint8)
    assert int(decode_windowed(models, empty, decode).sum()) == 0
    for m in models:
        assert int(decode(m, np.zeros(len(m.detector_ids), dtype=np.uint8))
                   .sum()) == 0
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    assert int(global_m.decode(empty).sum()) == 0


def test_boundary_priors_are_clipped_not_infinite():
    """Regression (found 2026-07-02, validation-matrix row 24): a prior of
    exactly 1.0 -- which any deterministic injected fault produces -- made the
    edge weight -inf and crashed pymatching with 'maximum absolute edge weight
    exceeded'; p=0 silently produced +inf edges. The decoders now clip priors
    to [1e-12, 1-1e-12]. A p=1 fault must decode as 'always there': its own
    syndrome yields exactly its observable effect."""
    import dataclasses
    circuit = _memory_circuit(d=3, rounds=6, p=0.003)
    models = build_window_error_models(circuit, [(1, 3, 6), (4, 6, 6)])
    m0 = models[0]
    empty = np.zeros(len(m0.detector_ids), dtype=np.uint8)
    # p=0: an impossible fault is never selected
    priors = np.array(m0.priors, dtype=float)
    priors[0] = 0.0
    selected = matching_window_decoder()(
        dataclasses.replace(m0, priors=priors), empty)
    assert selected[0] == 0 and int(selected.sum()) == 0
    # p=1: negative weight -- on an EMPTY syndrome MWPM correctly infers the
    # fault happened anyway, compensated by partner faults (this used to be
    # the -inf crash; now it is well-defined negative-weight matching)
    priors = np.array(m0.priors, dtype=float)
    priors[0] = 1.0
    selected = matching_window_decoder()(
        dataclasses.replace(m0, priors=priors), empty)
    assert selected[0] == 1
    # the p=1 fault's own syndrome must select exactly that fault
    priors = np.array(m0.priors, dtype=float)
    priors[0] = 1.0
    m2 = dataclasses.replace(m0, priors=priors)
    syndrome = np.zeros(len(m2.detector_ids), dtype=np.uint8)
    syndrome[np.flatnonzero(m2.check[:, 0])] = 1
    selected = matching_window_decoder()(m2, syndrome)
    assert selected[0] == 1 and int(selected.sum()) == 1, \
        "the clipped p=1 fault must be selected to explain its own syndrome"
    # the runtime adapter's graph-construction site is protected identically
    from decsim.mwpm_decoder import PyMatchingDecoder

    class _NullLatency:
        def latency(self, job):
            return 1

    assert PyMatchingDecoder(_NullLatency())._matching_for_model(m2) is not None
    # malformed priors are BUGS, not degenerate inputs: they must raise, not
    # be silently coerced into probabilities
    for bad in (-0.1, 1.5, float("nan")):
        priors = np.array(m0.priors, dtype=float)
        priors[0] = bad
        broken = dataclasses.replace(m0, priors=priors)
        with pytest.raises(ValueError, match="priors"):
            matching_window_decoder()(broken, empty)
        with pytest.raises(ValueError, match="priors"):
            PyMatchingDecoder(_NullLatency())._matching_for_model(broken)


def test_sub_d_tail_window_measurably_degrades_accuracy():
    """Characterization pin for the 2026-07-02 finding (validation-matrix rows
    11/19): a final commit window shorter than d decodes with too little
    history and measurably degrades windowed accuracy, while absorbing the
    tail into the last full window tracks global decoding almost exactly.

    This is NOT the desired end state -- it documents a real, currently
    shipped behavior (SlidingWindowScheme docstring carries the caveat; the
    timing goldens pin its layout). If a change makes the tail plan match
    global, this test should fail and be UPDATED, loudly, not silently.

    Fixed seed -> deterministic counts; margins allow pymatching tie-breaking
    drift, not behavioral change."""
    pymatching = pytest.importorskip("pymatching")
    d, rounds, p, shots = 3, 10, 0.008, 4000
    circuit = _memory_circuit(d=d, rounds=rounds, p=p)

    def _plan(absorb):
        plan, lo = [], 1
        while (rounds - lo + 1 >= 2 * d) if absorb else (lo + d - 1 <= rounds):
            plan.append((lo, lo + d - 1, min(lo + d - 1 + d, rounds)))
            lo += d
        if lo <= rounds:
            plan.append((lo, rounds, rounds))   # absorbed tail / short tail
        return plan

    assert _plan(True)[-1] == (7, 10, 10)       # 4-round absorbed final commit
    assert _plan(False)[-1] == (10, 10, 10)     # 1-round sub-d tail window
    # and the SHIPPED scheme really does produce a sub-d tail here: for these
    # 11 detector layers its final window commits only 2 (< d) layers. If
    # this assertion fails, the shipped plan changed -- re-evaluate the whole
    # caveat, not just this test.
    shipped = SlidingWindowScheme().plan_windows(0, 11, SurfaceCodeModel(d=d))
    assert shipped[-1] == (10, 11, 14)
    assert shipped[-1][1] - shipped[-1][0] + 1 < d

    dets, obs = circuit.compile_detector_sampler(seed=3).sample(
        shots, separate_observables=True)
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    g_fail = matcher.decode_batch(dets)[:, 0] != obs[:, 0]

    fails = {}
    for absorb in (True, False):
        models = build_window_error_models(circuit, _plan(absorb))
        decode = matching_window_decoder()
        fails[absorb] = np.array([
            decode_windowed(models, dets[k], decode)[0] != obs[k, 0]
            for k in range(shots)])

    # absorbed plan ~= global (measured 2 vs 1 discordant shots at this seed)
    assert int((fails[True] & ~g_fail).sum()) <= 5
    # sub-d tail is MEASURABLY worse than the absorbed plan on paired shots
    # (measured 16 vs 4); if this stops holding, the tail behavior changed
    worse = int((fails[False] & ~fails[True]).sum())
    better = int((fails[True] & ~fails[False]).sum())
    assert worse >= 2 * better + 4, (
        f"sub-d tail no longer degrades accuracy (worse={worse}, "
        f"better={better}) -- shipped-scheme caveat and docs need updating")


def test_parallel_two_sided_windows_match_global_decoding():
    """Two-sided parallel A/B windows (Skoric sec. III.C / Tan Eq. S10, w = s + 2b)
    carry a lookback buffer, so each window must decode its raw syndrome independently
    and commit only its core -- forward-passing artificial defects (the sliding-window
    technique) double-counts the boundary error and inflates the LER. This pins the
    two-sided path against global decoding for d=5, where the bug was glaring (windowed
    LER ~0.10 vs global ~0.06 before the fix). Fixed seed -> deterministic counts."""
    pymatching = pytest.importorskip("pymatching")
    d, rounds = 5, 20
    circuit = _memory_circuit(d=d, rounds=rounds, p=0.005)
    n_layers = 1 + max(int(c[-1]) for c in circuit.get_detector_coordinates().values())
    plan = ParallelWindowScheme().plan_windows(0, n_layers, SurfaceCodeModel(d=d))
    assert any(len(w) == 4 and w[0] < w[1] for w in plan), "expected two-sided windows"
    models = build_window_error_models(circuit, plan)
    assert any(m.has_leading_buffer for m in models)
    shots = 2000
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    global_m = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    global_pred = global_m.decode_batch(dets)
    decode = matching_window_decoder()
    windowed_pred = np.array([decode_windowed(models, dets[i], decode)
                              for i in range(shots)])
    disagree = int((windowed_pred != global_pred).any(axis=1).sum())
    g_fail = (global_pred != obs).any(axis=1)
    w_fail = (windowed_pred != obs).any(axis=1)
    only_w = int((w_fail & ~g_fail).sum())
    only_g = int((g_fail & ~w_fail).sum())
    # paired-shot pins (fixed seed; measured 1 disagreement, only_w=1,
    # only_g=0 -- margins allow pymatching tie-break drift, not regression)
    assert disagree <= 5, \
        f"two-sided windowed drifted from global: {disagree}/{shots} shots differ"
    assert only_w - only_g <= 4, \
        f"windowed strictly worse on paired shots: only_w={only_w} only_g={only_g}"


# ---------------------------------------------------------------------------------
# Bivariate-bicycle code: the slicing must be code-agnostic, not surface-only.
# ---------------------------------------------------------------------------------

_BB_FIXTURE = pathlib.Path(__file__).resolve().parent / "data" / \
    "bb72_12_6_p003_r10.stim"
_BB_CHECKS_PER_ROUND = 36     # the [[72,12,6]] code's Z checks, one detector layer each
# commit 3 / buffer 3 sliding plan over the fixture's 12 detector layers
# (10 noisy rounds + zeroth + final layer), scheme-style 1-based rounds
_BB_PLAN = [(1, 3, 6), (4, 6, 9), (7, 9, 12), (10, 12, 12)]


def _bb_circuit():
    return stim.Circuit.from_file(str(_BB_FIXTURE))


def _bb_models(circuit):
    """QUITS circuits attach no detector coordinates; detectors are emitted in time
    order, one layer of 36 per round, so round = id // 36 + 1."""
    rounds = {d: d // _BB_CHECKS_PER_ROUND + 1 for d in range(circuit.num_detectors)}
    return build_window_error_models(circuit, _BB_PLAN, decompose_errors=False,
                                     detector_rounds=rounds)


def test_bb_circuit_without_coordinates_requires_explicit_rounds():
    """Coordinate-less detectors must fail loudly, not be silently mis-binned."""
    circuit = _bb_circuit()
    with pytest.raises(ValueError, match="detector_rounds"):
        build_window_error_models(circuit, _BB_PLAN, decompose_errors=False)


def test_bb_faults_are_not_matchable_and_partition_exactly():
    """The BB DEM is genuinely non-graphlike (faults flip up to 6 detectors), and the
    ownership partition still holds: every fault committed by exactly one window."""
    circuit = _bb_circuit()
    det_sets, _, _ = detector_error_model_to_faults(
        circuit.detector_error_model(decompose_errors=False))
    assert max(len(s) for s in det_sets) > 2          # matching would be unsound here
    models = _bb_models(circuit)
    assert sum(int(m.owned.sum()) for m in models) == len(det_sets)
    # interior windows hand artificial defects forward; the last window closes
    assert all(len(m.future_flips) > 0 for m in models[:-1])
    assert models[-1].future_flips == {}


def test_bb_windowed_accuracy_matches_global_decoding():
    """The Skoric App C anchor, BB edition: windowed BP-OSD tracks whole-history
    BP-OSD. Unlike exact matching, BP-OSD is approximate, so windowed and global may
    legitimately differ on a few shots (QUITS reports the same character); we pin
    high agreement and LER within binomial wiggle. Fixed seed -> deterministic."""
    pytest.importorskip("ldpc")
    sp = pytest.importorskip("scipy.sparse")
    from ldpc import BpOsdDecoder
    from decsim.bposd_decoder import bposd_window_decoder

    circuit = _bb_circuit()
    models = _bb_models(circuit)
    dem = circuit.detector_error_model(decompose_errors=False)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    H = np.zeros((circuit.num_detectors, len(det_sets)), dtype=np.uint8)
    O = np.zeros((circuit.num_observables, len(det_sets)), dtype=np.uint8)
    for j, (ds, os_) in enumerate(zip(det_sets, obs_sets)):
        for d in ds:
            H[d, j] = 1
        for o in os_:
            O[o, j] = 1
    global_dec = BpOsdDecoder(sp.csr_matrix(H), error_channel=list(priors),
                              max_iter=2, bp_method="product_sum",
                              schedule="serial", osd_method="osd_cs", osd_order=0)
    shots = 300
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(
        shots, separate_observables=True)
    inner = bposd_window_decoder()
    agree = ler_w = ler_g = 0
    for s in range(shots):
        predicted_w = decode_windowed(models, dets[s], inner)
        predicted_g = (O @ global_dec.decode(dets[s].astype(np.uint8))) % 2
        actual = obs[s].astype(np.uint8)
        agree += int(np.array_equal(predicted_w, predicted_g))
        ler_w += int(not np.array_equal(predicted_w, actual))
        ler_g += int(not np.array_equal(predicted_g, actual))
    agree /= shots
    ler_w /= shots
    ler_g /= shots
    assert agree > 0.9, f"windowed disagrees with global too often: {agree}"
    assert ler_w <= ler_g + 2 * (ler_g * (1 - ler_g) / shots) ** 0.5 + 0.005, \
        f"windowed LER {ler_w} vs global {ler_g}"
