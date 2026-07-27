"""QLX whole-program PHYSICAL coupling (Gate 2b fixture).

Everything here runs on real QLX-origin artifacts in tests/data/qlx/
(regenerate with ./tools/qlx python3 decsim/tests/data/qlx/
generate_qlx_fixtures.py + dump_decoder_params.py):

  * mem_surface.stim — QLX-emitted whole-program circuit (d=3 surface
    memory, 8 syndrome rounds, 56 coordinate-less DETECTORs, 1 observable)
  * mem_surface_decoder_params.json — emit_decoder_params() output whose
    dem_detector_locs [packet, bit, baseline] provide the detector->round
    map the coordinate-less circuit lacks
  * mem_surface_walker_dem.txt — the analytic walker DEM
  * mem_surface_twin.json — QLX digital-twin reference (ler, det_rate)

Claim level: WHOLE-PROGRAM physical + per-window slicing equivalence.
No per-operation claim is made (compound feedback-bearing programs remain
a QLX exporter boundary — see docs/validation/QLX_UTILIZATION_AND_GAP_
ANALYSIS.md §5).
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from decsim.run_spec import RunSpec, simulate

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

DATA = pathlib.Path(__file__).resolve().parent / "data/qlx"

pytestmark = pytest.mark.skipif(
    not (DATA / "mem_surface_decoder_params.json").exists(),
    reason="QLX physical fixtures missing; run generate_qlx_fixtures.py "
           "and dump_decoder_params.py via ./tools/qlx")

SHOTS = 5000
SEED = 12345


@pytest.fixture(scope="module")
def circuit():
    return stim.Circuit((DATA / "mem_surface.stim").read_text())


@pytest.fixture(scope="module")
def params():
    return json.loads((DATA / "mem_surface_decoder_params.json").read_text())


@pytest.fixture(scope="module")
def detector_rounds(params):
    """QLX submit-packet index == 1-based comparison round."""
    return {i: loc[0] for i, loc in enumerate(params["dem_detector_locs"])}


@pytest.fixture(scope="module")
def sampled(circuit):
    dets, obs = circuit.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dets, obs[:, 0].astype(np.uint8)


def _canon_stim_dem(dem):
    mech = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = mech.get(key, 0.0)
        mech[key] = q * (1 - p) + p * (1 - q)
    return mech


def _canon_walker_dem(text):
    mech = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("error("):
            continue
        head, _, rest = line.partition(")")
        p = float(head[len("error("):])
        dets = tuple(sorted(int(t[1:]) for t in rest.split()
                            if t.startswith("D")))
        obs = tuple(sorted(int(t[1:]) for t in rest.split()
                           if t.startswith("L")))
        key = (dets, obs)
        q = mech.get(key, 0.0)
        mech[key] = q * (1 - p) + p * (1 - q)
    return mech


def test_detector_locs_cover_all_rounds(circuit, params, detector_rounds):
    assert params["dem_num_detectors"] == circuit.num_detectors == 56
    assert len(detector_rounds) == 56
    assert sorted(set(detector_rounds.values())) == list(range(1, 8))
    # 8 stabilizers per comparison round (4 X + 4 Z; dem_num_sx = 4)
    assert params["dem_num_sx"] == 4
    per_round = {}
    for det, rnd in detector_rounds.items():
        per_round[rnd] = per_round.get(rnd, 0) + 1
    assert all(count == 8 for count in per_round.values())


def test_stim_dem_semantically_consistent_with_walker(circuit):
    """Every stim-derived mechanism appears in the walker DEM under the
    current numeric detector ids. NOTE (Codex Gate-2 review, finding 1):
    this is a SEMANTIC-consistency check, NOT an order proof — any
    same-per-round bit permutation also satisfies it. The order proof is
    test_detector_bijection_from_measurement_records. (The walker
    superset is idle-depolarization the stim emitter does not include;
    gap G8, confirmed against EmitStim.cpp which emits nothing for bare
    idle.)"""
    stim_mech = _canon_stim_dem(circuit.detector_error_model())
    walker_mech = _canon_walker_dem(
        (DATA / "mem_surface_walker_dem.txt").read_text())
    missing = set(stim_mech) - set(walker_mech)
    assert not missing, f"stim mechanisms absent from walker DEM: {missing}"
    assert len(walker_mech) > len(stim_mech)   # superset is expected (G8)


def _detector_measurement_semantics(circuit):
    """For each stim DETECTOR (in index order): the absolute measurement
    indices it compares, derived by walking the instruction stream."""
    n_meas = 0
    per_detector = []
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            recs = sorted(n_meas + t.value for t in inst.targets_copy())
            per_detector.append(recs)
        else:
            n_meas += inst.num_measurements
    assert n_meas == circuit.num_measurements
    return per_detector


def test_detector_bijection_from_measurement_records(circuit, params):
    """TRUE index-correspondence proof (replaces the subset check that the
    Codex review refuted with a 40,319-permutation counterexample): for
    every stim DETECTOR d, its measurement-record semantics — later
    measurement m2, earlier m1, with checks-per-round measurements per
    syndrome round — must reproduce the walker's dem_detector_locs[d]
    exactly: packet == m2 // checks, bit == m2 % checks, baseline ==
    m1 // checks. Exact per-detector equality is sensitive to ANY
    permutation, per-round bit shuffles included."""
    checks = 2 * params["dem_num_sx"]           # X + Z checks per round
    locs = params["dem_detector_locs"]
    semantics = _detector_measurement_semantics(circuit)
    assert len(semantics) == len(locs) == circuit.num_detectors
    for det, (recs, loc) in enumerate(zip(semantics, locs)):
        assert len(recs) == 2, f"D{det}: expected a comparison pair"
        m1, m2 = recs
        expected = [m2 // checks, m2 % checks, m1 // checks]
        assert loc == expected, \
            f"D{det}: walker locs {loc} != measurement semantics {expected}"


def test_detector_bijection_rejects_bit_permutations(circuit, params):
    """Negative control: a same-round bit permutation (which fooled the
    old subset check) must FAIL the bijection check."""
    checks = 2 * params["dem_num_sx"]
    perm = [(b + 1) % checks for b in range(checks)]   # cyclic shift
    permuted = [[loc[0], perm[loc[1]], loc[2]]
                for loc in params["dem_detector_locs"]]
    semantics = _detector_measurement_semantics(circuit)
    violations = 0
    for recs, loc in zip(semantics, permuted):
        m1, m2 = recs
        if loc != [m2 // checks, m2 % checks, m1 // checks]:
            violations += 1
    assert violations == len(permuted), \
        "a same-round bit permutation slipped past the bijection check"


def test_windowed_decode_equals_global_on_qlx_circuit(
        circuit, detector_rounds, sampled):
    """decsim's sliding-window slicer, fed the QLX detector->round map,
    must reproduce the global MWPM decode bit-for-bit on every shot
    (NB: on this fixture the decode is VACUOUS -- see G9; this test now
    validates plumbing/mapping only, ledger V1 revised)."""
    from decsim.detector_error_model import build_window_error_models, decode_windowed

    dets, truth = sampled
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    pred_global = matching.decode_batch(dets)[:, 0].astype(np.uint8)

    plan = [(1, 3, 6), (4, 6, 7), (7, 7, 7)]
    models = build_window_error_models(circuit, plan,
                                       detector_rounds=detector_rounds)
    matchings = [pymatching.Matching.from_check_matrix(
        m.check, weights=np.log((1 - m.priors) / m.priors),
        faults_matrix=np.eye(np.asarray(m.check).shape[1],
                             dtype=np.uint8)) for m in models]
    # (G9 review fix: decode_window must return per-column selections;
    # faults_matrix=m.obs returned one obs bit that broadcast against
    # `owned` -- undetected while the fixture decode was vacuous)

    def decode_window(model, syndrome):
        return matchings[models.index(model)].decode(syndrome)

    disagreements = sum(
        int(decode_windowed(models, dets[s], decode_window)[0]
            != pred_global[s])
        for s in range(SHOTS))
    assert disagreements == 0

    fails = int((pred_global != truth).sum())
    twin = json.loads((DATA / "mem_surface_twin.json").read_text())
    twin_ler = float(twin["ler"].strip("'"))
    twin_std = float(twin["ler_std"].strip("'()," ).split(",")[0])
    ler = fails / SHOTS
    sigma = math.sqrt(ler * (1 - ler) / SHOTS + twin_std ** 2)
    # QLX digital_twin's default backend builds pymatching from the SAME
    # stim-emitted DEM (it ignores fq.decoder_config — Codex Gate-2 review
    # finding 3), so this is an independent-sampler comparison of the same
    # decoding pipeline; the tolerance covers pure sampling noise.
    assert abs(ler - twin_ler) < 4 * sigma, \
        (f"decsim LER {ler:.5f} vs QLX twin {twin_ler:.5f} beyond 4 sigma "
         f"({sigma:.5f}) — same decoder path, so this exceeds sampling noise")


def test_qlx_circuit_runs_through_the_full_engine(circuit, detector_rounds):
    """The QLX-origin whole-program circuit streams through the ACTUAL
    simulator (chip -> controller -> window manager -> decoder ->
    orchestrator) via StimDevice's explicit detector-round map, and the
    engine's per-shot logical result matches the offline windowed
    reference decode of the very same sampled shot."""
    from decsim.adapters.stim_device import StimDevice
    from decsim.codes import SurfaceCodeModel
    from decsim.detector_error_model import build_window_error_models, decode_windowed
    from decsim.message import Operation
    from decsim.mwpm_decoder import PyMatchingDecoder
    from decsim.planner import FixedRounds
    from decsim.schemes import SlidingWindowScheme

    class _ZeroLatency:
        def latency(self, job):
            return 1

    rounds = max(detector_rounds.values())          # 7 comparison rounds
    plan = [(1, 3, 6), (4, 6, 7), (7, 7, 7)]
    models = build_window_error_models(circuit, plan,
                                       detector_rounds=detector_rounds)
    matchings = [pymatching.Matching.from_check_matrix(
        m.check, weights=np.log((1 - m.priors) / m.priors),
        faults_matrix=np.eye(np.asarray(m.check).shape[1],
                             dtype=np.uint8)) for m in models]
    # (G9 review fix: decode_window must return per-column selections;
    # faults_matrix=m.obs returned one obs bit that broadcast against
    # `owned` -- undetected while the fixture decode was vacuous)

    def decode_window(model, syndrome):
        return matchings[models.index(model)].decode(syndrome)

    mismatches = 0
    engine_fails = 0
    for shot in range(40):
        device = StimDevice(seed=900 + shot,
                            detector_rounds={1: detector_rounds})
        op = Operation(id=1, name="qlx-mem", qubits=(0,), clifford=True,
                       circuit=circuit)
        res = simulate(RunSpec(
                  ops=[op],
                  num_units=4,
                  rounds_policy=FixedRounds(rounds),
                  code=SurfaceCodeModel(d=3),
                  scheme=SlidingWindowScheme(),
                  device=device,
                  decoder=PyMatchingDecoder(_ZeroLatency()),
              ), verbose=False)
        engine_value = res["cluster"].op_results[1][0]
        offline = int(decode_windowed(models, device._dets[1],
                                      decode_window)[0])
        mismatches += int(engine_value != offline)
        engine_fails += int(engine_value != int(device._truth[1][0]))
    assert mismatches == 0, \
        f"{mismatches}/40 engine shots disagree with the offline reference"
    # smoke-level sanity only (40 shots): LER ~ 0.056, so 0..8 failures
    assert engine_fails < 10


def test_stim_device_no_override_path_unchanged(circuit):
    """Formal regression for the detector_rounds extension: on a
    coordinate-bearing circuit, a StimDevice WITHOUT an override and one
    with an override for a DIFFERENT key produce identical round maps."""
    from decsim.adapters.stim_device import StimDevice
    from decsim.message import Operation

    coord_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=6,
        after_clifford_depolarization=0.001)
    plain = StimDevice(seed=5)
    other = StimDevice(seed=5, detector_rounds={999: {0: 1}})
    for device in (plain, other):
        op = Operation(id=1, name="m", qubits=(0,), clifford=True,
                       circuit=coord_circuit)
        device.begin_operation(op)
    assert plain._by_round[1] == other._by_round[1]
    assert (plain._detector_rounds_for_key(1, coord_circuit, 6)
            == other._detector_rounds_for_key(1, coord_circuit, 6)
            == StimDevice._detector_rounds(coord_circuit, 6))


def test_detection_rate_matches_twin(sampled):
    """Sampler-level agreement: detection fraction vs the twin's det_rate."""
    dets, _ = sampled
    twin = json.loads((DATA / "mem_surface_twin.json").read_text())
    twin_rate = float(twin["det_rate"].strip("'"))
    rate = float(np.asarray(dets, dtype=np.float64).mean())
    n_bits = dets.shape[0] * dets.shape[1]
    sigma = math.sqrt(max(rate * (1 - rate) / n_bits, 1e-12))
    assert abs(rate - twin_rate) < max(5 * sigma, 0.002), \
        f"det rate {rate:.5f} vs twin {twin_rate:.5f}"
