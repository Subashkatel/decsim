"""QLX-sourced PHYSICAL decode validation via the walker DEM (fixes G9).

G9 established that QLX's fabric-to-stim emission is a structural
skeleton (no stabilizer coupling; EmitStim.cpp emitMeasureSyndrome emits
bare noisy MR) — the Gate-2 decode claims on the emitted circuit were
vacuous. QLX's INTENDED physical error model lives in the DEM walker
(EmitDEM.cpp: "For each data qubit x round x Pauli {X,Y,Z}, enumerate
flipped detectors via the parity-check matrices hx/hz"), and the Gate-2
fixture tests/data/qlx/mem_surface_walker_dem.txt carries it: 214
mechanisms, 36 observable-coupled. Stim samples DEMs natively, so the
legitimate QLX physical pipeline is:

    walker DEM -> stim DEM sampler -> syndromes+observables -> decode.

These tests restore a NON-vacuous QLX-sourced decode validation on that
pipeline, with the detector->round map taken from QLX's own
emit_decoder_params()['dem_detector_locs'] (the per-window bridge,
bijection-proven in Gate 2). Scope: memory-class d=3; T programs remain
blocked ARCHITECTURALLY (EmitDEM has no produce_resource/inject
handling; Passes.td: "lattice surgery decomposition planned for v2").
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

DATA = pathlib.Path(__file__).resolve().parent / "data/qlx"
SHOTS = 20000
SEED = 20260703


def walker_dem() -> "stim.DetectorErrorModel":
    return stim.DetectorErrorModel(
        (DATA / "mem_surface_walker_dem.txt").read_text())


def detector_rounds() -> dict:
    """QLX's own detector->round map (dem_detector_locs[d][0] = the
    submit-packet/round index of the detector's later measurement)."""
    params = json.loads((DATA / "mem_surface_decoder_params.json").read_text())
    locs = params["dem_detector_locs"]      # list indexed by detector id
    return {d: int(v[0]) for d, v in enumerate(locs)}


def test_walker_dem_is_a_real_decoding_problem():
    dem = walker_dem()
    text = str(dem)
    assert dem.num_detectors == 56
    assert dem.num_observables == 1
    assert text.count("L0") >= 30          # rich observable coupling
    rounds = detector_rounds()
    assert set(rounds) == set(range(56))
    counts = {}
    for r in rounds.values():
        counts[r] = counts.get(r, 0) + 1
    assert all(c == 8 for c in counts.values()), counts


def test_decode_beats_raw_on_walker_dem_samples():
    """Non-vacuity: the decoder must actually correct errors on QLX's
    own error model (contrast: all-zero predictions on the emitted
    circuit, G9)."""
    dem = walker_dem()
    sampler = dem.compile_sampler(seed=SEED)
    dets, obs, _ = sampler.sample(SHOTS)
    dets = dets.astype(np.uint8)
    truth = obs[:, 0].astype(np.uint8)
    matching = pymatching.Matching.from_detector_error_model(dem)
    pred = matching.decode_batch(dets)[:, 0].astype(np.uint8)
    assert pred.any(), "vacuous decode: predictions all-zero"
    ler = float((pred != truth).mean())
    raw = float(truth.mean())
    assert 0.0 < ler < raw, (ler, raw)


def test_windowed_tracks_global_on_walker_dem_samples():
    """decsim window slicing on QLX's walker DEM + QLX's own
    detector->round map must track global MWPM statistically. NB
    bit-for-bit equality is NOT the correct expectation for sliding
    windows on a real decoding problem (the old Gate-2 bit-for-bit
    result was vacuous, G9); at d=3 with the walker's 2-4% mechanism
    probabilities and 7 rounds, window-boundary effects are material,
    so the claim is: windowed decoding genuinely corrects (beats raw)
    and stays within a declared envelope of global."""
    from decsim.detector_error_model import (detector_error_model_to_faults,
                            _build_models_from_plan,
                            _detector_position_in_round, decode_windowed)

    dem = walker_dem()
    rounds_of = detector_rounds()
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    # The walker DEM carries hyperedge mechanisms (Y errors, 3-4
    # detectors) WITHOUT decomposition hints (G9 sub-gap: EmitDEM emits
    # no `^` decompositions). A matching decoder is a graph decoder, so
    # BOTH sides of this comparison use the graphlike subset -- same
    # model, bit-for-bit comparable.
    keep = [i for i, ds in enumerate(det_sets) if 1 <= len(ds) <= 2]
    det_sets = [det_sets[i] for i in keep]
    obs_sets = [obs_sets[i] for i in keep]
    priors = [priors[i] for i in keep]
    fault_rounds = [tuple(rounds_of[d] for d in ds) for ds in det_sets]
    models = _build_models_from_plan(
        plan=[(1, 3, 5), (4, 5, 6), (6, 7, 7)],
        det_sets=det_sets, obs_sets=obs_sets, priors=priors,
        n_obs=1, round_of=rounds_of, fault_rounds=fault_rounds,
        pos_of=_detector_position_in_round(rounds_of),
        belief_matching=False, h_det_sets=None, h_priors=None,
        hyperedge_to_edge_map=None)
    # ownership must tile over DETECTABLE faults: detector-less
    # mechanisms (undetectable logical flips, e.g. "error(p) L0") touch
    # no window and are equally invisible to the global matcher
    assert sum(int(np.asarray(m.owned).sum()) for m in models) == len(det_sets)

    sampler = dem.compile_sampler(seed=SEED)
    dets, obs, _ = sampler.sample(SHOTS)
    dets = dets.astype(np.uint8)
    n_det = dem.num_detectors
    check = np.zeros((n_det, len(det_sets)), dtype=np.uint8)
    obsm = np.zeros((1, len(det_sets)), dtype=np.uint8)
    for j, (ds, os_) in enumerate(zip(det_sets, obs_sets)):
        for dd in ds:
            check[dd, j] = 1
        for oo in os_:
            obsm[oo, j] = 1
    pri = np.asarray(priors)
    matching = pymatching.Matching.from_check_matrix(
        check, weights=np.log((1 - pri) / pri), faults_matrix=obsm)
    pred_global = matching.decode_batch(dets)[:, 0].astype(np.uint8)

    # decode_window must return PER-COLUMN fault selections (decsim
    # XORs model.obs over selected&owned itself) -> identity faults
    # matrix, NOT model.obs (that returns one obs bit which then
    # broadcasts against `owned` -- a bug that only looks fine on
    # vacuous fixtures; see G9 review)
    matchings = [pymatching.Matching.from_check_matrix(
        np.asarray(m.check),
        weights=np.log((1 - np.asarray(m.priors)) / np.asarray(m.priors)),
        faults_matrix=np.eye(np.asarray(m.check).shape[1], dtype=np.uint8))
        for m in models]

    def decode_window(model, syndrome):
        return matchings[models.index(model)].decode(syndrome)

    truth = obs[:, 0].astype(np.uint8)
    pred_win = np.array(
        [int(decode_windowed(models, dets[s], decode_window)[0])
         for s in range(SHOTS)], dtype=np.uint8)
    raw = float(truth.mean())
    ler_glob = float((pred_global != truth).mean())
    ler_win = float((pred_win != truth).mean())
    assert 0.0 < ler_glob < raw, (ler_glob, raw)
    assert ler_win < raw, (ler_win, raw)          # windows genuinely correct
    # Envelope: generous regression bound, NOT a derived tolerance --
    # measured ratio is ~1.01 (Codex-verified 0.1304/0.12905); 1.5x exists
    # to catch gross window-boundary regressions while tolerating seed
    # variation on 20k shots at d=3.
    assert ler_win <= 1.5 * ler_glob, (ler_win, ler_glob)
