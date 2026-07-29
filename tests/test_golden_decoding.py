"""Frozen decoding-regression corpus: load PRE-GENERATED circuits + shots from tests/data/
and assert our decoders reproduce the frozen golden failure counts on those exact inputs.

Unlike the other decode tests (which sample stim live with a seed -- reproducible only within
a stim version), these run on FROZEN detection-event samples, so they are deterministic,
robust to stim's sampler changing, and don't regenerate data. Each scenario also checks the
Skoric/Tan windowed==global anchor on the frozen shots.

The goldens are tied to the library versions recorded in golden_decoding.json; if a
stim/pymatching/ldpc bump legitimately shifts a few decodes, rebaseline with:
    python tests/data/make_fixtures.py
"""
import json
import pathlib

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")
pytest.importorskip("ldpc")

from decsim.detector_error_model import (  # noqa: E402
    FaultRepresentation,
    GRAPHLIKE_FAULT_MODEL_REQUIRED,
    LINKED_FAULT_MODELS_REQUIRED,
    PHYSICAL_FAULT_MODEL_REQUIRED,
    build_window_error_models,
    decode_windowed,
)
from decsim.belief_matching_decoder import belief_matching_window_decoder    # noqa: E402
from decsim.mwpm_decoder import matching_window_decoder                       # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
GOLDEN = json.loads((DATA / "golden_decoding.json").read_text())


def _windowed_fails(models, inner, representation, dets, obs, n):
    return sum(
        int(
            decode_windowed(
                models,
                dets[i],
                inner,
                selected_fault_representation=representation,
            )[0]
            != obs[i, 0]
        )
        for i in range(n)
    )


def _names(kind):
    return sorted(n for n, g in GOLDEN["scenarios"].items() if g["kind"] == kind)


@pytest.mark.parametrize("name", _names("surface"))
def test_golden_decoding(name):
    g = GOLDEN["scenarios"][name]
    circ = stim.Circuit.from_file(str(DATA / f"{name}.stim"))
    shots = np.load(DATA / f"{name}.shots.npz")
    dets, obs = shots["dets"], shots["obs"]
    assert dets.shape == (g["n"], g["num_detectors"]), "frozen shots shape drifted"
    plan = [tuple(w) for w in g["plan"]]

    # global MWPM -- exact reproduction of the golden
    gm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    gm_fails = int((gm.decode_batch(dets)[:, 0] != obs[:, 0]).sum())
    assert gm_fails == g["global_mwpm_fails"]

    # windowed MWPM -- exact reproduction
    wm_fails = _windowed_fails(
        build_window_error_models(
            circ,
            plan,
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        ),
        matching_window_decoder(),
        FaultRepresentation.GRAPHLIKE,
        dets,
        obs,
        g["n"],
    )
    assert wm_fails == g["windowed_mwpm_fails"]

    # windowed belief-matching -- exact reproduction (on the frozen subset, BP is slow)
    nb = g["bm_subset"]
    bm_fails = _windowed_fails(
        build_window_error_models(
            circ,
            plan,
            fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
        ),
        belief_matching_window_decoder(),
        FaultRepresentation.GRAPHLIKE,
        dets,
        obs,
        nb,
    )
    assert bm_fails == g["windowed_bm_fails"]

    # CONSISTENCY anchors (relative -> robust):
    #  - windowed MWPM tracks global MWPM (Skoric/Tan; exact at buffer=d for these sizes)
    assert abs(wm_fails - gm_fails) <= max(2, int(0.01 * g["n"]))
    #  - belief-matching is at least as accurate as MWPM on the same frozen subset
    mwpm_sub = _windowed_fails(
        build_window_error_models(
            circ,
            plan,
            fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        ),
        matching_window_decoder(),
        FaultRepresentation.GRAPHLIKE,
        dets,
        obs,
        nb,
    )
    assert bm_fails <= mwpm_sub + max(2, int(0.01 * nb))


@pytest.mark.parametrize("name", _names("qldpc_bposd"))
def test_golden_bposd(name):
    """qLDPC (bb72) BP-OSD on frozen shots: windowed BP-OSD reproduces the golden exactly and
    tracks whole-history BP-OSD (approximate decoder -> high but not exact agreement)."""
    sp = pytest.importorskip("scipy.sparse")
    from ldpc import BpOsdDecoder

    from decsim.detector_error_model import detector_error_model_to_faults
    from decsim.bposd_decoder import bposd_window_decoder

    g = GOLDEN["scenarios"][name]
    circ = stim.Circuit.from_file(str(DATA / f"{name}.stim"))
    shots = np.load(DATA / f"{name}.shots.npz")
    dets, obs = shots["dets"], shots["obs"]
    assert dets.shape == (g["n"], g["num_detectors"]), "frozen shots shape drifted"

    rounds = {d: d // g["checks_per_round"] + 1 for d in range(circ.num_detectors)}
    plan = [tuple(w) for w in g["plan"]]
    models = build_window_error_models(
        circ,
        plan,
        detector_rounds=rounds,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
    )
    inner = bposd_window_decoder()
    dem = circ.detector_error_model(decompose_errors=False)
    det_sets, obs_sets, priors = detector_error_model_to_faults(dem)
    H = np.zeros((circ.num_detectors, len(det_sets)), dtype=np.uint8)
    O = np.zeros((circ.num_observables, len(det_sets)), dtype=np.uint8)
    for j, (ds, os_) in enumerate(zip(det_sets, obs_sets)):
        for d in ds:
            H[d, j] = 1
        for o in os_:
            O[o, j] = 1
    gdec = BpOsdDecoder(sp.csr_matrix(H), error_channel=list(priors), max_iter=2,
                        bp_method="product_sum", schedule="serial", osd_method="osd_cs",
                        osd_order=0)
    wf = gf = agree = 0
    for s in range(g["n"]):
        pw = decode_windowed(
            models,
            dets[s],
            inner,
            selected_fault_representation=FaultRepresentation.PHYSICAL,
        )
        pg = (O @ gdec.decode(dets[s])) % 2
        wf += int(not np.array_equal(pw, obs[s]))
        gf += int(not np.array_equal(pg, obs[s]))
        agree += int(np.array_equal(pw, pg))
    assert wf == g["windowed_bposd_fails"]        # exact reproduction (our code)
    assert gf == g["global_bposd_fails"]
    assert agree == g["agree"]
    assert agree / g["n"] > 0.9                   # windowed tracks global (BB Skoric anchor)
