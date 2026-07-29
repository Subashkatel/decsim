"""Generate the frozen decoding-regression corpus under tests/data/.

RUN ONCE, and re-run only to DELIBERATELY rebaseline (e.g. after a stim/pymatching/ldpc
version bump). Per scenario it writes:

  <name>.stim       -- the circuit (stim-generated, frozen here so the noise model is pinned)
  <name>.shots.npz  -- N frozen detection-event samples: dets (N x num_detectors uint8) +
                       obs (N x num_observables uint8). Freezing the SHOTS (not just the
                       circuit, as bb72 does) makes decode tests deterministic and robust to
                       stim's sampler RNG changing across versions -- no live sampling.

  golden_decoding.json -- per scenario, the exact failure counts OUR decoders produce on those
                       frozen shots (global MWPM, windowed MWPM, windowed belief-matching), plus
                       the library versions they were produced with. Deterministic decoders ->
                       exact reproduction is the regression signal.
"""
import importlib.metadata as md
import json
import math
import pathlib
import sys

import numpy as np
import pymatching
import stim

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
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

# Surface-code scenarios (MWPM + belief-matching) -> add rows here, then re-run this script.
SCENARIOS = [
    dict(name="rsc-d3-r6-p0.005", d=3, rounds=6, p=0.005, n=2000, seed=1001, bm_subset=400),
    dict(name="rsc-d5-r10-p0.005", d=5, rounds=10, p=0.005, n=2000, seed=1002, bm_subset=200),
]

# qLDPC scenario (BP-OSD): reuse the EXISTING frozen bb72 circuit (QUITS-built, not ours to
# regenerate); we freeze its shots + golden BP-OSD failure counts here.
BB = dict(name="bb72_12_6_p003_r10", checks_per_round=36, n=400, seed=11,
          plan=[[1, 3, 6], [4, 6, 9], [7, 9, 12], [10, 12, 12]])


def _circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def _sliding_plan(n_rounds, commit, buffer):
    window_count = max(1, math.ceil(n_rounds / commit))
    plan = []
    for k in range(window_count):
        hi = min((k + 1) * commit, n_rounds)
        plan.append([k * commit + 1, hi, min(hi + buffer, n_rounds)])
    return plan


def _windowed_fails(models, inner, dets, obs, n, representation):
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


def main():
    golden = {
        "_versions": {p: md.version(p) for p in ("stim", "pymatching", "ldpc", "numpy")},
        "scenarios": {},
    }
    for sc in SCENARIOS:
        circ = _circuit(sc["d"], sc["rounds"], sc["p"])
        (HERE / f"{sc['name']}.stim").write_text(str(circ))
        dets, obs = circ.compile_detector_sampler(seed=sc["seed"]).sample(
            sc["n"], separate_observables=True)
        dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
        np.savez_compressed(HERE / f"{sc['name']}.shots.npz", dets=dets, obs=obs)

        plan = _sliding_plan(sc["rounds"], sc["d"], sc["d"])
        gm = pymatching.Matching.from_detector_error_model(
            circ.detector_error_model(decompose_errors=True))
        nb = sc["bm_subset"]
        g = {
            "kind": "surface", "n": sc["n"], "bm_subset": nb,
            "num_detectors": circ.num_detectors, "plan": plan,
            "global_mwpm_fails": int((gm.decode_batch(dets)[:, 0] != obs[:, 0]).sum()),
            "windowed_mwpm_fails": _windowed_fails(
                build_window_error_models(
                    circ,
                    plan,
                    fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
                    fault_exclusion_ranges=(),
                ),
                matching_window_decoder(),
                dets, obs, sc["n"], FaultRepresentation.GRAPHLIKE),
            "windowed_bm_fails": _windowed_fails(
                build_window_error_models(
                    circ,
                    plan,
                    fault_model_requirement=LINKED_FAULT_MODELS_REQUIRED,
                    fault_exclusion_ranges=(),
                ),
                belief_matching_window_decoder(), dets, obs, nb,
                FaultRepresentation.GRAPHLIKE),
        }
        golden["scenarios"][sc["name"]] = g
        print(sc["name"], g)
    _bb_fixture(golden)
    (HERE / "golden_decoding.json").write_text(json.dumps(golden, indent=2))
    print("wrote", HERE / "golden_decoding.json")


def _bb_fixture(golden):
    """Freeze shots + golden BP-OSD failure counts for the existing bb72 qLDPC circuit (the
    circuit is reused, not regenerated). Scoring matches the bb72 test: a shot fails if the
    predicted 12-observable vector differs from the truth."""
    import scipy.sparse as sp
    from ldpc import BpOsdDecoder

    from decsim.detector_error_model import detector_error_model_to_faults
    from decsim.bposd_decoder import bposd_window_decoder

    circ = stim.Circuit.from_file(str(HERE / f"{BB['name']}.stim"))
    dets, obs = circ.compile_detector_sampler(seed=BB["seed"]).sample(
        BB["n"], separate_observables=True)
    dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
    np.savez_compressed(HERE / f"{BB['name']}.shots.npz", dets=dets, obs=obs)

    rounds = {d: d // BB["checks_per_round"] + 1 for d in range(circ.num_detectors)}
    plan = [tuple(w) for w in BB["plan"]]
    models = build_window_error_models(
        circ,
        plan,
        detector_rounds=rounds,
        fault_model_requirement=PHYSICAL_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    inner = bposd_window_decoder()
    # whole-history BP-OSD for the windowed==global consistency anchor
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
    for s in range(BB["n"]):
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
    golden["scenarios"][BB["name"]] = {
        "kind": "qldpc_bposd", "n": BB["n"], "num_detectors": circ.num_detectors,
        "num_observables": circ.num_observables, "checks_per_round": BB["checks_per_round"],
        "plan": BB["plan"], "windowed_bposd_fails": wf, "global_bposd_fails": gf,
        "agree": agree}
    print(BB["name"], golden["scenarios"][BB["name"]])


if __name__ == "__main__":
    main()
