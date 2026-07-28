"""Hand-verifiable single-fault decode cases (validation-matrix row 4).

Unlike the golden corpora (frozen outputs of this same stack) these cases are
derived by hand from surface-code theory, so they establish first-time
correctness, not just drift-pinning:

d=3 rotated planar Z-memory, data qubits at odd (x, y), ancillas between them.
A single deterministic fault injected between rounds 3 and 4 must light
exactly the detectors theory names, and both the global MWPM decoder and
decsim's sliding-window decoder must recover the correct logical outcome.

  * X on the BULK data qubit at (3,3): flips the two adjacent Z checks, whose
    ancillas sit diagonally at (2,2) and (4,4). Detectors compare consecutive
    rounds, so exactly the two detectors (2,2,t=3) and (4,4,t=3) fire. The
    matched correction is the error itself, so it cancels: no logical flip.
  * X on the CORNER data qubit at (1,1): only ONE Z-check neighbors it (the
    ancilla at (2,2)), so exactly one detector fires, (2,2,t=3), and MWPM must
    match it to the boundary. (1,1) IS on the logical-Z support, so the raw
    observable flips; the decoder must predict that flip (logical error 0).
  * X on a MEASUREMENT ancilla just after its reset: a classic measurement
    error. It flips that ancilla's round-4 outcome only, so the SAME spatial
    coordinate fires in two consecutive rounds -- (x,y,t=3) and (x,y,t=4) --
    and the correct correction touches no data qubit: no logical flip.

The syndromes are sampled from a NOISELESS circuit carrying one
probability-1 error, so they are deterministic; the decoders' matching graphs
come from the same circuit generated WITH noise (identical detector
structure), since a noiseless DEM has no graph to match on.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.detector_error_model import build_window_error_models, decode_windowed
from decsim.mwpm_decoder import matching_window_decoder
from decsim.codes import SurfaceCodeModel
from decsim.schemes import SlidingWindowScheme

D, ROUNDS, INJECT_AFTER_ROUND = 3, 9, 3


def _generated(p=0.0):
    kwargs = dict(distance=D, rounds=ROUNDS)
    if p:
        kwargs.update(after_clifford_depolarization=p,
                      after_reset_flip_probability=p,
                      before_measure_flip_probability=p,
                      before_round_data_depolarization=p)
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)


def _inject(qubit):
    """The clean circuit with one X_ERROR(1) on `qubit` between rounds 3 and 4
    (right after round 3's MR + DETECTOR block in the flattened text)."""
    flat = str(_generated().flattened()).splitlines()
    mr = [i for i, l in enumerate(flat) if l.startswith("MR")]
    j = mr[INJECT_AFTER_ROUND - 1] + 1
    while flat[j].startswith("DETECTOR"):
        j += 1
    return stim.Circuit("\n".join(flat[:j] + [f"X_ERROR(1) {qubit}"] + flat[j:]))


def _sample(qubit):
    dets, obs = _inject(qubit).compile_detector_sampler().sample(
        2, separate_observables=True)
    assert (dets[0] == dets[1]).all(), "single p=1 fault must be deterministic"
    return dets[0], int(obs[0, 0])


@pytest.fixture(scope="module")
def graphs():
    noisy = _generated(p=0.001)
    clean = _generated()
    assert noisy.num_detectors == clean.num_detectors
    matching = pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model(decompose_errors=True))
    n_layers = 1 + max(int(c[-1])
                       for c in noisy.get_detector_coordinates().values())
    plan = [
        (window.commit_lo, window.commit_hi, window.buffer_hi)
        for window in SlidingWindowScheme().plan_operation(
            0,
            n_layers,
            commit_round_count=D,
            buffer_round_count=D,
        ).windows
    ]
    models = build_window_error_models(noisy, plan)
    coords = {i: tuple(c) for i, c in clean.get_detector_coordinates().items()}
    return matching, models, coords


def _fired(dets, coords):
    return sorted(coords[int(i)] for i in np.flatnonzero(dets))


# qubit indices read off stim's QUBIT_COORDS for this generated circuit:
# data qubit 10 sits at (3,3) [bulk], data qubit 1 at (1,1) [corner],
# ancilla 9 at (2,2) [the Z check the corner qubit touches].

def test_bulk_data_x_fires_exactly_the_two_diagonal_z_checks(graphs):
    matching, models, coords = graphs
    dets, obs = _sample(10)
    assert _fired(dets, coords) == [(2.0, 2.0, 3.0), (4.0, 4.0, 3.0)]
    assert obs == 0
    assert int(matching.decode(dets)[0]) == 0
    assert int(decode_windowed(models, dets, matching_window_decoder())[0]) == 0


def test_corner_data_x_fires_one_check_and_flips_the_logical(graphs):
    matching, models, coords = graphs
    dets, obs = _sample(1)
    assert _fired(dets, coords) == [(2.0, 2.0, 3.0)]
    assert obs == 1, "(1,1) is on the logical-Z support"
    assert int(matching.decode(dets)[0]) == 1, \
        "MWPM must match the lone defect to the boundary through (1,1)"
    assert int(decode_windowed(models, dets, matching_window_decoder())[0]) == 1


def test_measurement_error_fires_the_same_check_in_consecutive_rounds(graphs):
    matching, models, coords = graphs
    dets, obs = _sample(9)
    assert _fired(dets, coords) == [(2.0, 2.0, 3.0), (2.0, 2.0, 4.0)]
    assert obs == 0
    assert int(matching.decode(dets)[0]) == 0
    assert int(decode_windowed(models, dets, matching_window_decoder())[0]) == 0
