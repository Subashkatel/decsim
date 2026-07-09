"""WindowSlicer -- the per-window slicer shared by the static builder and the RUNTIME round-driven
window builder (the core of dynamic idle-stream decoding, 5-real dynamic part).

Grounding: SWIPER (arXiv:2412.05115 Sec 2.4/5.1, Fig. 9) cuts decode windows from rounds AS THEY
ARRIVE, so an idle stretch of runtime-unknown length is absorbed by creating more windows. For that
to be correct, slicing one window at a time (threading fault ownership) must reproduce BOTH (a) the
all-at-once build_window_error_models decoding problems exactly, and (b) the global decode.

Requires stim + pymatching."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel
from decsim.schemes import SlidingWindowScheme, ParallelWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.detector_error_model import (build_window_error_models, WindowSlicer,
                                                 decode_windowed)
from decsim.mwpm_decoder import matching_window_decoder

D = 3


def _same(a, b):
    return (a.detector_ids == b.detector_ids and np.array_equal(a.check, b.check)
            and np.array_equal(a.obs, b.obs) and np.array_equal(a.owned, b.owned)
            and a.future_flips == b.future_flips and np.allclose(a.priors, b.priors))


def _incremental(circ, plan, folded):
    slicer = WindowSlicer(circ, detector_rounds=folded)
    out = []
    for k, win in enumerate(plan):
        if len(win) == 4:
            bl, cl, ch, bh = win
        else:
            cl, ch, bh = win
            bl = cl
        out.append(slicer.slice_window(bl, cl, ch, bh, is_last=(k == len(plan) - 1)))
    return out


@pytest.mark.parametrize("scheme,R", [(SlidingWindowScheme(), 24), (ParallelWindowScheme(), 30)])
def test_slicer_identical_to_static_builder(scheme, R):
    """Window-for-window, the incremental slicer reproduces build_window_error_models exactly --
    pins the two against drift for both the sliding and the A/B (two-sided-buffer) schemes."""
    circ = NoiseModel.circuit_level(0.003).circuit(distance=D, rounds=R)
    folded = {det: min(int(c[-1]) + 1, R) for det, c in circ.get_detector_coordinates().items()}
    plan = [tuple(list(t[:-1]) + [min(t[-1], R)]) for t in scheme.plan_windows(0, R, SurfaceCodeModel(d=D))]
    ref = build_window_error_models(circ, plan, detector_rounds=folded)
    inc = _incremental(circ, plan, folded)
    assert len(inc) == len(ref)
    assert all(_same(a, b) for a, b in zip(ref, inc))


def test_incremental_sliding_decode_equals_global_per_shot():
    """Windows built ONE AT A TIME (round-driven) decode equal to the global decoder, per shot --
    the SWIPER round-driven WindowBuilder is correct on real syndromes (what SWIPER-SIM itself,
    a timing-only simulator, never does)."""
    R = 24
    circ = NoiseModel.circuit_level(0.003).circuit(distance=D, rounds=R)
    folded = {det: min(int(c[-1]) + 1, R) for det, c in circ.get_detector_coordinates().items()}
    plan = SlidingWindowScheme().plan_windows(0, R, SurfaceCodeModel(d=D))
    inc = _incremental(circ, plan, folded)
    gm = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    inner = matching_window_decoder()
    shots = 1500
    dets, obs = circ.compile_detector_sampler(seed=5).sample(shots, separate_observables=True)
    agree = 0
    for s in range(shots):
        pw = int(decode_windowed(inc, dets[s], inner)[0])
        pg = int(gm.decode(dets[s])[0])
        agree += (pw == pg)
    assert agree == shots                          # incremental == global, exactly, every shot


def test_matching_window_decoder_cache_survives_id_reuse():
    """Replication-run finding: the matching cache was keyed by id() with no
    eviction, so a model allocated at a dead model's address received the
    dead model's matching (shape errors or silently wrong corrections)."""
    import gc
    import numpy as np
    from decsim.detector_error_model import build_window_error_models
    from decsim.mwpm_decoder import matching_window_decoder

    import pathlib
    data = pathlib.Path(__file__).resolve().parent / "data"
    circ = stim.Circuit.from_file(str(data / "rsc-d3-r6-p0.005.stim"))
    inner = matching_window_decoder()

    def one_model(buffer_rounds):
        plan = [(1, 3, min(3 + buffer_rounds, 6))]
        return build_window_error_models(circ, plan)[0]

    a = one_model(0)
    n_dets_a = a.check.shape[0]
    inner(a, np.zeros(n_dets_a, dtype=np.uint8))
    target = id(a)
    del a
    gc.collect()
    for _ in range(500):                       # try to force an id() collision
        b = one_model(3)                       # DIFFERENT window shape
        if id(b) == target:
            break
        del b
        gc.collect()
    else:
        pytest.skip("could not provoke an id() reuse on this platform")
    # with the stale cache this raised ValueError (wrong matching graph)
    inner(b, np.zeros(b.check.shape[0], dtype=np.uint8))
