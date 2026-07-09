"""Buffering-invariance sweep.

The literature floor for the sliding-window buffer is d rounds per side:
Skoric arXiv:2209.08552 §I.B, Fig. 2 ("it suffices to make the buffer region
of the same size n_buf = d", validated at n_buf = n_com = d in App. C);
Bombín arXiv:2303.04846 §IV.D, Def. 2 + Thm. 1 (b >= d gives soundness and
the full fault distance). The DEGRADATION direction — buffer < d strictly
hurts on the same shots — is Bombín Fig. 11 (LER ~ 50% at b = 0, exponential
improvement in b, saturating at b = d); Skoric never sweeps below d.
(d/2 is the correctable WEIGHT, the wrong quantity for the buffer.)
Sweep the trailing buffer over
{⌈d/2⌉, d, 2d} at d ∈ {3, 5} on the FROZEN decode corpus — identical
detection-event shots for every buffer, so any failure-count change is pure
windowing quality:

  buffer ≥ d  →  statistically flat (tracks the global decode),
  buffer < d  →  degraded (strictly more failures on the same shots),

and `validate_buffer` rejects exactly the configurations that degrade.
"""
import json
import pathlib

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pytest.importorskip("pymatching")

from decsim.detector_error_model import (build_window_error_models,  # noqa: E402
                                                 decode_windowed)
from decsim.codes import SurfaceCodeModel                                     # noqa: E402
from decsim.mwpm_decoder import matching_window_decoder                       # noqa: E402
from decsim.schemes import SlidingWindowScheme                                # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "data"
GOLDEN = json.loads((DATA / "golden_decoding.json").read_text())

# frozen surface-code memory scenarios: (name, d, rounds)
SCENARIOS = [("rsc-d3-r6-p0.005", 3, 6), ("rsc-d5-r10-p0.005", 5, 10)]


def _plan(d, rounds, buffer_rounds):
    """Sliding commit windows of d rounds with a trailing buffer."""
    return [(lo, lo + d - 1, min(lo + d - 1 + buffer_rounds, rounds))
            for lo in range(1, rounds + 1, d)]


def _fails(circ, plan, dets, obs):
    models = build_window_error_models(circ, plan)
    inner = matching_window_decoder()
    return sum(int(decode_windowed(models, dets[i], inner)[0] != obs[i, 0])
               for i in range(dets.shape[0]))


@pytest.mark.parametrize("name,d,rounds", SCENARIOS)
def test_buffer_sweep_floor_at_d(name, d, rounds):
    g = GOLDEN["scenarios"][name]
    circ = stim.Circuit.from_file(str(DATA / f"{name}.stim"))
    shots = np.load(DATA / f"{name}.shots.npz")
    dets, obs = shots["dets"], shots["obs"]

    sweep = {b: _fails(circ, _plan(d, rounds, b), dets, obs)
             for b in (0, (d + 1) // 2, d, 2 * d)}
    n = g["n"]

    # buffer >= d: statistically flat, and tracks the global decode
    assert abs(sweep[2 * d] - sweep[d]) <= max(2, int(0.01 * n))
    assert abs(sweep[d] - g["global_mwpm_fails"]) <= max(2, int(0.01 * n))
    # buffer < d: degraded on the SAME shots (paired comparison, no sampling
    # noise), monotonically in the deficit; the no-buffer point is grossly off
    assert sweep[0] >= sweep[(d + 1) // 2] >= sweep[d]
    assert sweep[0] > sweep[d] + 0.005 * n
    if rounds > 2 * d:
        # enough windows that even one-to-two rounds short of the floor shows
        # (at d=3/r=6 the last window absorbs the tail and hides ⌈d/2⌉=2)
        assert sweep[(d + 1) // 2] > sweep[d]


@pytest.mark.parametrize("name,d,rounds", SCENARIOS)
def test_validate_buffer_rejects_exactly_the_degraded_configs(name, d, rounds):
    scheme = SlidingWindowScheme()
    for b in ((d + 1) // 2, d, 2 * d):
        code = SurfaceCodeModel(d=d, buffer_rounds_override=b)
        if b < d:
            with pytest.raises(ValueError, match="buffering"):
                scheme.validate_buffer(code)
        else:
            scheme.validate_buffer(code)
        assert code.buffering_floor(scheme) == (d, d)
