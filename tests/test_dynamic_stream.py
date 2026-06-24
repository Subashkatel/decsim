"""Runtime window insertion -- 5-real dynamic, Stage B.

The cluster builds a continuous stream's decode windows AT RUNTIME as rounds arrive (the SWIPER
arXiv:2412.05115 Sec 2.4/5.1 round-driven WindowBuilder), instead of from the compile-time plan, so
a stretch of length unknown at plan time can be absorbed by creating more windows. Each window is
created eagerly (when its commit region begins) so the dependency chain + artificial-defect handoff
are wired before any window commits; the slicing is the validated WindowSlicer (test_window_slicer).

Acceptance (Stage B): replaying a KNOWN circuit through the runtime path reproduces the global decode
per shot, and the windows built at runtime match the static plan's count and geometry. Requires
stim + pymatching."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel
from decsim.streams import continuous_stream
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.schemes import SlidingWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import PerOpRounds
from decsim.wiring import build_and_run

D, P = 3, 0.003


class _ZeroLatency:
    def latency(self, job):
        return 1


def _run(circ, segments, stream_op, rounds_map, device):
    return build_and_run(ops=segments, dynamic_streams=[stream_op], device=device,
                         num_units=4, d=D, rounds_policy=PerOpRounds(rounds_map),
                         code=SurfaceCodeModel(d=D), scheme=SlidingWindowScheme(),
                         decoder=PyMatchingDecoder(_ZeroLatency()), verbose=False)


def test_runtime_built_windows_equal_global_per_shot():
    circ = NoiseModel.circuit_level(P).circuit(distance=D, rounds=24)
    segments, stream_op, rounds_map = continuous_stream(circ, [12, 12], patch=0, base_id=0)
    gm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    device = StimDevice(seed=11)
    agree = eng_err = glob_err = 0
    shots = 250
    for _ in range(shots):
        res = _run(circ, segments, stream_op, rounds_map, device)
        pe = int(res["cluster"].op_results.get(stream_op.id, 0))
        pg = int(gm.decode(device._dets[stream_op.id])[0])
        t = int(device._truth[stream_op.id][0])
        agree += (pe == pg); eng_err += (pe != t); glob_err += (pg != t)
    assert agree == shots                          # windows built at runtime decode == global
    assert eng_err == glob_err


def test_runtime_window_count_and_geometry_match_static_plan():
    R = 24
    circ = NoiseModel.circuit_level(P).circuit(distance=D, rounds=R)
    segments, stream_op, rounds_map = continuous_stream(circ, [12, 12], patch=0, base_id=0)
    res = _run(circ, segments, stream_op, rounds_map, StimDevice(seed=2))
    cluster = res["cluster"]
    static = SlidingWindowScheme().plan_windows(0, R, SurfaceCodeModel(d=D))
    assert cluster.window_count[stream_op.id] == len(static)                  # same number of windows
    assert set(cluster.window_count) == {stream_op.id}                        # segments are not decode units
    built = sorted((cluster.windows[(stream_op.id, k)].commit_lo,
                    min(cluster.windows[(stream_op.id, k)].commit_hi, R))
                   for k in range(cluster.window_count[stream_op.id]))
    want = sorted((cl, ch) for (cl, ch, _bh) in static)
    assert built == want                                             # same commit regions
    assert len(cluster.committed_windows) == cluster.total_windows


def test_dynamic_matches_static_decode_unit_per_shot():
    """The dynamic (runtime-built) stream and the static (pre-planned) continuous stream give the
    SAME logical value per shot -- the runtime builder reproduces the compile-time plan exactly."""
    circ = NoiseModel.circuit_level(P).circuit(distance=D, rounds=24)
    segs_d, stream_d, rmap = continuous_stream(circ, [12, 12], patch=0, base_id=0)
    segs_s, stream_s, _ = continuous_stream(circ, [12, 12], patch=0, base_id=0)
    dyn_dev, stat_dev = StimDevice(seed=9), StimDevice(seed=9)        # same seed -> same shots
    for _ in range(120):
        rd = _run(circ, segs_d, stream_d, rmap, dyn_dev)
        rs = build_and_run(ops=segs_s, decode_ops=[stream_s], device=stat_dev, num_units=4, d=D,
                           rounds_policy=PerOpRounds(rmap), code=SurfaceCodeModel(d=D),
                           scheme=SlidingWindowScheme(), decoder=PyMatchingDecoder(_ZeroLatency()),
                           verbose=False)
        assert int(rd["cluster"].op_results.get(stream_d.id, 0)) == \
               int(rs["cluster"].op_results.get(stream_s.id, 0))
