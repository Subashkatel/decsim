"""Continuous-patch real decoding across operation seams -- feature 3b.

A continuous patch is several scheduling operations on one logical qubit whose syndrome stream is
unbroken (no destructive measurement between them). decsim decodes it as ONE continuous record:
the chip schedules the segment ops, the device tags each segment's rounds to the stream at the
right global round, and the cluster windows + decodes the single stream (one observable) across
the seams.

Acceptance (the exact oracle from the research): the engine's continuous-stream decode equals the
GLOBAL decode of the same continuous circuit, per shot (Skoric App C; interior seams are open
boundaries, Tan). Requires stim + pymatching."""
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


def _agreement(segment_rounds, shots, seed=11):
    R = sum(segment_rounds)
    circ = NoiseModel.circuit_level(P).circuit(distance=D, rounds=R)
    segments, stream_op, rounds_map = continuous_stream(circ, segment_rounds, patch=0, base_id=0)
    gm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    device = StimDevice(seed=seed)
    agree = eng_err = glob_err = 0
    last = None
    for _ in range(shots):
        res = build_and_run(ops=segments, decode_ops=[stream_op], device=device,
                            num_units=4, d=D, rounds_policy=PerOpRounds(rounds_map),
                            code=SurfaceCodeModel(d=D), scheme=SlidingWindowScheme(),
                            decoder=PyMatchingDecoder(_ZeroLatency()), verbose=False)
        pe = int(res["cluster"].op_results.get(stream_op.id, 0))
        pg = int(gm.decode(device._dets[stream_op.id])[0])
        t = int(device._truth[stream_op.id][0])
        agree += (pe == pg); eng_err += (pe != t); glob_err += (pg != t)
        last = res
    return agree, eng_err, glob_err, last, stream_op


def test_two_segment_stream_equals_global_per_shot():
    agree, eng_err, glob_err, _, _ = _agreement([12, 12], shots=250)
    assert agree == 250                          # continuous-stream decode == global, every shot
    assert eng_err == glob_err


def test_three_segment_stream_equals_global_per_shot():
    agree, _, _, _, _ = _agreement([8, 8, 8], shots=150)
    assert agree == 150                          # generality: 3 segments, one continuous decode


def test_stream_is_the_decode_unit_and_a_window_spans_the_seam():
    _, _, _, res, stream_op = _agreement([12, 12], shots=1)
    cluster = res["cluster"]
    # the STREAM (id 0) is windowed; the scheduling segments (ids 1, 2) are not
    assert stream_op.id in cluster.window_count and cluster.window_count[stream_op.id] > 0
    assert 1 not in cluster.window_count and 2 not in cluster.window_count
    # a window genuinely straddles the operation seam at round 12/13
    seam = 12
    spans = [w for (op, k), w in cluster.windows.items()
             if op == stream_op.id and w.commit_lo <= seam and w.buffer_hi >= seam + 1]
    assert spans, "no window spans the operation seam -- decode is not continuous"
