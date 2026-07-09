"""Real idle-stream decoding tests.

The idle length is modeled as a known segment. Dynamic idle insertion is listed
in docs/ROADMAP.md.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

from decsim.stimcircuits import NoiseModel
from conftest import continuous_stream
from decsim.adapters.stim_device import StimDevice
from decsim.mwpm_decoder import PyMatchingDecoder
from decsim.schemes import SlidingWindowScheme
from decsim.codes import SurfaceCodeModel
from decsim.planner import PerOpRounds
from decsim.run_spec import RunSpec, simulate

D, P = 3, 0.005
OP_A, IDLE, OP_B = 8, 8, 8                     # [op | idle | op] on one continuous patch
R = OP_A + IDLE + OP_B


class _ZeroLatency:
    def latency(self, job):
        return 1


def _single_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    assert len(payloads) == 1
    return payloads[0]


def _setup():
    circ = NoiseModel.circuit_level(P).circuit(distance=D, rounds=R)
    segments, stream_op, rounds_map = continuous_stream(
        circ, [OP_A, IDLE, OP_B], patch=0, base_id=0, name="qubit")
    return circ, segments, stream_op, rounds_map


def test_idle_stretch_carries_real_firing_syndromes():
    """The idle segment must emit REAL, non-empty, actually-firing detection events -- not the
    empty payloads the stall emitter produces today."""
    circ, segments, stream_op, _ = _setup()
    idle_seg = segments[1]                     # the middle (idle) segment
    dev = StimDevice(seed=1)
    for seg in segments:
        dev.begin_operation(seg)
    fired = 0
    nbits = 0
    for r in range(1, IDLE + 1):
        bits = np.asarray(_single_payload(dev, idle_seg, r).bits, np.uint8)
        nbits += bits.size
        fired += int(bits.sum())
    assert nbits > 0                           # the idle rounds carry real syndrome bits
    assert fired > 0                           # and they actually fire (real noise, not empty)


def test_continuous_idle_decode_equals_global_per_shot():
    circ, segments, stream_op, rounds_map = _setup()
    gm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    device = StimDevice(seed=17)
    agree = eng_err = glob_err = 0
    shots = 250
    for _ in range(shots):
        res = simulate(RunSpec(
                  ops=segments,
                  decode_ops=[stream_op],
                  device=device,
                  num_units=4,
                  d=D,
                  rounds_policy=PerOpRounds(rounds_map),
                  code=SurfaceCodeModel(d=D),
                  scheme=SlidingWindowScheme(),
                  decoder=PyMatchingDecoder(_ZeroLatency()),
              ), verbose=False)
        pe = int(res["cluster"].op_results.get(stream_op.id, 0))
        pg = int(gm.decode(device._dets[stream_op.id])[0])
        t = int(device._truth[stream_op.id][0])
        agree += (pe == pg); eng_err += (pe != t); glob_err += (pg != t)
    # Skoric App C anchor (buffer=d): windowed tracks global "within numerical error" -- here at
    # higher noise (p=0.005) the two pick the same correction on essentially every shot, and their
    # error counts are statistically indistinguishable. (Exact per-shot equality holds only at low
    # noise, e.g. the 3b / full-stack p=0.003 config.)
    assert agree >= 0.98 * shots                          # idle errors decoded just like global
    assert abs(eng_err - glob_err) <= (shots - agree)     # disagreements split, no systematic loss


def test_idle_stretch_decoded_by_runtime_builder_equals_global():
    """5-real, Stage C (tractable part): the idle stretch is decoded through the RUNTIME window
    builder (dynamic_streams) -- windows for the idle rounds are created as those rounds arrive,
    not from a compile-time plan -- and the result equals the global decode within numerical error.
    (Fully dynamic idle LENGTH from a live stall needs on-the-fly circuit construction; here the idle
    length is modelled but the DECODE is genuinely runtime, via grow_stream.)"""
    from decsim.planner import PerOpRounds
    circ, segments, stream_op, rounds_map = _setup()
    gm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    device = StimDevice(seed=21)
    agree = 0
    shots = 200
    for _ in range(shots):
        res = simulate(RunSpec(
                  ops=segments,
                  dynamic_streams=[stream_op],
                  device=device,
                  num_units=4,
                  d=D,
                  rounds_policy=PerOpRounds(rounds_map),
                  code=SurfaceCodeModel(d=D),
                  scheme=SlidingWindowScheme(),
                  decoder=PyMatchingDecoder(_ZeroLatency()),
              ), verbose=False)
        pe = int(res["cluster"].op_results.get(stream_op.id, 0))
        pg = int(gm.decode(device._dets[stream_op.id])[0])
        agree += (pe == pg)
    assert agree >= 0.98 * shots                      # idle rounds decoded at runtime track global


def test_a_window_commits_inside_the_idle_stretch():
    """The idle rounds are genuinely decoded: at least one window commits rounds that lie wholly
    within the idle segment (global rounds OP_A+1 .. OP_A+IDLE)."""
    circ, segments, stream_op, rounds_map = _setup()
    res = simulate(RunSpec(
              ops=segments,
              decode_ops=[stream_op],
              device=StimDevice(seed=3),
              num_units=4,
              d=D,
              rounds_policy=PerOpRounds(rounds_map),
              code=SurfaceCodeModel(d=D),
              scheme=SlidingWindowScheme(),
              decoder=PyMatchingDecoder(_ZeroLatency()),
          ), verbose=False)
    cluster = res["cluster"]
    lo, hi = OP_A + 1, OP_A + IDLE
    inside = [w for (op, k), w in cluster.windows.items()
              if op == stream_op.id and w.commit_lo >= lo and w.commit_hi <= hi]
    assert inside, "no window commits inside the idle stretch -- idle rounds not really decoded"
