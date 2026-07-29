"""DecodeBacklog metric: the faithful, paper-grounded backlog signal (rounds of unprocessed
syndrome), and the reason it exists -- the ready-queue is blind to windowed-decode backlog.

Grounding (all one condition, f = tau_dec/tau_gen): converges if f<1, diverges if f>1
  - Terhal, Rev. Mod. Phys. 87, 307 (the backlog argument)
  - decoder-switching arXiv:2510.25222 Eq. 5  (r_i recursion, diverges iff f>1)
  - Skoric parallel-window arXiv:2209.08552   (tau_W < n_com*tau_rd)
  - SWIPER arXiv:2412.05115                    (latency factor r > 0.5; backlog is throughput,
                                                NOT a queue/window count)

Setup: distance-3 sliding window (commit=buffer=d=3), round = 1 us, zero link latencies, so a
commit region is generated every n_com*tau_rd = 3 us. A decoder with per-window latency tau_W
sets f = tau_W / 3us. We sweep tau_W across the f<1 / f>1 threshold and across computation
lengths, and read the metric off build_and_run."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.controllers import ModularController, LinkModel
from decsim.message import DecodeResult, Operation
from decsim.metrics import DecodeBacklog, ReadyQueueStats
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds

D = 3                       # commit = buffer = d = 3 rounds; window = 2d = 6 rounds
GEN_US = D * 1.0            # one commit region generated every 3 us (round = 1 us)


class _FixedLatencyDecoder:
    """Timing-only decoder: fixed per-window service time, trivial logical output. Lets a test
    set f = latency / (n_com * round_us) precisely without stim/pymatching."""
    def __init__(self, latency_us):
        self._t = us(latency_us)

    def latency(self, job):
        return self._t

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


def _zero_link_controller(engine):
    return ModularController(engine, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0), log_syndromes=False)


def _run(latency_us, rounds):
    """Return (peak_backlog_rounds, peak_ready_queue) for a memory op of `rounds` rounds."""
    op = Operation(0, "mem", (0,), clifford=True, patches=(0,))
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(rounds),
              round_us=1.0,
              decoder=_FixedLatencyDecoder(latency_us),
              scheme=SlidingWindowScheme(),
              code=SurfaceCodeModel(d=D),
              make_controller=_zero_link_controller,
              make_metrics=lambda e, cl, ch, f: [DecodeBacklog(cl), ReadyQueueStats(cl)],
          ), verbose=False)
    m = res.result.metric_values()
    return m["decode_backlog"]["peak_rounds"], m["ready_queue"]["peak_jobs"]


def test_backlog_bounded_when_decoder_keeps_up():
    """f < 1 (decode faster than generation): backlog stays at the ~2d floor and does NOT grow
    with computation length. (latency 1 us, f = 1/3.)"""
    b30, _ = _run(1.0, 30)
    b60, _ = _run(1.0, 60)
    assert b30 <= 2 * D + 2          # bounded near the commit+buffer floor (= 2d rounds in flight)
    assert b60 <= b30                # doubling the computation does not grow the backlog


def test_backlog_diverges_when_decoder_too_slow():
    """f > 1 (decode slower than generation): backlog grows with computation length -- Terhal's
    accumulating, eventually-exponential backlog. (latency 9 us, f = 3.)"""
    b30, _ = _run(9.0, 30)
    b60, _ = _run(9.0, 60)
    assert b30 > 2 * D + 2           # already well past the bounded floor
    assert b60 > b30 + 5             # and it GROWS as the computation gets longer


def test_ready_queue_is_blind_to_windowed_backlog():
    """The reason this metric exists: in a serial sliding-window chain the ready-queue measures
    decoder-UNIT contention, which stays ~0-1 even when the backlog (rounds) is large. So
    peak ready-queue is the WRONG lens; peak decode backlog is the right one."""
    backlog, queue = _run(9.0, 60)
    assert queue <= 1                # ready-queue says "fine"
    assert backlog >= 5 * max(queue, 1)   # while the real backlog is many rounds deep


def test_backlog_threshold_matches_generation_rate():
    """f crosses 1 exactly at latency = n_com*tau_rd (= GEN_US): just under stays bounded,
    just over begins to accumulate -- the f = tau_dec/tau_gen = 1 boundary of the literature."""
    just_under, _ = _run(GEN_US * 0.8, 60)   # f = 0.8
    just_over, _ = _run(GEN_US * 1.5, 60)    # f = 1.5
    assert just_over > just_under
