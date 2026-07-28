"""Backlog cross-check vs QLX's analytic stall model (gap G3).

QLX (estimate/schedule/orchestrator.py, spec §8.7.1) models decoder
backlog analytically: with per-round decode latency `lat` and cycle time
`cycle`, stall accumulates linearly —

    stall_rounds = total_rounds * (lat - cycle) / cycle      (lat > cycle)

and the program's wall clock extends by stall_rounds * cycle. In decsim
terms, a serial sliding-window chain with per-window service tau_W over
commit regions of n_com rounds has per-round latency lat = tau_W / n_com,
and the same quantity is the DRAIN TIME: fully_done - chip_done (decoder
finishing after the last syndrome round was generated). The two models
agree analytically; this test pins that the SIMULATED drain matches the
QLX formula, and that the peak backlog matches the matching closed form

    peak_backlog_rounds ~= rounds * (1 - tau_gen / tau_W)

both within one window of slack (startup/boundary effects).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.codes import SurfaceCodeModel
from decsim.config import us
from decsim.controllers import ModularController, LinkModel
from decsim.message import DecodeResult, Operation
from decsim.metrics import DecodeBacklog
from decsim.schemes import SlidingWindowScheme
from decsim.run_spec import RunSpec, simulate
from decsim.planner import FixedRounds

D = 3
N_COM = D                    # commit region rounds
ROUND_US = 1.0
GEN_US = N_COM * ROUND_US    # window generation period


class _FixedLatencyDecoder:
    def __init__(self, latency_us):
        self._t = us(latency_us)

    def latency(self, job):
        return self._t

    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(0,))


def _zero_link_controller(engine):
    return ModularController(engine, links=LinkModel(qc=0, cd=0, dd=0,
                                                     do=0, oc=0, cq=0),
                             log_syndromes=False)


def _run(latency_us, rounds):
    op = Operation(0, "mem", (0,), clifford=True, patches=(0,))
    res = simulate(RunSpec(
              ops=[op],
              num_units=4,
              rounds_policy=FixedRounds(rounds),
              round_us=ROUND_US,
              decoder=_FixedLatencyDecoder(latency_us),
              scheme=SlidingWindowScheme(),
              code=SurfaceCodeModel(d=D),
              make_controller=_zero_link_controller,
              make_metrics=lambda e, cl, ch, f: [DecodeBacklog(cl)],
          ), verbose=False)
    drain_rounds = (res.result.fully_done_ticks - res.result.chip_done_ticks) / us(ROUND_US)
    peak = res.result.metric_values()["decode_backlog"]["peak_rounds"]
    return drain_rounds, peak


@pytest.mark.parametrize("tau_w_us,rounds", [(6.0, 60), (9.0, 60),
                                             (9.0, 120)])
def test_drain_matches_qlx_stall_formula(tau_w_us, rounds):
    lat_per_round = tau_w_us / N_COM
    qlx_stall = rounds * (lat_per_round - ROUND_US) / ROUND_US
    drain, _ = _run(tau_w_us, rounds)
    tolerance = 2 * tau_w_us / ROUND_US        # startup + final-window slack
    assert abs(drain - qlx_stall) <= tolerance, \
        (f"simulated drain {drain} rounds vs QLX stall formula {qlx_stall} "
         f"(tau_W={tau_w_us}us, rounds={rounds})")


@pytest.mark.parametrize("tau_w_us,rounds", [(6.0, 60), (9.0, 120)])
def test_peak_backlog_matches_closed_form(tau_w_us, rounds):
    predicted = rounds * (1.0 - GEN_US / tau_w_us)
    _, peak = _run(tau_w_us, rounds)
    slack = 2 * N_COM + D                      # in-flight floor + boundary
    assert abs(peak - predicted) <= slack, \
        f"peak backlog {peak} vs closed form {predicted:.1f}"


def test_no_stall_when_decoder_keeps_pace():
    """lat < cycle: QLX predicts zero stall; decsim's drain must be
    bounded by ONE window's service (the final window finishing)."""
    drain, _ = _run(1.5, 60)                   # f = 0.5
    assert drain <= 1.5 + GEN_US
