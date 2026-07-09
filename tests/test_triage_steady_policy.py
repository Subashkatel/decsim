"""Triage steady-mode priority policy (V9-caveat partial closure).

WeightedUrgencyCostScheduler = the decsim mapping of Triage Eq. 2
(arXiv:2605.04459), steady mode ONLY (dual-mode emergency + MDF are
explicitly out of scope — see the class docstring). Tests pin the
ordering semantics and the two degenerate limits.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.decoders import CodeRouter, PerRoundDecoder
from decsim.decoder_manager import DecoderManager
from decsim.schedulers import WeightedUrgencyCostScheduler


def build(w_u, w_c):
    eng = Engine(verbose=False)
    manager = DecoderManager(
        eng, router=CodeRouter(default=PerRoundDecoder(tau_us=1.0)),
        scheduler=WeightedUrgencyCostScheduler(eng, w_u=w_u, w_c=w_c),
        unit_pools={"default": 1})
    return eng, manager


def run_order(w_u, w_c, jobs):
    """jobs: (label, rounds, deadline_us); one busy head, rest queue."""
    eng, manager = build(w_u, w_c)
    order = []
    manager.submit_decode(50, lambda: order.append("head"), label="head",
                          deadline=us(1))
    for label, rounds, dl in jobs:
        manager.submit_decode(rounds, lambda l=label: order.append(l),
                              label=label, deadline=us(dl))
    eng.run()
    assert order[0] == "head"
    return order[1:]


def test_weights_must_sum_to_one():
    eng = Engine(verbose=False)
    with pytest.raises(ValueError, match="must be 1"):
        WeightedUrgencyCostScheduler(eng, w_u=0.7, w_c=0.7)


def test_pure_urgency_orders_by_nearest_deadline():
    """w_c = 0 degenerates to EDF among unexpired jobs."""
    jobs = [("a", 5, 400), ("b", 5, 100), ("c", 5, 300), ("d", 5, 200)]
    assert run_order(1.0, 0.0, jobs) == ["b", "d", "c", "a"]


def test_pure_cost_orders_by_shortest_job():
    """w_u = 0 degenerates to shortest-job-first."""
    jobs = [("a", 8, 100), ("b", 2, 400), ("c", 12, 200), ("d", 4, 300)]
    assert run_order(0.0, 1.0, jobs) == ["b", "d", "a", "c"]


def test_mixed_weights_trade_urgency_against_cost():
    """A cheap far-deadline job outranks an expensive near-deadline one
    when cost weight dominates, and vice versa. Deadlines in ticks are
    large, so urgency ~ w_u/slack is tiny vs w_c/rounds: pin the
    regime explicitly at both extremes of the weighted blend."""
    jobs = [("cheap_far", 1, 10_000), ("costly_near", 12, 60)]
    assert run_order(0.01, 0.99, jobs) == ["cheap_far", "costly_near"]
    assert run_order(1.0, 0.0, jobs) == ["costly_near", "cheap_far"]


def test_expired_jobs_share_the_floor_slack_and_fifo_tiebreak():
    """Jobs past deadline clamp to 1-tick slack: equal urgency, so
    equal-cost expired jobs serve in FIFO order (stable -i tiebreak)."""
    jobs = [("x", 5, 1), ("y", 5, 1), ("z", 5, 1)]
    assert run_order(1.0, 0.0, jobs) == ["x", "y", "z"]
