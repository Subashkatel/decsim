"""Factory throughput cross-check vs QLX's analytic model (gap G4).

QLX (qlx.estimate.analytical.throughput.compute_factory_throughput) models
the 15-to-1 T factory (DISTILL_15TO1_T: cycles_per_attempt=120,
pipeline_depth=4) as depth * p_acc / cycles states/round with
p_acc = 1 - 15*p_phys (container-verified 2026-07-03):

    p_phys=1e-4 -> 0.033283   p_phys=1e-3 -> 0.032833   p_phys=1e-2 -> 0.028333

decsim's DistillationFactory simulates the same shape (num_units
pipelined attempts, cycle_ticks each, Bernoulli(p_success) acceptance,
correction decode then delivery). Under matched parameters and sustained
demand, the simulated fulfillment rate must land on the analytic value up
to pipeline fill/drain edge effects and seeded acceptance noise.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from decsim.config import us
from decsim.engine import Engine
from decsim.factories import DistillationFactory

ROUND = us(1)
CYCLES = 120          # DISTILL_15TO1_T.cycles_per_attempt
DEPTH = 4             # DISTILL_15TO1_T.pipeline_depth
QLX_ANALYTIC = {1e-4: 0.033283, 1e-3: 0.032833, 1e-2: 0.028333}


class ImmediateService:
    def submit_decode(self, round_count, on_done, label="", deadline=None,
                      code=None, spatial_nodes=None):
        on_done()


@pytest.mark.parametrize("p_phys", sorted(QLX_ANALYTIC))
def test_simulated_throughput_matches_qlx_analytic(p_phys):
    demand = 200
    eng = Engine(verbose=False)
    factory = DistillationFactory(
        eng, num_units=DEPTH, cycle_ticks=CYCLES * ROUND,
        decode_service=ImmediateService(), corr_rounds=1, n_corr=11,
        p_success=1.0 - 15.0 * p_phys, seed=7)
    delivered = []
    for request_index in range(demand):
        factory.request(request_index, lambda: delivered.append(eng.now))
    eng.run()
    assert len(delivered) == demand
    total_rounds = delivered[-1] / ROUND
    simulated = demand / total_rounds
    analytic = QLX_ANALYTIC[p_phys]
    assert abs(simulated - analytic) / analytic < 0.03, \
        (f"p_phys={p_phys}: simulated {simulated:.6f} vs QLX analytic "
         f"{analytic:.6f} ({abs(simulated-analytic)/analytic:.1%} off)")


def test_stall_accounting_under_starved_demand():
    """One extra request beyond a full pipeline stalls for ~one more cycle;
    the factory's total_stall accounting must see it."""
    eng = Engine(verbose=False)
    factory = DistillationFactory(
        eng, num_units=1, cycle_ticks=CYCLES * ROUND,
        decode_service=ImmediateService(), corr_rounds=1, n_corr=11,
        p_success=1.0, seed=1)
    done = []
    factory.request(0, lambda: done.append(eng.now))
    factory.request(1, lambda: done.append(eng.now))
    eng.run()
    assert len(done) == 2
    assert done[0] / ROUND == CYCLES            # first state after one cycle
    assert done[1] / ROUND == 2 * CYCLES        # second waited a full cycle
    assert factory.total_stall > 0
