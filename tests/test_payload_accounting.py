"""Syndrome-RAM accounting (cluster.payloads_held / peak_payloads).

The high-water mark is a running counter on the PayloadStore: +1 when a
payload is stored, -N when rounds/ops free. These tests check the counter
against a ground-truth live set maintained OUTSIDE the store through the
MemoryModel seam (port 18: store()/evict() fire on exactly the fragments the
store retains/frees), and that the store drains to zero at completion."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from decsim.decoders import PresetLatencyDecoder
from decsim.frontends.circuit import cnot_plus_two_t_circuit, three_cnot_circuit
from decsim.run_spec import simulate
from decsim.planner import FixedRounds
from decsim.run_spec import RunSpec


class LiveSetModel:
    """Ground truth: the set of fragments currently retained (port 18 seam)."""

    def __init__(self):
        self.live, self.peak = set(), 0

    def store(self, key, payload):
        self.live.add(key)
        self.peak = max(self.peak, len(self.live))

    def evict(self, key):
        self.live.discard(key)


def _run(ops):
    model = LiveSetModel()
    res = simulate(RunSpec(ops=ops, d=3, rounds_policy=FixedRounds(11),
                           num_units=2,
                           decoder=PresetLatencyDecoder(1.0),
                           memory_model=model))
    return res.cluster, model


def test_peak_payloads_matches_brute_force_recount():
    cluster, model = _run(three_cnot_circuit())
    assert cluster.peak_payloads == model.peak > 0
    assert cluster.payloads_held == len(model.live)


def test_accounting_with_blocked_ops_and_store_drains_to_zero():
    """Gated T gates exercise idle rounds and late window commits; afterwards every
    op's store has been freed, so an exact counter must read zero."""
    cluster, model = _run(cnot_plus_two_t_circuit())
    assert cluster.peak_payloads == model.peak > 0
    assert cluster.payloads_held == len(model.live) == 0


def test_per_window_release_holds_only_the_live_set():
    """arXiv:2511.10633 Sec VI.B: syndromes are discarded "as soon as the associated decoding
    tasks are complete". With per-window release the syndrome-RAM high-water is the LIVE set --
    ~one sliding window (commit+buffer) -- so it stays bounded as the computation grows, instead
    of scaling with the operation length (the per-op resident upper bound)."""
    from decsim.config import us
    from decsim.codes import SurfaceCodeModel
    from decsim.controllers import ModularController, LinkModel
    from decsim.message import DecodeResult, Operation
    from decsim.schemes import SlidingWindowScheme

    class _Dec:
        def latency(self, job):
            return us(1.0)
        def decode(self, job):
            return DecodeResult(job.op_id, job.window_id,
                                logical_observables=(0,))

    def _links(engine):
        return ModularController(engine, links=LinkModel(qc=0, cd=0, dd=0, do=0, oc=0, cq=0), log_syndromes=False)

    def peak_for(rounds):
        op = Operation(0, "mem", (0,), clifford=True, patches=(0,))
        res = simulate(RunSpec(
                  ops=[op],
                  num_units=4,
                  rounds_policy=FixedRounds(rounds),
                  round_us=1.0,
                  decoder=_Dec(),
                  scheme=SlidingWindowScheme(),
                  code=SurfaceCodeModel(d=3),
                  make_controller=_links,
              ), verbose=False)
        c = res.cluster
        assert c.payloads_held == 0            # drains fully
        return c.peak_payloads

    short, long = peak_for(15), peak_for(120)
    assert short == long                       # live set is independent of computation length
    assert long <= 4 * 3                       # bounded by ~one window (commit+buffer = 2d), not R


def test_round_arriving_after_op_completed_fails_loudly():
    """A payload for an op whose syndrome RAM was already freed (its last window
    committed) means the device emitted more rounds than planned -- the cluster must
    say so, not die on a KeyError or corrupt the running counter."""
    import pytest
    from decsim.message import SyndromePayload
    cluster, _ = _run(three_cnot_circuit())               # run to completion
    with pytest.raises(RuntimeError, match="syndrome RAM was freed"):
        cluster.on_syndrome_arrival(SyndromePayload(0, 0, 99))
