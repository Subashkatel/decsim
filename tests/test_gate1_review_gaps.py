"""Focused tests for Codex Gate-1 review findings 7-10 + hidden assumptions.

Each test pins a branch the 2026-07-03 review found uncovered:
  7  held-early-strong result discarded when the weak proves confident
  8  PayloadStore.replace strictly frees rounds the new lease drops
  9  dem.decode_windowed raises when artificial defects are never consumed
  10 ClusterGapMetric.from_window_model + the SoftOutputDecoder wrapper path
  HA switching threshold equality keeps weak (>= semantics)
  HA SlidingWindowScheme.data_complete overflow branches
  HA ComplementaryGapMetric multi-observable guard
"""
import pytest

from decsim.engine import Engine
from decsim.message import DecodeJob, DecodeResult, SyndromePayload, Window
from decsim.decoder_manager import StrategyServicesImpl, DecoderManager
from decsim.payload_store import PayloadStore
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

WS = 500_000


class _FifoScheduler:
    def insert(self, queue, job): queue.append(job)
    def pop(self, queue, now_ticks): return queue.pop(0)


class _Decoder:
    def __init__(self, latency, soft=None, logical=0):
        self._latency, self.soft, self.logical = latency, soft, logical
    def latency(self, job): return self._latency
    def decode(self, job):
        return DecodeResult(job.op_id, job.window_id,
                            logical_observables=(self.logical,),
                            soft_output=self.soft)


class _Router:
    def __init__(self, weak, strong): self.weak, self.strong = weak, strong
    def route(self, job): return self.strong if job.hint == "strong" else self.weak


class _RuntimeStub:
    def __init__(self):
        self.commits, self.strong_commits = [], []
    def make_strong_decode_job(self, weak_job, n_rounds, label):
        return DecodeJob(op_id=weak_job.op_id, window_id=weak_job.window_id,
                         n_rounds=n_rounds, label=label, hint="strong",
                         attempt=1, window=weak_job.window,
                         strong_decode_for=(weak_job.op_id, weak_job.window_id))
    def on_decode_done(self, job, result):
        self.commits.append((job.op_id, job.window_id, job.awaiting_strong_result))
    def on_strong_decode_done(self, key, result):
        self.strong_commits.append(key)


def _window():
    return Window(op_id=0, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)


def _pool(strategy, weak, strong):
    eng = Engine(verbose=False)
    rt = _RuntimeStub()
    pool = DecoderManager(eng, router=_Router(weak, strong),
                          scheduler=_FifoScheduler(),
                          unit_pools={"default": 1, "strong": 1},
                          ws_delay_ticks=WS)
    pool.strategy = strategy
    pool.services = StrategyServicesImpl(eng, rt, pool)
    pool.on_window_decoded = rt.on_decode_done
    pool.on_strong_window_decoded = rt.on_strong_decode_done
    return eng, rt, pool


# ---------------------------------------------------- finding 7: held discard

def test_held_early_strong_discarded_on_confident_weak():
    """Parallel mode: strong finishes FIRST (held), weak then proves
    confident -> FINALIZE cancels the held result; it must never apply."""
    weak = _Decoder(10_000, soft=0.9)           # slow weak, HIGH confidence
    strong = _Decoder(10, logical=1)            # strong completes early -> held
    strat = Switching(confidence_threshold=0.5, run_both_at_once=True)
    eng, rt, pool = _pool(strat, weak, strong)
    w = _window()
    job = DecodeJob(op_id=0, window_id=0, n_rounds=6, window=w, label="op0 W0")
    job.strong_label = "strong(op0 W0)"
    for sub in strat.on_window_ready(w, job, pool.services):
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    assert rt.commits == [(0, 0, False)]        # weak committed, not awaiting
    assert rt.strong_commits == []              # held strong never applied
    assert pool._completed_strong_results == {} # held result discarded
    # Counter semantics (documented, not a bug): strong_cancelled counts
    # queued/running cancellations only; a held COMPLETED result that gets
    # discarded is invisible to it (the strong unit's service time was
    # spent). Utilization/gamma accounting must use strong_needed +
    # completions, not strong_cancelled, for the parallel-mode discard case.
    assert pool.strong_cancelled == 0
    assert pool.strong_needed == 0


# ------------------------------------------------- finding 8: strict replace

def test_payload_store_replace_strictly_frees_dropped_rounds():
    ps = PayloadStore()
    ps.register_op(0)
    for r in (1, 2, 3):
        ps.store(0, r, payload=f"round{r}")
    ps.lease("L", [(0, 1), (0, 2), (0, 3)])
    assert ps.payloads_held == 3
    ps.replace("L", [(0, 3)])
    # rounds 1 and 2 lost their only lease and MUST be freed
    assert ps.fragments(0, 1) is None
    assert ps.fragments(0, 2) is None
    assert ps.fragments(0, 3) is not None       # still leased
    assert ps.payloads_held == 1
    ps.release("L")
    assert ps.fragments(0, 3) is None
    assert ps.payloads_held == 0


# ------------------------------------- finding 9: unconsumed-defect negative

def _surface_circuit(d=3, rounds=9, p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def test_decode_windowed_raises_on_unconsumed_artificial_defects():
    """A plan that truncates the stream leaves committed faults' future
    flips pending; the forward-only walk must raise, not silently drop."""
    from decsim.detector_error_model import build_window_error_models, decode_windowed

    circuit = _surface_circuit()
    # Full coverage plan is [(1,3,6),(4,6,9),(7,9,9)]; truncate to the
    # first TWO windows and force is_last=False semantics by building the
    # full plan, then dropping the tail window from the walk.
    models = build_window_error_models(circuit, [(1, 3, 6), (4, 6, 9), (7, 9, 9)])
    assert any(m.future_flips for m in models[:2])
    matchings = [pymatching.Matching.from_check_matrix(
        m.check, weights=np.log((1 - m.priors) / m.priors),
        faults_matrix=np.eye(np.asarray(m.check).shape[1], dtype=np.uint8))
        for m in models[:2]]   # per-column selections (G9 review fix)

    def decode_window(model, syndrome):
        idx = models.index(model)
        return matchings[idx].decode(syndrome)

    dets = circuit.compile_detector_sampler(seed=7).sample(shots=64)
    saw_defect_shot = False
    for shot in dets:
        # find a shot whose first-two-window decode commits a future flip
        try:
            decode_windowed(list(models[:2]), shot, decode_window)
        except RuntimeError as err:
            assert "artificial defects were never consumed" in str(err)
            saw_defect_shot = True
            break
    assert saw_defect_shot, ("no shot committed a boundary-crossing fault in "
                             "64 seeded shots; raise the shot count")


# ------------------------- finding 10: cluster gap from window model + wrapper

def test_cluster_gap_from_window_model_and_soft_output_decoder_path():
    from decsim.detector_error_model import build_window_error_models
    from decsim.soft_output import ClusterGapMetric, SoftOutputDecoder

    circuit = _surface_circuit(rounds=3)
    (model,) = build_window_error_models(circuit, [(1, 3, 3)])
    metric = ClusterGapMetric.from_window_model(model)

    empty = np.zeros(model.check.shape[0], dtype=np.uint8)
    out = metric.evaluate(empty)
    assert out.logical_value == 0
    assert out.gap > 0.0                        # confident on empty syndrome
    single = empty.copy()
    single[0] = 1
    out_single = metric.evaluate(single)
    assert out_single.gap >= 0.0                # finite, defined

    # wrapper path: SoftOutputDecoder attaches the cluster gap to a result
    base = _Decoder(10, logical=0)
    wrapper = SoftOutputDecoder(base, ClusterGapMetric)
    payload = SyndromePayload(0, 0, 1, bits=empty)
    job = DecodeJob(op_id=0, window_id=0, n_rounds=3, dem=model,
                    payloads=[payload], label="op0 W0")
    result = wrapper.decode(job)
    assert result.soft_output == pytest.approx(out.gap)


# ----------------------------------------- hidden assumption: >= threshold

def test_switching_threshold_equality_keeps_weak():
    strat = Switching(confidence_threshold=0.5)
    assert strat.keep_weak_result(
        DecodeResult(0, 0, soft_output=0.5), None) is True
    assert strat.keep_weak_result(
        DecodeResult(0, 0, soft_output=0.4999), None) is False
    assert strat.keep_weak_result(
        DecodeResult(0, 0, soft_output=None), None) is False
    assert strat.keep_weak_result(None, None) is False


# ------------------------------- hidden assumption: data_complete overflow

def test_data_complete_overflow_branches():
    scheme = SlidingWindowScheme()
    w = Window(op_id=0, k=1, commit_lo=4, commit_hi=6, buffer_hi=9, n_rounds=9)
    round_count = 6                              # overflow = 9 - 6 = 3
    common = dict(round_count=round_count, operation=None)
    # not enough in-op data yet -> False regardless of overflow sources
    assert not scheme.data_complete(w, rounds_arrived=5, successor_rounds=9,
                                    memory_rounds=9, has_successor=True, **common)
    # overflow with NO successor -> complete once in-op data present
    assert scheme.data_complete(w, rounds_arrived=6, successor_rounds=0,
                                memory_rounds=0, has_successor=False, **common)
    # successor exists: successor data covers overflow -> complete
    assert scheme.data_complete(w, rounds_arrived=6, successor_rounds=3,
                                memory_rounds=0, has_successor=True, **common)
    # successor exists: memory rounds cover overflow -> complete
    assert scheme.data_complete(w, rounds_arrived=6, successor_rounds=0,
                                memory_rounds=3, has_successor=True, **common)
    # successor exists but neither source covers overflow -> incomplete
    assert not scheme.data_complete(w, rounds_arrived=6, successor_rounds=2,
                                    memory_rounds=2, has_successor=True, **common)


# --------------------------- hidden assumption: multi-observable guard

def test_complementary_gap_rejects_multiple_observables():
    from decsim.soft_output import ComplementaryGapMetric

    check = np.eye(2, dtype=np.uint8)
    obs = np.ones((2, 2), dtype=np.uint8)       # two observables -> invalid
    with pytest.raises(ValueError, match="one observable"):
        ComplementaryGapMetric(check, obs, np.ones(2))
