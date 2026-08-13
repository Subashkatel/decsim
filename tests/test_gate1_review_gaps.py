"""Focused tests for Codex Gate-1 review findings 7-10 + hidden assumptions.

Each test pins a branch the 2026-07-03 review found uncovered:
  7  held-early-strong result discarded when the weak proves confident
  8  SyndromeBuffer.replace_hold strictly frees rounds the new lease drops
  9  dem.decode_windowed raises when artificial defects are never consumed
  HA switching threshold equality keeps weak (>= semantics)
  HA SlidingWindowScheme.data_complete overflow branches
  HA ComplementaryGapMetric multi-observable guard
"""
import pytest

from decsim.decoders import SAMPLED_CONFIDENCE_SOURCE
from decsim.engine import Engine
from decsim.message import (
    DecodeJob,
    DecodeResult,
    DecoderRequestKey,
    DecoderTier,
    SoftOutput,
    SuccessorReadiness,
    Window,
    WindowReadiness,
)
from decsim.decoder_manager import StrategyServicesImpl, DecoderManager
from decsim.syndrome_buffer import SyndromeBuffer
from decsim.schemes import SlidingWindowScheme
from decsim.switching import Switching

stim = pytest.importorskip("stim")
np = pytest.importorskip("numpy")
pymatching = pytest.importorskip("pymatching")

WS = 500_000


class _FifoScheduler:
    def pop(self, queue): return queue.pop(0)


class _Decoder:
    def __init__(self, latency, soft=None, logical=0):
        self._latency = latency
        self.soft = (
            None
            if soft is None
            else SoftOutput(
                gap=soft,
                source=SAMPLED_CONFIDENCE_SOURCE,
            )
        )
        self.logical = logical
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
                         strong_decode_for=(weak_job.op_id, weak_job.window_id),
                         request_key=DecoderRequestKey(
                             weak_job.op_id, weak_job.window_id,
                             DecoderTier.STRONG, 1))
    def on_decode_done(self, job, result):
        self.commits.append((job.op_id, job.window_id, job.awaiting_strong_result))
    def on_strong_decode_done(self, completion):
        self.strong_commits.append((completion.request_key.operation_id,
                                    completion.request_key.window_id))
    def prepare_strong_selection(self, weak_job, strong_request_key,
                                 serial_strong_job, *, deferred):
        if deferred:
            raise RuntimeError("deferred selection has no pending request")
        if serial_strong_job is not None:
            self.pool.enqueue(serial_strong_job, WS)
        return WS


def _window():
    return Window(op_id=0, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)


def _pool(strategy, weak, strong):
    eng = Engine(verbose=False)
    rt = _RuntimeStub()
    pool = DecoderManager(eng, router=_Router(weak, strong),
                          scheduler=_FifoScheduler(),
                          unit_pools={"default": 1, "strong": 1})
    rt.pool = pool
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
    strat = Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5, run_both_at_once=True)
    eng, rt, pool = _pool(strat, weak, strong)
    w = _window()
    job = DecodeJob(op_id=0, window_id=0, n_rounds=6, window=w, label="op0 W0",
                    request_key=DecoderRequestKey(0, 0, DecoderTier.WEAK, 0))
    job.strong_label = "strong(op0 W0)"
    for sub in strat.on_window_ready(w, job, pool.services):
        pool.enqueue(sub.job, sub.delay_ticks)
    eng.run()
    assert rt.commits == [(0, 0, False)]        # weak committed, not awaiting
    assert rt.strong_commits == []              # held strong never applied
    assert pool._completed_strong_results == {} # held result discarded
    # Discarding a held completion cancels the request even though its service
    # time was already spent.
    assert pool.strong_cancelled == 1
    assert pool.strong_needed == 0


# ------------------------------------------------- finding 8: strict replace

def test_syndrome_buffer_replace_strictly_frees_dropped_rounds():
    from decsim.message import RetainedSyndromeFragment, SyndromePayload

    buffer = SyndromeBuffer()
    buffer.open_operation(0)
    identities = tuple((0, r) for r in (1, 2, 3))
    buffer.register_hold("L", identities)
    for r in (1, 2, 3):
        payload = SyndromePayload(0, 0, r)
        buffer.accept_fragment(
            RetainedSyndromeFragment.from_payload(payload),
            expected_fragments=1,
        )
        buffer.finish_packing((0, r), publication_tick=r)
    assert buffer.payloads_held == 3
    buffer.replace_hold("L", ((0, 3),))
    # rounds 1 and 2 lost their only lease and MUST be freed
    assert buffer.retained_fragments((0, 1)) is None
    assert buffer.retained_fragments((0, 2)) is None
    assert buffer.retained_fragments((0, 3)) is not None    # still leased
    assert buffer.payloads_held == 1
    buffer.release_hold("L")
    assert buffer.retained_fragments((0, 3)) is None
    assert buffer.payloads_held == 0


# ------------------------------------- finding 9: unconsumed-defect negative

def _surface_circuit(d=3, rounds=9, p=0.003):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def test_decode_windowed_raises_on_unconsumed_artificial_defects():
    """A plan that truncates the stream leaves committed faults' future
    flips pending; the forward-only walk must raise, not silently drop."""
    from decsim.detector_error_model import (
        FaultRepresentation,
        GRAPHLIKE_FAULT_MODEL_REQUIRED,
        build_window_error_models,
        decode_windowed,
    )

    circuit = _surface_circuit()
    # Full coverage plan is [(1,3,6),(4,6,9),(7,9,9)]; truncate to the
    # first TWO windows and force is_last=False semantics by building the
    # full plan, then dropping the tail window from the walk.
    models = build_window_error_models(
        circuit,
        [(1, 3, 6), (4, 6, 9), (7, 9, 9)],
        round_count=9,
        fault_model_requirement=GRAPHLIKE_FAULT_MODEL_REQUIRED,
        fault_exclusion_ranges=(),
    )
    graphlike_faults = [
        model.require_faults(FaultRepresentation.GRAPHLIKE)
        for model in models[:2]
    ]
    assert any(faults.future_flips for faults in graphlike_faults)
    matchings = [pymatching.Matching.from_check_matrix(
        faults.check,
        weights=np.log((1 - faults.priors) / faults.priors),
        faults_matrix=np.eye(
            np.asarray(faults.check).shape[1],
            dtype=np.uint8,
        ))
        for faults in graphlike_faults]   # per-column selections (G9 review fix)

    def decode_window(model, syndrome):
        idx = models.index(model)
        return matchings[idx].decode(syndrome)

    dets = circuit.compile_detector_sampler(seed=7).sample(shots=64)
    saw_defect_shot = False
    for shot in dets:
        # find a shot whose first-two-window decode commits a future flip
        try:
            decode_windowed(
                list(models[:2]),
                shot,
                decode_window,
                selected_fault_representation=FaultRepresentation.GRAPHLIKE,
            )
        except RuntimeError as err:
            assert "artificial defects were never consumed" in str(err)
            saw_defect_shot = True
            break
    assert saw_defect_shot, ("no shot committed a boundary-crossing fault in "
                             "64 seeded shots; raise the shot count")


# ----------------------------------------- hidden assumption: >= threshold

def test_switching_threshold_equality_keeps_weak():
    strat = Switching(expected_source=SAMPLED_CONFIDENCE_SOURCE, confidence_threshold=0.5)
    assert strat.keep_weak_result(
        DecodeResult(
            0,
            0,
            soft_output=SoftOutput(
                gap=0.5,
                source=SAMPLED_CONFIDENCE_SOURCE,
            ),
        ),
        None,
    ) is True
    assert strat.keep_weak_result(
        DecodeResult(
            0,
            0,
            soft_output=SoftOutput(
                gap=0.4999,
                source=SAMPLED_CONFIDENCE_SOURCE,
            ),
        ),
        None,
    ) is False
    assert strat.keep_weak_result(
        DecodeResult(0, 0, soft_output=None), None) is False
    assert strat.keep_weak_result(None, None) is False


# ------------------------------- hidden assumption: data_complete overflow

def test_data_complete_overflow_branches():
    window = Window(0, 1, 4, 6, 9, 9)
    live = (SuccessorReadiness(9, 2, 9),)
    cases = (
        (5, live, 3, False, False),  # local commit data missing
        (6, (), 0, False, True),     # terminal operation
        (6, live, 2, False, False),  # live successor is short
        (6, (SuccessorReadiness(9, 3, 9),), 0, False, True),
        (6, live, 3, False, True),   # memory covers overflow
        (6, (SuccessorReadiness(9, 2, 2),), 0, False, True),
        (6, (SuccessorReadiness(9, 2, 2),
             SuccessorReadiness(10, 1, 6)), 0, False, False),
        (6, live, 0, True, True),    # explicit closed tail
    )
    for local, successors, memory, closed, expected in cases:
        readiness = WindowReadiness(local, 6, successors, memory, closed)
        assert SlidingWindowScheme().data_complete(
            window, readiness=readiness, operation=None) is expected


# --------------------------- hidden assumption: multi-observable guard

def test_complementary_gap_rejects_multiple_observables():
    from decsim.soft_output import ComplementaryGapMetric

    check = np.eye(2, dtype=np.uint8)
    obs = np.ones((2, 2), dtype=np.uint8)       # two observables -> invalid
    with pytest.raises(ValueError, match="one observable"):
        ComplementaryGapMetric(check, obs, np.ones(2))
