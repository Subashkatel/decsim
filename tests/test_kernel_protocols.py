"""Port protocols are runtime-checkable and the seam types carry the contract."""
from decsim.message import DecodeJob, DecodeOutcome, DecodeResult, Window
from decsim.protocols import (BoundaryPolicy, DecodingStrategy, Directive,
                          OutcomeDirective, RoundsPolicy, StrategyServices,
                          Submission)


class _FakeServices:
    now = 0
    def make_strong_job(self, weak_job, n_rounds, label): return weak_job
    def cancel_strong(self, key): pass
    def ws_delay(self): return 500_000


class _FakeStrategy:
    def on_window_ready(self, window, weak_job, services):
        return [Submission(weak_job)]
    def on_decode_outcome(self, outcome, services):
        return OutcomeDirective(Directive.FINALIZE)
    def metrics(self): return {}


class _FakeBoundary:
    def on_commit(self, window, final): return True


class _FakeRounds:
    def rounds_for(self, op, code): return 11


def test_protocols_are_structural():
    assert isinstance(_FakeStrategy(), DecodingStrategy)
    assert isinstance(_FakeServices(), StrategyServices)
    assert isinstance(_FakeBoundary(), BoundaryPolicy)
    assert isinstance(_FakeRounds(), RoundsPolicy)


def test_directive_has_exactly_three_members():
    assert {d.name for d in Directive} == {"FINALIZE", "AWAIT_STRONG",
                                           "FINALIZE_STRONG"}


def test_seam_types_flow():
    job = DecodeJob(1, 0, 11)
    w = Window(op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=6, n_rounds=6)
    subs = _FakeStrategy().on_window_ready(w, job, _FakeServices())
    assert subs[0].job is job and subs[0].delay_ticks == 0
    directive = _FakeStrategy().on_decode_outcome(
        DecodeOutcome(job, DecodeResult(1, 0)), _FakeServices())
    assert directive.directive is Directive.FINALIZE and directive.extra is None
