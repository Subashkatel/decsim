"""The outcomes' laws: keep cancels the strong, a result reaches once.

Toshio et al. 2510.25222: the weak result is kept or the window waits
for its strong re-decode; SimPy's callback on the event (simpy/core.py)
is the shape of on_decoded.
"""

import decsim.decoders.decode_outcomes as decode_outcomes
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.message as message
import decsim.observe.decode_records as decode_records


class _Policy:
    """Keeps every weak result, or holds every one for the strong tier."""

    def __init__(self, directive):
        self.directive = directive

    def on_decode_outcome(self, outcome, services):
        del outcome
        del services
        return message.OutcomeDirective(self.directive)


class _Services:
    """Delivers the selection at once."""

    def prepare_strong_selection(
        self,
        weak_job,
        strong_request_key,
        serial_strong_job,
        *,
        deferred,
        on_selection_delivered,
    ):
        del weak_job
        del strong_request_key
        del serial_strong_job
        del deferred
        on_selection_delivered()


def _outcomes(directive, cancelled):
    engine = engine_module.Engine(verbose=False)
    requests = strong_requests_module.StrongRequests()
    records = decode_records.DecodeRecordLedger(is_enabled=False)
    policy = _Policy(directive)
    services = _Services()
    outcomes = decode_outcomes.DecodeOutcomes(
        engine,
        policy,
        services,
        requests,
        records,
        cancel_strong=cancelled.append,
    )
    return outcomes, requests


def _delivering_to(delivered):
    """An on_decoded that keeps each (job, result) it is handed."""

    def on_decoded(job, result):
        delivered.append((job, result))

    return on_decoded


def _weak_job(delivered):
    on_decoded = _delivering_to(delivered)
    return message.DecodeJob(
        op_id=1, window_id=0, n_rounds=3, on_decoded=on_decoded
    )


def test_a_kept_weak_result_cancels_the_live_strong_request():
    cancelled = []
    outcomes, requests = _outcomes(message.Directive.FINALIZE, cancelled)
    delivered = []
    job = _weak_job(delivered)
    requests.admit(job, now=0)
    result = message.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    assert cancelled == [(1, 0)]
    assert job.awaiting_strong_result is False
    assert delivered == [(job, result)]


def test_a_weak_result_held_for_the_strong_tier_still_reaches_on_decoded():
    cancelled = []
    outcomes, requests = _outcomes(message.Directive.AWAIT_STRONG, cancelled)
    delivered = []
    job = _weak_job(delivered)
    requests.admit(job, now=0)
    strong_key = message.DecoderRequestKey(1, 0, message.DecoderTier.STRONG, 5)
    strong_job = message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=9,
        strong_decode_for=(1, 0),
        request_key=strong_key,
    )
    requests.admit_strong(strong_job, now=0)
    result = message.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    assert cancelled == []
    assert job.awaiting_strong_result is True
    assert delivered == [(job, result)]
    assert requests.counts.needed == 1


def test_a_strong_result_reaches_its_destination_once():
    cancelled = []
    outcomes, requests = _outcomes(message.Directive.FINALIZE, cancelled)
    delivered = []
    strong_key = message.DecoderRequestKey(1, 0, message.DecoderTier.STRONG, 5)
    on_decoded = _delivering_to(delivered)
    strong_job = message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=9,
        strong_decode_for=(1, 0),
        request_key=strong_key,
        on_decoded=on_decoded,
    )
    requests.admit_strong(strong_job, now=0)
    requests.begin_selection((1, 0), strong_key)
    requests.select((1, 0), strong_key)
    result = message.DecodeResult(1, 0, logical_observables=(1,))
    deliveries = requests.deliveries_for(strong_job, result, now=30)
    outcomes.conclude_strong(strong_job, result, deliveries)
    assert delivered == [(strong_job, result)]
    assert requests.unsettled() == {}
