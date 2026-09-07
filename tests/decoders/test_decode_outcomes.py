"""The outcomes' laws: keep cancels the strong, a result reaches once.

Toshio et al. 2510.25222: the weak result is kept or the window waits
for its strong re-decode; SimPy's callback on the event (simpy/core.py)
is the shape of on_decoded.
"""

import decsim.decoders.decode_outcomes as decode_outcomes
import decsim.decoders.strong_requests as strong_requests_module
import decsim.engine as engine_module
import decsim.observe.decode_records as decode_records
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


class _Policy:
    """Answers every weak result with one verdict; remembers what it learns."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.learned = []

    def verdict_for_weak_result(self, job, result):
        del job
        del result
        return self.verdict

    def learn_from_strong_result(self, window_key, result):
        self.learned.append((window_key, result))


def _outcomes(verdict, cancelled):
    engine = engine_module.Engine()
    requests = strong_requests_module.StrongRequests()
    policy = _Policy(verdict)
    outcomes = decode_outcomes.DecodeOutcomes(
        engine,
        policy,
        requests,
        cancel_strong=cancelled.append,
    )
    return outcomes, requests, policy


def _delivering_to(delivered):
    """An on_decoded that keeps each (job, result) it is handed."""

    def on_decoded(job, result):
        delivered.append((job, result))

    return on_decoded


def _weak_job(delivered):
    on_decoded = _delivering_to(delivered)
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, on_decoded=on_decoded
    )


def test_a_kept_weak_result_cancels_the_live_strong_request():
    cancelled = []
    outcomes, requests, _policy = _outcomes(
        decoding_records.Verdict.KEEP, cancelled
    )
    delivered = []
    job = _weak_job(delivered)
    requests.admit(job, now=0)
    result = decoding_records.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    assert cancelled == [(1, 0)]
    assert job.awaiting_strong_result is False
    assert delivered == [(job, result)]


def test_an_escalated_weak_result_reaches_on_decoded_awaiting_the_strong():
    cancelled = []
    outcomes, requests, _policy = _outcomes(
        decoding_records.Verdict.ESCALATE, cancelled
    )
    delivered = []
    job = _weak_job(delivered)
    requests.admit(job, now=0)
    result = decoding_records.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    assert cancelled == []
    assert job.awaiting_strong_result is True
    assert delivered == [(job, result)]
    assert (1, 0) not in requests.unresolved_weak_windows


def test_a_strong_result_teaches_the_policy_and_reaches_its_destination_once():
    cancelled = []
    outcomes, requests, policy = _outcomes(
        decoding_records.Verdict.KEEP, cancelled
    )
    delivered = []
    strong_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 5
    )
    on_decoded = _delivering_to(delivered)
    strong_job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=9,
        strong_decode_for=(1, 0),
        request_key=strong_key,
        on_decoded=on_decoded,
    )
    requests.admit_strong(strong_job, now=0)
    requests.begin_selection((1, 0), strong_key)
    requests.select((1, 0), strong_key)
    result = decoding_records.DecodeResult(1, 0, logical_observables=(1,))
    deliveries = requests.deliveries_for(strong_job, result, now=30)
    outcomes.conclude_strong(strong_job, result, deliveries)
    assert delivered == [(strong_job, result)]
    assert policy.learned == [((1, 0), result)]
    assert requests.unsettled() == {}


def test_a_selection_that_lands_after_the_strong_result_releases_it():
    cancelled = []
    outcomes, requests, _policy = _outcomes(
        decoding_records.Verdict.KEEP, cancelled
    )
    delivered = []
    strong_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, 5
    )
    on_decoded = _delivering_to(delivered)
    strong_job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=9,
        strong_decode_for=(1, 0),
        request_key=strong_key,
        on_decoded=on_decoded,
    )
    requests.admit_strong(strong_job, now=0)
    requests.begin_selection((1, 0), strong_key)
    result = decoding_records.DecodeResult(1, 0, logical_observables=(1,))
    deliveries = requests.deliveries_for(strong_job, result, now=30)
    outcomes.conclude_strong(strong_job, result, deliveries)
    assert delivered == []
    outcomes.select_strong_result((1, 0), strong_key)
    assert delivered == [(strong_job, result)]
    assert requests.unsettled() == {}


def test_the_terminal_sources_carry_every_ended_request_and_service():
    """The record ledger connects and hears both; nothing else is needed."""
    cancelled = []
    outcomes, requests, _policy = _outcomes(
        decoding_records.Verdict.KEEP, cancelled
    )
    ledger = decode_records.DecodeRecordLedger()
    outcomes.request_ended.connect(ledger.request_ended)
    outcomes.service_ended.connect(ledger.service_ended)
    delivered = []
    job = _weak_job(delivered)
    job.request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    job.service_key = decoding_records.DecoderServiceKey(0)
    job.service_original_request_keys = (job.request_key,)
    job.service_dispatch_ticks = 0
    job.window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
    )
    requests.admit(job, now=0)
    result = decoding_records.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    (request,) = ledger.requests
    (service,) = ledger.services
    assert request.request_key == job.request_key
    assert service.completed_request_keys == (job.request_key,)


def test_a_run_with_no_listener_concludes_the_same_way():
    """Rule 7: the outcomes work with nothing connected to their sources."""
    cancelled = []
    outcomes, requests, policy = _outcomes(
        decoding_records.Verdict.KEEP, cancelled
    )
    delivered = []
    job = _weak_job(delivered)
    requests.admit(job, now=0)
    result = decoding_records.DecodeResult(1, 0)
    outcomes.conclude_weak(job, result)
    assert delivered == [(job, result)]
    assert policy.learned == []
