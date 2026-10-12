"""One unconsumed result per destination; the unit holds it; batches split.

Toshio et al. 2510.25222: one strong re-decode per escalated window,
selected by the weak side and consumed by the window that asked.
"""

import pytest

import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


def _request_key(sequence):
    return window_records.DecoderRequestKey(
        1, 0, window_records.STRONG_POOL_ID, sequence
    )


def _strong_job(request_key, window_key=(1, 0)):
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=9,
        strong_decode_for=window_key,
        request_key=request_key,
    )


def _completion(job, now=50):
    result = decoding_records.DecodeResult(1, 0)
    return strong_requests_module.StrongCompletion(job, result, now, None)


def test_a_result_before_its_selection_is_not_consumed_yet():
    """The ledger takes no result: the unit that produced it holds it."""
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.begin_selection((1, 0), key)
    requests.finish_service(job)
    completion = _completion(job)
    assert requests.complete(completion) is False
    assert requests.select((1, 0), key) is True
    assert requests.complete(completion) is True


def test_a_selection_before_the_result_consumes_it_at_once():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.begin_selection((1, 0), key)
    assert requests.select((1, 0), key) is True
    requests.finish_service(job)
    completion = _completion(job)
    assert requests.complete(completion) is True
    assert requests.counts.needed == 1


def test_a_selection_the_destination_never_sent_is_ignored():
    """Only the request key the destination selected releases a result."""
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    assert requests.select((1, 0), key) is False
    requests.begin_selection((1, 0), key)
    other_key = _request_key(9)
    assert requests.select((1, 0), other_key) is False
    assert requests.select((1, 0), key) is True


def test_a_stale_result_is_refused_once_a_newer_request_owns_the_window():
    requests = strong_requests_module.StrongRequests()
    old_key = _request_key(7)
    old_job = _strong_job(old_key)
    requests.admit_strong(old_job, now=0)
    requests.take_live((1, 0), window_records.STRONG_POOL_ID)
    new_key = _request_key(9)
    new_job = _strong_job(new_key)
    requests.admit_strong(new_job, now=1)
    stale = _completion(old_job)
    with pytest.raises(RuntimeError, match="arrived after a newer strong"):
        requests.complete(stale)


def test_a_result_nobody_waits_for_is_refused():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.finish_service(job)
    orphan = _completion(job)
    with pytest.raises(RuntimeError, match="the destination registered"):
        requests.complete(orphan)


def test_a_destination_keeps_at_most_one_unconsumed_strong_result():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(7)
    first = _strong_job(first_key)
    requests.admit_strong(first, now=0)
    second_key = _request_key(8)
    second = _strong_job(second_key)
    with pytest.raises(RuntimeError, match="a destination window has at most"):
        requests.admit_strong(second, now=1)


def test_a_windows_attempt_holds_its_forced_class_requests():
    """One attempt, one or two requests, resolved as one.

    The complementary gap decodes a window once per logical class, so
    the window's open attempt names both request keys and resolves as
    one when the join hands the window's answer on.
    """
    requests = strong_requests_module.StrongRequests()
    first_key = window_records.DecoderRequestKey(
        1, 0, window_records.WEAK_POOL_ID, 0
    )
    second_key = window_records.DecoderRequestKey(
        1, 0, window_records.WEAK_POOL_ID, 1
    )
    first = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, request_key=first_key
    )
    second = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, request_key=second_key
    )
    requests.admit(first, now=0)
    requests.admit(second, now=0)
    (record,) = requests.by_window.values()
    assert record.open_weak_requests == {first_key, second_key}
    requests.resolve_weak((1, 0), window_records.WEAK_POOL_ID)
    assert not requests.by_window


def test_a_cancelled_middle_request_closes_its_attempt_on_the_next_edge():
    """A chain's middle job opens an attempt; a cancel ends it with it."""
    requests = _chain_requests()
    request_key = _request_key(7)
    job = _strong_job(request_key)
    requests.admit_strong(job, now=0)

    requests.take_live((1, 0), window_records.STRONG_POOL_ID)

    assert requests.by_window == {}


def test_an_abandoned_middle_result_closes_its_attempt_when_dropped():
    requests = _chain_requests()
    request_key = _request_key(7)
    job = _strong_job(request_key)
    requests.admit_strong(job, now=0)
    requests.abandon((1, 0), window_records.STRONG_POOL_ID)
    requests.finish_service(job)
    completion = _completion(job)

    is_dropped = requests.drop_abandoned(completion)

    assert is_dropped
    assert requests.by_window == {}


def _companion_of(lead, sequence):
    """The lead's solve in the other logical class, reading its input."""
    request_key = _request_key(sequence)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=9,
        strong_decode_for=(1, 0),
        request_key=request_key,
        input_key=lead.request_key,
        forced_logical_class=1,
    )


def test_a_selection_waits_for_each_forced_class_solve_of_its_attempt():
    """One selection names the attempt; both of its solves reach it."""
    requests = strong_requests_module.StrongRequests()
    lead_key = _request_key(7)
    lead = _strong_job(lead_key)
    companion = _companion_of(lead, 8)
    requests.admit_strong(lead, now=0)
    requests.admit_strong(companion, now=0)
    requests.begin_selection((1, 0), lead_key)
    requests.select((1, 0), lead_key)
    requests.finish_service(lead)
    requests.finish_service(companion)
    lead_completion = _completion(lead)
    companion_completion = _completion(companion)

    takes_lead = requests.complete(lead_completion)
    still_waiting = requests.destination_may_consume(
        (1, 0), window_records.STRONG_POOL_ID
    )
    takes_companion = requests.complete(companion_completion)

    assert takes_lead
    assert still_waiting
    assert takes_companion
    assert requests.by_window == {}


def test_a_companion_admitted_after_its_selection_is_owed_too():
    """A held attempt is selected before its jobs are built."""
    requests = strong_requests_module.StrongRequests()
    lead_key = _request_key(7)
    lead = _strong_job(lead_key)
    companion = _companion_of(lead, 8)
    requests.begin_selection((1, 0), lead_key)
    requests.select((1, 0), lead_key)
    requests.admit_strong(lead, now=0)
    requests.admit_strong(companion, now=0)
    requests.finish_service(lead)
    requests.finish_service(companion)
    lead_completion = _completion(lead)
    companion_completion = _completion(companion)
    requests.complete(lead_completion)

    takes_companion = requests.complete(companion_completion)

    assert takes_companion
    assert requests.by_window == {}


def test_a_cancel_takes_every_solve_of_the_attempt():
    requests = strong_requests_module.StrongRequests()
    lead_key = _request_key(7)
    lead = _strong_job(lead_key)
    companion = _companion_of(lead, 8)
    requests.admit_strong(lead, now=0)
    requests.admit_strong(companion, now=0)

    taken = requests.take_live((1, 0), window_records.STRONG_POOL_ID)

    taken_jobs = [live.request_job for live in taken]
    assert taken_jobs == [lead, companion]
    assert requests.by_window == {}


def _chain_requests():
    """The ledger of a chain whose strong pool escalates on to deep."""
    deep = window_records.PoolId("deep")
    target_by_source = {
        window_records.WEAK_POOL_ID: window_records.STRONG_POOL_ID,
        window_records.STRONG_POOL_ID: deep,
    }
    return strong_requests_module.StrongRequests(target_by_source)


def test_a_merged_batch_splits_into_one_empty_completion_per_member():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    second_key = window_records.DecoderRequestKey(
        1, 1, window_records.STRONG_POOL_ID, 2
    )
    first = _strong_job(first_key, window_key=(1, 0))
    second = decoding_records.DecodeJob(
        operation_id=1,
        window_id=1,
        round_count=9,
        strong_decode_for=(1, 1),
        request_key=second_key,
    )
    requests.admit_strong(first, now=0)
    requests.admit_strong(second, now=0)
    batch = decoding_records.DecodeJob(
        operation_id=-1, window_id=0, round_count=18, strong_decode_for=(1, 0)
    )
    requests.register_batch([(1, 0), (1, 1)], [first, second], batch)
    result = decoding_records.DecodeResult(-1, 0)
    deliveries = requests.deliveries_for(batch, result, now=40)
    request_jobs = [delivery.request_job for delivery in deliveries]
    request_keys = [request_job.request_key for request_job in request_jobs]
    assert request_jobs == [first, second]
    assert request_keys == [first_key, second_key]
    assert deliveries[1].result.window_id == 1
    assert deliveries[1].result.logical_observables is None


def test_a_merged_batch_may_carry_no_accuracy_bearing_field():
    """A user decoder may answer a batch; its members would drop the answer."""
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    first = _strong_job(first_key, window_key=(1, 0))
    second_key = _request_key(2)
    second = _strong_job(second_key, window_key=(1, 1))
    requests.admit_strong(first, now=0)
    requests.admit_strong(second, now=0)
    batch = decoding_records.DecodeJob(
        operation_id=-1, window_id=0, round_count=18, strong_decode_for=(1, 0)
    )
    requests.register_batch([(1, 0), (1, 1)], [first, second], batch)
    result = decoding_records.DecodeResult(-1, 0, logical_observables=(1,))

    with pytest.raises(RuntimeError, match="accuracy-bearing fields"):
        requests.deliveries_for(batch, result, now=40)


def test_a_batch_left_with_one_member_still_splits_into_an_empty_result():
    """The batch's own result names no window, so it is never delivered."""
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    first = _strong_job(first_key, window_key=(1, 0))
    second_key = _request_key(2)
    second = _strong_job(second_key, window_key=(1, 1))
    requests.admit_strong(first, now=0)
    requests.admit_strong(second, now=0)
    batch = decoding_records.DecodeJob(
        operation_id=-1, window_id=0, round_count=18, strong_decode_for=(1, 0)
    )
    requests.register_batch([(1, 0), (1, 1)], [first, second], batch)
    requests.take_live((1, 0), window_records.STRONG_POOL_ID)
    result = decoding_records.DecodeResult(-1, 0)
    (delivery,) = requests.deliveries_for(batch, result, now=40)
    assert delivery.request_job is second
    assert delivery.result.window_id == 1
