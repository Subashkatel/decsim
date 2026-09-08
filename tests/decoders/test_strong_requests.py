"""One unconsumed result per destination; the unit holds it; batches split.

Toshio et al. 2510.25222: one strong re-decode per escalated window,
selected by the weak side and consumed by the window that asked.
"""

import pytest

import decsim.decoders.strong_requests as strong_requests_module
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records
from decsim.decoders.strong_requests import StrongCompletion


def _request_key(sequence):
    return window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.STRONG, sequence
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
    return StrongCompletion(job, result, now, None)


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
    requests.take_live((1, 0))
    new_key = _request_key(9)
    new_job = _strong_job(new_key)
    requests.admit_strong(new_job, now=1)
    stale = _completion(old_job)
    with pytest.raises(RuntimeError, match="newer strong request"):
        requests.complete(stale)


def test_a_result_nobody_waits_for_is_refused():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.finish_service(job)
    orphan = _completion(job)
    with pytest.raises(RuntimeError, match="no destination waiting"):
        requests.complete(orphan)


def test_a_destination_keeps_at_most_one_unconsumed_strong_result():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(7)
    first = _strong_job(first_key)
    requests.admit_strong(first, now=0)
    second_key = _request_key(8)
    second = _strong_job(second_key)
    with pytest.raises(RuntimeError, match="duplicate strong decode"):
        requests.admit_strong(second, now=1)


def test_a_windows_attempt_holds_its_forced_class_requests_and_no_repeat():
    """One attempt, one or two requests; a request is admitted once.

    The complementary gap decodes a window once per logical class, so
    the window's open attempt names both request keys and resolves as
    one when the join hands the window's answer on.
    """
    requests = strong_requests_module.StrongRequests()
    first_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    second_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 1
    )
    first = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, request_key=first_key
    )
    second = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=3, request_key=second_key
    )
    requests.admit(first, now=0)
    requests.admit(second, now=0)
    record = requests.by_window[(1, 0)]
    assert record.open_weak_requests == {first_key, second_key}
    with pytest.raises(RuntimeError, match="is already open"):
        requests.admit(second, now=1)
    requests.resolve_weak((1, 0))
    assert (1, 0) not in requests.by_window


def test_a_merged_batch_splits_into_one_empty_completion_per_member():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    second_key = window_records.DecoderRequestKey(
        1, 1, window_records.DecoderTier.STRONG, 2
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
    request_jobs = []
    for delivery in deliveries:
        request_jobs.append(delivery.request_job)
    assert request_jobs == [first, second]
    request_keys = []
    for delivery in deliveries:
        request_keys.append(delivery.request_job.request_key)
    assert request_keys == [first_key, second_key]
    assert deliveries[1].result.window_id == 1
    assert deliveries[1].result.logical_observables is None


def test_a_merged_batch_may_carry_no_accuracy_bearing_field():
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
    with pytest.raises(RuntimeError, match="accuracy-bearing"):
        requests.deliveries_for(batch, result, now=40)
