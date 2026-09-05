"""One unconsumed result per destination; held until selected; batches split.

Toshio et al. 2510.25222: one strong re-decode per escalated window,
selected by the weak side and consumed by the window that asked.
"""

import pytest

import decsim.decoders.strong_requests as strong_requests_module
import decsim.message as message
from decsim.decoders.strong_requests import HeldStrongCompletion


def _request_key(sequence):
    return message.DecoderRequestKey(1, 0, message.DecoderTier.STRONG, sequence)


def _strong_job(request_key, window_key=(1, 0)):
    return message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=9,
        strong_decode_for=window_key,
        request_key=request_key,
    )


def _held(job, key, now=50):
    result = message.DecodeResult(1, 0)
    completion = message.StrongDecodeCompletion(key, result)
    return HeldStrongCompletion(job, completion, now)


def test_a_result_before_its_selection_is_held_then_consumed():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.begin_selection((1, 0), key)
    requests.finish_service(job)
    held = _held(job, key)
    assert requests.complete(held) is False
    selected = requests.select((1, 0), key)
    assert selected.completion.request_key == key


def test_a_selection_before_the_result_consumes_it_at_once():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.begin_selection((1, 0), key)
    assert requests.select((1, 0), key) is None
    requests.finish_service(job)
    held = _held(job, key)
    assert requests.complete(held) is True
    assert requests.counts.needed == 1


def test_a_stale_result_is_refused_once_a_newer_request_owns_the_window():
    requests = strong_requests_module.StrongRequests()
    old_key = _request_key(7)
    old_job = _strong_job(old_key)
    requests.admit_strong(old_job, now=0)
    requests.take_live((1, 0))
    new_key = _request_key(9)
    new_job = _strong_job(new_key)
    requests.admit_strong(new_job, now=1)
    stale = _held(old_job, old_key)
    with pytest.raises(RuntimeError, match="newer strong request"):
        requests.complete(stale)


def test_a_result_nobody_waits_for_is_refused():
    requests = strong_requests_module.StrongRequests()
    key = _request_key(7)
    job = _strong_job(key)
    requests.admit_strong(job, now=0)
    requests.finish_service(job)
    orphan = _held(job, key)
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


def test_a_second_weak_decode_of_an_unresolved_window_is_refused():
    requests = strong_requests_module.StrongRequests()
    first = message.DecodeJob(op_id=1, window_id=0, n_rounds=3)
    second = message.DecodeJob(op_id=1, window_id=0, n_rounds=3)
    requests.admit(first, now=0)
    with pytest.raises(RuntimeError, match="decodes once at a time"):
        requests.admit(second, now=1)


def test_a_merged_batch_splits_into_one_empty_completion_per_member():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    second_key = message.DecoderRequestKey(1, 1, message.DecoderTier.STRONG, 2)
    first = _strong_job(first_key, window_key=(1, 0))
    second = message.DecodeJob(
        op_id=1,
        window_id=1,
        n_rounds=9,
        strong_decode_for=(1, 1),
        request_key=second_key,
    )
    requests.admit_strong(first, now=0)
    requests.admit_strong(second, now=0)
    batch = message.DecodeJob(
        op_id=-1, window_id=0, n_rounds=18, strong_decode_for=(1, 0)
    )
    requests.register_batch([(1, 0), (1, 1)], [first, second], batch)
    result = message.DecodeResult(-1, 0)
    deliveries = requests.deliveries_for(batch, result, now=40)
    assert [held.request_job for held in deliveries] == [first, second]
    assert [held.completion.request_key for held in deliveries] == [
        first_key,
        second_key,
    ]
    assert deliveries[1].completion.result.window_id == 1
    assert deliveries[1].completion.result.logical_observables is None


def test_a_merged_batch_may_carry_no_accuracy_bearing_field():
    requests = strong_requests_module.StrongRequests()
    first_key = _request_key(1)
    first = _strong_job(first_key, window_key=(1, 0))
    second_key = _request_key(2)
    second = _strong_job(second_key, window_key=(1, 1))
    requests.admit_strong(first, now=0)
    requests.admit_strong(second, now=0)
    batch = message.DecodeJob(
        op_id=-1, window_id=0, n_rounds=18, strong_decode_for=(1, 0)
    )
    requests.register_batch([(1, 0), (1, 1)], [first, second], batch)
    result = message.DecodeResult(-1, 0, logical_observables=(1,))
    with pytest.raises(RuntimeError, match="accuracy-bearing"):
        requests.deliveries_for(batch, result, now=40)
