"""The record ledger: a listener that keeps a record only when enabled."""

import decsim.message as message
import decsim.observe.decode_records as decode_records


def _job():
    window = message.Window(
        op_id=1, k=0, commit_lo=1, commit_hi=3, buffer_hi=5, n_rounds=5
    )
    request_key = message.DecoderRequestKey(1, 0, message.DecoderTier.WEAK, 0)
    service_key = message.DecoderServiceKey(0)
    return message.DecodeJob(
        op_id=1,
        window_id=0,
        n_rounds=5,
        request_key=request_key,
        window=window,
        request_created_ticks=10,
        request_admitted_ticks=12,
        ready_time=12,
        service_dispatch_ticks=15,
        service_key=service_key,
        service_original_request_keys=(request_key,),
    )


def test_a_request_and_its_service_are_recorded_at_their_end():
    ledger = decode_records.DecodeRecordLedger(is_enabled=True)
    job = _job()
    result = message.DecodeResult(1, 0)
    outcome = message.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
    ledger.request_ended(job, result, outcome, 40)
    ledger.service_ended(job, 40)
    (request,) = ledger.requests
    (service,) = ledger.services
    assert request.input_round_lo == 1
    assert request.input_round_hi == 5
    assert request.decode_output_ticks == 40
    assert request.terminal_processing_outcome is outcome
    assert service.service_ticks == 25
    assert service.completed_request_keys == (job.request_key,)


def test_a_disabled_ledger_keeps_nothing():
    ledger = decode_records.DecodeRecordLedger(is_enabled=False)
    job = _job()
    outcome = message.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
    ledger.request_ended(job, None, outcome, 40)
    ledger.service_ended(job, 40)
    assert ledger.requests == []
    assert ledger.services == []
