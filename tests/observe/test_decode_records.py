"""The record ledgers: listeners on the two terminal sources."""

import decsim.confidence.complementary as complementary
import decsim.observe.decode_records as decode_records
import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


def _job():
    window = window_records.Window(
        operation_id=1,
        window_index=0,
        commit_lo=1,
        commit_hi=3,
        buffer_hi=5,
        round_count=5,
    )
    request_key = window_records.DecoderRequestKey(
        1, 0, window_records.DecoderTier.WEAK, 0
    )
    service_key = decoding_records.DecoderServiceKey(0)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=5,
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
    ledger = decode_records.DecodeRecordLedger()
    job = _job()
    result = decoding_records.DecodeResult(1, 0)
    outcome = (
        decoding_records.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
    )
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


WEAK_KEPT = (
    decoding_records.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY
)
WEAK_ESCALATED = decoding_records.RequestProcessingOutcome.WEAK_AWAITED_STRONG
STRONG_ANSWER = (
    decoding_records.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY
)
CANCELLED = (
    decoding_records.RequestProcessingOutcome.STRONG_CANCELLED_BEFORE_DISPATCH
)


def _gap(nats):
    source = complementary.COMPLEMENTARY_GAP_SOURCE
    return decoding_records.SoftOutput(gap=nats, source=source)


def test_a_kept_window_has_its_gap_and_no_strong_answer():
    ledger = decode_records.ConfidenceLedger()
    job = _job()
    gap = _gap(5.0)
    kept = decoding_records.DecodeResult(1, 0, soft_output=gap)

    ledger.request_ended(job, kept, WEAK_KEPT, 40)

    (window,) = ledger.windows()
    assert window == decoding_records.WindowConfidence((1, 0), 5.0, False, None)


def test_an_escalated_window_the_strong_tier_answered_otherwise_is_revised():
    ledger = decode_records.ConfidenceLedger()
    job = _job()
    gap = _gap(0.5)
    weak = decoding_records.DecodeResult(
        1, 0, logical_observables=(0,), soft_output=gap
    )
    strong = decoding_records.DecodeResult(1, 0, logical_observables=(1,))

    ledger.request_ended(job, weak, WEAK_ESCALATED, 40)
    ledger.request_ended(job, strong, STRONG_ANSWER, 90)

    (window,) = ledger.windows()
    assert window == decoding_records.WindowConfidence((1, 0), 0.5, True, True)


def test_a_window_the_signal_gave_no_gap_is_listed_with_none():
    ledger = decode_records.ConfidenceLedger()
    job = _job()
    gapless = decoding_records.DecodeResult(1, 0)

    ledger.request_ended(job, gapless, WEAK_ESCALATED, 40)

    (window,) = ledger.windows()
    assert window.gap_nats is None
    assert window.is_escalated


def test_a_request_that_ended_with_no_result_is_no_window():
    ledger = decode_records.ConfidenceLedger()
    job = _job()

    ledger.request_ended(job, None, CANCELLED, None)

    assert ledger.windows() == ()
