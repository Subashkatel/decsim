"""The switching study's terminal records: per request, per service, per gap.

Listeners on the decode outcomes' request_ended and service_ended
sources; they never read the decoder. The request and service ledger is
built only when the observation section asks for the switching windows,
and the confidence ledger only when a confidence signal decides the
escalation, so the decoder runs with no record kept.
"""

import dataclasses
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.identity as identity_records
import decsim.records.windows as window_records

# the outcomes of the decode whose confidence the verdict read: kept
# and answered, or escalated to the strong tier (Toshio et al.
# 2510.25222 Sec. III A, step 3)
VERDICT_OUTCOMES = (
    decoding_records.RequestProcessingOutcome.PRIMARY_FORWARDED_FOR_DELIVERY,
    decoding_records.RequestProcessingOutcome.WEAK_AWAITED_STRONG,
)
_ESCALATED = decoding_records.RequestProcessingOutcome.WEAK_AWAITED_STRONG
_STRONG_ANSWER = (
    decoding_records.RequestProcessingOutcome.STRONG_FORWARDED_FOR_DELIVERY
)


@dataclasses.dataclass(frozen=True)
class TerminalRequestRecord:
    """One decode request at its end: identity, input, ticks, outcome."""

    request_key: window_records.DecoderRequestKey
    input_round_lo: int
    input_round_hi: int
    input_round_count: int
    syndrome_bit_count: Optional[int]
    syndrome_weight: Optional[int]
    created_ticks: int
    admitted_ticks: Optional[int]
    ready_ticks: int
    dispatch_ticks: Optional[int]
    decode_output_ticks: Optional[int]
    service_key: Optional[decoding_records.DecoderServiceKey]
    soft_output: Optional[decoding_records.SoftOutput]
    terminal_processing_outcome: decoding_records.RequestProcessingOutcome


@dataclasses.dataclass(frozen=True)
class TerminalServiceRecord:
    """One decode service at its end: the requests it served and its ticks."""

    service_key: decoding_records.DecoderServiceKey
    pool: str
    original_request_keys: tuple[window_records.DecoderRequestKey, ...]
    completed_request_keys: tuple[window_records.DecoderRequestKey, ...]
    cancelled_request_keys: tuple[window_records.DecoderRequestKey, ...]
    input_round_count: int
    dispatch_ticks: int
    terminal_ticks: int
    service_ticks: int


class DecodeRecordLedger:
    """The request and service records, in the order they ended."""

    def __init__(self) -> None:
        self.requests: list[TerminalRequestRecord] = []
        self.services: list[TerminalServiceRecord] = []

    def request_ended(
        self,
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
        outcome: decoding_records.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """One request reached its terminal outcome."""
        window = job.window
        local_fragments = ()
        if job.decoder_input is not None:
            fragments = job.decoder_input.fragments()
            local_fragments = tuple(fragments)
        bit_count, weight = _syndrome_bit_count_and_weight(local_fragments)
        soft_output = None
        if result is not None:
            soft_output = result.soft_output
        record = TerminalRequestRecord(
            job.request_key,
            window.start_round,
            window.buffer_hi,
            job.round_count,
            bit_count,
            weight,
            job.request_created_ticks,
            job.request_admitted_ticks,
            job.ready_time,
            job.service_dispatch_ticks,
            decode_output_ticks,
            job.service_key,
            soft_output,
            outcome,
        )
        self.requests.append(record)

    def service_ended(self, job: decoding_records.DecodeJob, now: int) -> None:
        """One physical decode ended, with every request it served."""
        if job.service_key is None:
            return
        original = job.service_original_request_keys
        cancelled = []
        completed = []
        for key in original:
            if key in job.service_cancelled_request_keys:
                cancelled.append(key)
            else:
                completed.append(key)
        dispatch = job.service_dispatch_ticks
        service_ticks = now - dispatch
        record = TerminalServiceRecord(
            job.service_key,
            job.pool,
            original,
            tuple(completed),
            tuple(cancelled),
            job.round_count,
            dispatch,
            now,
            service_ticks,
        )
        self.services.append(record)


class ConfidenceLedger:
    """Each window's confidence gap, verdict and strong answer, as they end."""

    def __init__(self) -> None:
        self.verdict_results: dict = {}
        self.escalated_keys: set = set()
        self.strong_observables: dict = {}

    def request_ended(
        self,
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
        outcome: decoding_records.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """One request ended: the verdict's decode, or the strong answer."""
        del decode_output_ticks
        if result is None:
            return
        key = (job.operation_id, job.window_id)
        if outcome is _STRONG_ANSWER:
            self.strong_observables[key] = result.logical_observables
            return
        if outcome not in VERDICT_OUTCOMES:
            return
        self.verdict_results[key] = result
        if outcome is _ESCALATED:
            self.escalated_keys.add(key)

    def windows(self) -> tuple:
        """One WindowConfidence per window the verdict read, in order."""
        confidences = []
        keys = sorted(self.verdict_results, key=_window_order)
        for key in keys:
            confidence = self._confidence_of(key)
            confidences.append(confidence)
        return tuple(confidences)

    def _confidence_of(self, key: tuple) -> decoding_records.WindowConfidence:
        result = self.verdict_results[key]
        is_escalated = key in self.escalated_keys
        is_strong_revised = None
        if is_escalated and key in self.strong_observables:
            strong = self.strong_observables[key]
            is_strong_revised = strong != result.logical_observables
        gap = None
        if result.soft_output is not None:
            gap = result.soft_output.gap
        return decoding_records.WindowConfidence(
            key, gap, is_escalated, is_strong_revised
        )


def _window_order(key: tuple) -> tuple:
    """A window key's place: its operation's identity, then its index."""
    operation_id, window_id = key
    operation_order = identity_records.stable_identity_order_key(operation_id)
    return (operation_order, window_id)


def _syndrome_bit_count_and_weight(fragments: tuple) -> tuple:
    """(bit count, set bits) of the landed input; None when bits are unknown."""
    if not fragments:
        return None, None
    bit_count = 0
    weight = 0
    for fragment in fragments:
        if fragment.bits is None:
            return None, None
        bit_count += len(fragment.bits)
        weight += sum(fragment.bits)
    return bit_count, weight
