"""The switching study's terminal records: per request and per gap.

Listeners on the decode outcomes' request_ended source; they never read
the decoder. The request ledger is built only when the observation
section asks for the switching windows, and the confidence ledger only
when a confidence signal decides the escalation, so the decoder runs
with no record kept.
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
    """One decode request at its end: identity, input, ticks, confidence."""

    request_key: window_records.DecoderRequestKey
    input_round_count: int
    syndrome_weight: Optional[int]
    ready_ticks: int
    dispatch_ticks: Optional[int]
    decode_output_ticks: Optional[int]
    service_key: Optional[decoding_records.DecoderServiceKey]
    soft_output: Optional[decoding_records.SoftOutput]


class DecodeRecordLedger:
    """The request records, in the order they ended."""

    def __init__(self) -> None:
        self.requests: list[TerminalRequestRecord] = []

    def request_ended(
        self,
        job: decoding_records.DecodeJob,
        result: Optional[decoding_records.DecodeResult],
        outcome: decoding_records.RequestProcessingOutcome,
        decode_output_ticks: Optional[int],
    ) -> None:
        """One request reached its terminal outcome."""
        del outcome
        weight = _decoded_syndrome_weight(job)
        soft_output = None
        if result is not None:
            soft_output = result.soft_output
        record = TerminalRequestRecord(
            job.request_key,
            job.round_count,
            weight,
            job.ready_time,
            job.service_dispatch_ticks,
            decode_output_ticks,
            job.service_key,
            soft_output,
        )
        self.requests.append(record)


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
    operation_order = identity_records.stable_identity_bytes(operation_id)
    return (operation_order, window_id)


def _decoded_syndrome_weight(job: decoding_records.DecodeJob) -> Optional[int]:
    """The set bits the job's decode read; None when it never started.

    The decode reads its unit's memory into payloads as it starts
    (decoders/decode_service.py), and the unit frees that memory at the
    decode's end, before a confidence walk lets the verdict end the
    request, so payloads is what still holds the input then.
    """
    if not job.service_started:
        return None
    fragments = tuple(job.payloads)
    return _syndrome_weight(fragments)


def _syndrome_weight(fragments: tuple) -> Optional[int]:
    """The set bits of the input; None when its bits are unknown."""
    if not fragments:
        return None
    weight = 0
    for fragment in fragments:
        if fragment.bits is None:
            return None
        weight += sum(fragment.bits)
    return weight
