"""The switching study's terminal records, per request and per service.

A listener on the decode outcomes' request_ended and service_ended
sources; it never reads the decoder. Built and connected only when the
observation section asks for the switching windows, so the decoder runs
with no record kept.
"""

import dataclasses
from typing import Optional

import decsim.records.decoding as decoding_records
import decsim.records.windows as window_records


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
