"""The window transfers' laws on the reference link card.

A send carries the window's round range and the request it serves; a
result reaches the frame as one bit per logical observable.
"""

import decsim.engine as engine_module
import decsim.links.window_transfers as window_transfers
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


class _RecordingLink:
    def __init__(self) -> None:
        self.sent = []

    def expected_delay_ticks(self, _path, _payload_bits, _now_ticks) -> int:
        return 7

    def send(self, path, payload_bits, now_ticks, attribution, on_delivered):
        self.sent.append((path, payload_bits, now_ticks, attribution))
        on_delivered(None)


def test_a_window_send_carries_its_round_range_and_request_key():
    engine = engine_module.Engine()
    link = _RecordingLink()
    transfers = window_transfers.WindowTransfers(engine, link)
    operation = program_records.Operation(1, "memory", (0,), patches=(3, 2))
    window = window_records.Window(
        operation_id=1,
        window_index=4,
        commit_lo=9,
        commit_hi=11,
        buffer_hi=13,
        round_count=7,
        buffer_lo=7,
    )
    request_key = window_records.DecoderRequestKey(
        1, 4, window_records.DecoderTier.WEAK, 5
    )
    delivered = []
    transfers.send_for_window(
        transfer_records.LinkPath.WEAK_DECODER_TO_FRAME,
        window,
        operation,
        request_key,
        2,
        lambda: delivered.append(engine.now),
    )
    (path, payload_bits, now_ticks, attribution) = link.sent[0]
    assert path is transfer_records.LinkPath.WEAK_DECODER_TO_FRAME
    assert payload_bits == 2
    assert now_ticks == 0
    assert attribution.window_id == 4
    assert (attribution.first_round, attribution.last_round) == (7, 13)
    assert attribution.patch_ids == (2, 3)
    assert attribution.relation.request_key is request_key
    assert delivered == [0]


def test_a_job_send_returns_the_delay_the_link_expects():
    engine = engine_module.Engine()
    link = _RecordingLink()
    transfers = window_transfers.WindowTransfers(engine, link)
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
    job = decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=5,
        window=window,
        request_key=request_key,
    )
    expected = transfers.send_for_job(
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        job,
        payload_bits=40,
        on_delivered=lambda: None,
    )
    assert expected == 7
    (_path, payload_bits, _now, attribution) = link.sent[0]
    assert payload_bits == 40
    assert (attribution.first_round, attribution.last_round) == (1, 5)
