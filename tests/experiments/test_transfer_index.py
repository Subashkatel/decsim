"""The transfer rows folded by key (transfer_index.py).

Rows are fed one at a time, as a stream lands them; each reduction is
its fold (a span by window and hop, a first send, a summed input hop),
and a window's keys go when it is let go while another's stay.
"""

import decsim.experiments.transfer_index as transfer_index
import decsim.records.identity as identity_records

ANSWER_HOP = "weak_decoder_to_controller"
INPUT_HOP = "weak_buffer_to_weak_decoder"


def _row(path, window_index, times, run_sequence=None, stream_id=1) -> dict:
    """One transfer of a stream's window; times: request, delivery."""
    request, delivery = times
    owner = identity_records.stable_identity_json(stream_id)
    attribution = {"owner_id": owner, "window_id": window_index}
    if run_sequence is not None:
        request_key = {"run_sequence": run_sequence}
        attribution["relation"] = {"request_key": request_key}
    return {
        "path": path,
        "attribution": attribution,
        "delivery_ticks": delivery,
        "total_delay_ticks": delivery - request,
    }


def _index(rows) -> transfer_index.TransferIndex:
    index = transfer_index.TransferIndex(answer_hops={ANSWER_HOP})
    for row in rows:
        index.add(row)
    return index


def test_a_hops_overlapping_transfers_span_once_and_an_input_sums():
    """Two input rows of one decode: span 100 to 160, delay 40 + 30."""
    first = _row(INPUT_HOP, 2, (100, 140), run_sequence=7)
    second = _row(INPUT_HOP, 2, (130, 160), run_sequence=7)

    index = _index([first, second])

    assert index.link_delays() == {(INPUT_HOP, 1, 2): 60}
    assert index.input_hops() == {(INPUT_HOP, 1, 2, 7): (70, 160)}


def test_an_answer_counts_from_its_first_send_home():
    early = _row(ANSWER_HOP, 3, (500, 520))
    late = _row(ANSWER_HOP, 3, (510, 530))

    index = _index([late, early])

    assert index.answer_send == {(1, 3): 500}


def test_letting_a_window_go_keeps_the_others():
    rows = [
        _row(INPUT_HOP, 2, (100, 140), run_sequence=7),
        _row(INPUT_HOP, 3, (200, 240), run_sequence=8),
        _row(ANSWER_HOP, 2, (300, 310)),
    ]
    index = _index(rows)

    index.forget_window((1, 2))

    assert index.link_delays() == {(INPUT_HOP, 1, 3): 40}
    assert index.input_hops() == {(INPUT_HOP, 1, 3, 8): (40, 240)}
    assert index.answer_send == {}


def test_letting_one_streams_window_go_keeps_anothers_of_the_same_index():
    """Every stream counts its windows from 0; window 0 of each is its own."""
    rows = [
        _row(INPUT_HOP, 0, (100, 105), run_sequence=1, stream_id=1),
        _row(INPUT_HOP, 0, (100, 105), run_sequence=2, stream_id=2),
    ]
    index = _index(rows)

    index.forget_window((1, 0))

    assert index.input_hops() == {(INPUT_HOP, 2, 0, 2): (5, 105)}
