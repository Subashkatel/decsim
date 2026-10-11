"""The transfer rows a window's latency is read from, folded by key.

A shot's link ledger is read four ways (measure.py): each window's span
on each hop, when its answer first left for the frame, each decode's
input hop, and when each round left the QPU. Each is a fold over the
rows by key, a minimum, a maximum or a sum, so it is fed one row at a
time: measure_shot feeds a finished shot's rows, and a stream feeds each
row as it lands and lets a window's keys go once the window is measured
(stream_measures.py), as ns-3's FlowMonitor keeps a running record per
flow rather than its packets (flow-monitor.h, FlowStats).
"""

import collections.abc
from typing import Optional

import decsim.records.identity as identity_records

# the hop a boundary's delay is kept under, whatever path it rides
BOUNDARY_HOP = "boundary"


class TransferIndex:
    """The four reductions of the rows fed so far.

    answer_hops are the hops an answer rides home
    (Routes.name_bits_by_answer_hop); the answer sends read only those.
    """

    def __init__(self, answer_hops: collections.abc.Container = ()) -> None:
        self.answer_hops = answer_hops
        # (operation, window) -> {hop: (first request, last delivery)}
        self.spans_by_window: dict = {}
        # (operation, window) -> the tick its answer first left
        self.answer_send: dict = {}
        # window id -> {(path, run ordinal): (delay sum, last delivery)}
        self.inputs_by_window: dict = {}
        # (operation, round) -> the tick the round first left the QPU
        self.qpu_send: dict = {}

    def add(self, row: dict) -> None:
        """One transfer row, folded into each reduction it belongs to."""
        self._add_span(row)
        self._add_answer_send(row)
        self._add_input_hop(row)
        self._add_qpu_send(row)

    def link_delays(self) -> dict:
        """Ticks from first request to last delivery, by (hop, window key)."""
        delays = {}
        for window_key in self.spans_by_window:
            window_delays = self.link_delays_of(window_key)
            delays.update(window_delays)
        return delays

    def link_delays_of(self, window_key: tuple) -> dict:
        """One window's span on each hop, keyed as link_delays keys it."""
        operation_id, window_id = window_key
        spans = self.spans_by_window.get(window_key, {})
        delays = {}
        for hop, (first_request, last_delivery) in spans.items():
            delays[(hop, operation_id, window_id)] = (
                last_delivery - first_request
            )
        return delays

    def input_hops(self) -> dict:
        """(path, window id, run ordinal) -> (delay ticks, last delivery)."""
        hops = {}
        for window_id in self.inputs_by_window:
            window_hops = self.input_hops_of(window_id)
            hops.update(window_hops)
        return hops

    def input_hops_of(self, window_id: int) -> dict:
        """One window's input hops, keyed as input_hops keys them."""
        inputs = self.inputs_by_window.get(window_id, {})
        hops = {}
        for (path, run_sequence), hop in inputs.items():
            hops[(path, window_id, run_sequence)] = hop
        return hops

    def forget_window(self, window_key: tuple) -> None:
        """Let a measured window's keys go."""
        _operation_id, window_id = window_key
        self.spans_by_window.pop(window_key, None)
        self.answer_send.pop(window_key, None)
        self.inputs_by_window.pop(window_id, None)

    def forget_rounds(self, operation_id: object, round_indices: range) -> None:
        """Let the sends of rounds no unmeasured window reads go."""
        for round_index in round_indices:
            self.qpu_send.pop((operation_id, round_index), None)

    def _add_span(self, row: dict) -> None:
        """A hop's span for a window: the earliest request, latest delivery.

        A hop's transfers for one window can overlap (the selection and
        its rounds), so the span is what the window waited and a sum
        would count the overlap twice.
        """
        hop = hop_of(row)
        window_key = waiting_window(row)
        request = hop_start_ticks(row)
        delivery = row["delivery_ticks"]
        spans = self.spans_by_window.setdefault(window_key, {})
        first_request, last_delivery = spans.get(hop, (request, delivery))
        earliest = min(first_request, request)
        latest = max(last_delivery, delivery)
        spans[hop] = (earliest, latest)

    def _add_answer_send(self, row: dict) -> None:
        if row["path"] not in self.answer_hops or is_boundary(row):
            return
        attribution = row["attribution"]
        operation_id = identity_records.stable_identity_from_json(
            attribution["operation_id"]
        )
        key = (operation_id, attribution["window_id"])
        start = hop_start_ticks(row)
        earlier = self.answer_send.get(key, start)
        self.answer_send[key] = min(earlier, start)

    def _add_input_hop(self, row: dict) -> None:
        """A decode's input hop: its rows' delays summed, the last landing.

        A decode starts once every input transfer landed.
        """
        run_sequence = request_run_sequence(row)
        if run_sequence is None:
            return
        window_id = row["attribution"]["window_id"]
        inputs = self.inputs_by_window.setdefault(window_id, {})
        key = (row["path"], run_sequence)
        delay, landing = inputs.get(key, (0, 0))
        delay += row["total_delay_ticks"]
        landing = max(landing, row["delivery_ticks"])
        inputs[key] = (delay, landing)

    def _add_qpu_send(self, row: dict) -> None:
        if row["path"] != "qpu_to_controller":
            return
        (rounds,) = row["attribution"]["rounds_by_operation"]
        operation_id = identity_records.stable_identity_from_json(
            rounds["operation_id"]
        )
        round_key = (operation_id, rounds["round_lo"])
        start = hop_start_ticks(row)
        earlier = self.qpu_send.get(round_key, start)
        self.qpu_send[round_key] = min(earlier, start)


def hop_of(row: dict) -> str:
    """The hop a transfer's delay is kept under: BOUNDARY_HOP, or its path."""
    if is_boundary(row):
        return BOUNDARY_HOP
    return row["path"]


def waiting_window(row: dict) -> tuple:
    """(operation_id, window index) of the window the transfer holds up.

    A boundary holds up its destination; every other transfer the window
    its attribution names.
    """
    attribution = row["attribution"]
    if is_boundary(row):
        destination = attribution["relation"]["destination_window_key"]
        return identity_records.stable_identity_from_json(destination)
    recorded_operation = attribution["operation_id"]
    operation_id = identity_records.stable_identity_from_json(
        recorded_operation
    )
    return operation_id, attribution["window_id"]


def is_boundary(row: dict) -> bool:
    """Whether the transfer carries a boundary, which names its source."""
    relation = row["attribution"].get("relation")
    if not relation:
        return False
    return "source_window_key" in relation


def hop_start_ticks(row: dict) -> int:
    """The tick a transfer was asked for, which is where its hop starts.

    The request is the delivery less total_delay_ticks. The setup is the
    hop's own cost, as gem5 adds a DMA's fixed delay to the completion
    (src/dev/dma_device.cc:116-118).
    """
    return row["delivery_ticks"] - row["total_delay_ticks"]


def request_run_sequence(row: dict) -> Optional[int]:
    """The run ordinal of the request a transfer serves, None when it has none.

    A boundary hand-off is no decode's input, though it names the
    request that committed it, and a round's transfer names no request.
    """
    relation = row["attribution"].get("relation")
    if not relation or is_boundary(row):
        return None
    request_key = relation.get("request_key")
    if request_key is None:
        return None
    return request_key["run_sequence"]
