"""The decoder side's outgoing sends: the frame, the strong tier, a peer.

Whoever executes a send is an end of that hop: OMNeT++ refuses a send of
a message another module owns (omnetpp-6.1.0
src/sim/csimplemodule.cc:333-334), and gem5 bills a transfer to the
ports it crossed, never to a proxy that arranged it
(coherent_xbar.cc:354-357, xbar.hh:400-411, packet.hh:424-431). Two hops
leave a decoder: the correction to the Pauli frame, and the escalation
to the strong decoder, the selection of the strong request and then the
rounds of its window read out of the weak syndrome buffer (Toshio et al.
2510.25222 lines 1247-1250). Yang et al. 2605.04892 Table I counts the
frame update inside the decoder's own subtotal. A strong answer takes
the route its run generated to the frame (build/routes.py). A window's
boundary is the window side's record and leaves from there
(windows/window_boundaries.py).
"""

import functools
from collections.abc import Callable
from typing import Optional

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_model_contracts
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.routes as route_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


class DecoderOutput:
    """Sends one decoder's answers where they go.

    A correction commits into the frame as a priced write. Each pool's
    answer takes its route in routes, and clock prices a route's commit
    step on the weak chip. An answer names the request it answers on
    every hop that crosses the wall: CUDA-Q's reply echoes the request's
    id, "enabling out-of-order or pipelined verification of responses"
    (cudaq_realtime_message_protocol.md, Request ID Semantics), and gem5
    sizes a response that carries data as that data plus the control
    size (gem5 src/mem/ruby/network/Network.cc MessageSizeType_to_int,
    Response_Data). An answer on the board with its frame carries no
    name; a run that frames that hop sets its header_bits_per_transfer.
    """

    transfers = ports.Port(ports.WindowTransfers)
    # a run with no frame commits its corrections nowhere
    frame = ports.Port(ports.Frame, optional=True)
    # the store an escalated region's rounds are read out of, and the
    # fabric asked what their move will pay once that read is done
    weak_store = ports.Port(ports.SyndromeBuffer)
    link = ports.Port(ports.Link)

    def __init__(
        self,
        engine: engine_module.Engine,
        routes: route_records.Routes,
        clock: Optional[config.Clock] = None,
    ) -> None:
        self.engine = engine
        self.routes = routes
        self.clock = clock

    def publish(
        self,
        window: window_records.Window,
        operation: program_records.Operation,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Send the result home on its pool's route; commit it at delivery.

        A route with a hop down sends the answer to the weak chip first,
        whose commit step sends it home; the frame's priced write, when
        the run has a frame, gates on_committed.
        """
        route = self.routes.answer(request_key.pool)
        flip_bits = result_payload_bits(result, operation)
        commit = functools.partial(
            self._commit, window.key, result, request_key, on_committed
        )
        send_home = functools.partial(
            self._send_answer,
            route.home,
            window,
            operation,
            request_key,
            flip_bits,
            commit,
        )
        if route.down is None:
            send_home()
            return
        at_the_chip = functools.partial(
            self._commit_on_the_chip, route.commit_cycles, send_home
        )
        self._send_answer(
            route.down, window, operation, request_key, flip_bits, at_the_chip
        )

    def send_selection(
        self,
        weak_job: decoding_records.DecodeJob,
        strong_request_key: window_records.DecoderRequestKey,
        on_delivered: Callable[[], None],
    ) -> int:
        """Send one window's escalation to the strong decoder.

        The send is in the weak job's name for the strong request it
        selects; returns the delay the link expects. A selection is the
        request's name (records/windows.py REQUEST_KEY_WIRE_BITS), and
        the weak crossing commit when the strong host joins it, so a
        bounded hop serializes those bits and its header.
        """
        payload_bits = self._selection_bits(weak_job, strong_request_key)
        return self.transfers.send_for_job(
            transfer_records.WEAK_DECODER_TO_STRONG_DECODER,
            weak_job,
            payload_bits=payload_bits,
            request_key=strong_request_key,
            on_delivered=on_delivered,
        )

    def send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        """Read a strong window's rounds, then send them to the strong store.

        The rounds leave the weak syndrome buffer here, so that store
        prices the read once (book_read) and the send starts at its end.
        The send carries the request's name and the rounds
        (EscalatedRegion.message_bits); returns the read's time plus the
        link's estimate (links/channel.py expected_delay_ticks).
        """
        round_keys = region.round_keys
        read_tick = self.weak_store.book_read(round_keys)
        if read_tick == self.engine.now:
            return self._send_region(region, on_delivered)
        send = functools.partial(self._send_region, region, on_delivered)
        read_delay = read_tick - self.engine.now
        self.engine.schedule(read_delay, send, label="syndrome buffer read")
        message_bits = region.message_bits()
        link_delay = self.link.expected_delay_ticks(
            transfer_records.WEAK_DECODER_TO_STRONG_DECODER,
            message_bits,
            read_tick,
        )
        return read_delay + link_delay

    def _send_region(
        self,
        region: round_records.EscalatedRegion,
        on_delivered: Callable[[], None],
    ) -> int:
        return self.transfers.send_region(
            transfer_records.WEAK_DECODER_TO_STRONG_DECODER,
            region,
            on_delivered,
        )

    def _selection_bits(
        self,
        weak_job: decoding_records.DecodeJob,
        strong_request_key: window_records.DecoderRequestKey,
    ) -> int:
        """The request's name, then the crossing commit the host joins."""
        name_bits = window_records.REQUEST_KEY_WIRE_BITS
        strong_route = self.routes.answer(strong_request_key.pool)
        if strong_route.down is not None:
            return name_bits
        crossing_bits = crossing_commit_bits(weak_job)
        return name_bits + crossing_bits

    def _send_answer(
        self,
        path: transfer_records.LinkPath,
        window: window_records.Window,
        operation: program_records.Operation,
        request_key: window_records.DecoderRequestKey,
        flip_bits: int,
        on_delivered: Callable[[], None],
    ) -> None:
        name_bits = self.routes.answer_name_bits(path)
        payload_bits = flip_bits + name_bits
        self.transfers.send_for_window(
            path, window, operation, request_key, payload_bits, on_delivered
        )

    def _commit_on_the_chip(
        self, commit_cycles: int, send_home: Callable[[], None]
    ) -> None:
        """The weak chip joins the landed answer, then sends it home."""
        delay_ticks = self.clock.ticks_to_edge(commit_cycles, self.engine.now)
        self.engine.schedule(delay_ticks, send_home, label="weak chip commit")

    def _commit(
        self,
        window_key: tuple,
        result: decoding_records.DecodeResult,
        request_key: window_records.DecoderRequestKey,
        on_committed: Callable[[], None],
    ) -> None:
        """Charge and install one final correction, then call back."""
        if self.frame is None:
            on_committed()
            return
        self.frame.commit_correction(
            window_key=window_key,
            logical_observables=result.logical_observables,
            request_key=request_key,
            on_committed=on_committed,
        )


def result_payload_bits(
    result: decoding_records.DecodeResult, operation: program_records.Operation
) -> int:
    """A result reaches the frame as one bit per logical observable.

    A timing-only result stands for one observable per patch.
    """
    if result.logical_observables is not None:
        return len(result.logical_observables)
    patch_count = len(operation.patches)
    return max(1, patch_count)


def crossing_commit_bits(weak_job: decoding_records.DecodeJob) -> int:
    """The bits of a weak job's crossing commit: one per logical observable.

    The selection leaves before the weak result is committed, so the
    count is the window's own: the observables of its error model, the
    rows its result's flips come from (decoder.py
    result_from_selected_faults). A job whose model holds no faults, or
    that has no model, is timing-only. It holds no operation, so, as
    result_payload_bits does, it stands for one observable per patch,
    the patches of its own rounds.
    """
    faults = _window_faults(weak_job.detector_error_model)
    if faults is None:
        payloads = weak_job.payloads or ()
        patch_ids = round_records.fragment_patch_ids(payloads)
        return max(1, len(patch_ids))
    observables = faults.observables
    return observables.shape[0]


def _window_faults(
    model: Optional[fault_model_contracts.WindowErrorModel],
) -> Optional[fault_model_contracts.PlacedFaultModel]:
    """The model's graphlike faults, else its physical ones, else None.

    A run whose decoders and signal ask for no fault view still builds
    the window's model, with neither view.
    """
    if model is None:
        return None
    if model.graphlike_faults is not None:
        return model.graphlike_faults
    return model.physical_faults
