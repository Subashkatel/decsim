"""A store's outgoing port: what it sends, and what it lands itself.

Whoever executes a send is an end of that hop, and the end a decoder
input leaves from is the store that holds the rounds (OMNeT++
cSimpleModule.cc:334, gem5 packet.hh:426-428). The other half of that
rule is here too: an input the decoder reads in place rides no link, and
the store lands it without asking the fabric for anything.
"""

import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.round_output as round_output


class _Transfers:
    """The WindowTransfers port, recording every send it is asked for."""

    def __init__(self) -> None:
        self.sends = []

    def send_for_window(
        self,
        path,
        window,
        operation,
        request_key,
        payload_bits,
        on_delivered,
    ) -> None:
        del window, operation, request_key, on_delivered
        self.sends.append((path, payload_bits))

    def send_for_job(
        self, path, job, *, payload_bits, request_key=None, on_delivered
    ) -> int:
        del job, request_key
        self.sends.append((path, payload_bits))
        on_delivered()
        return 3

    def send_boundary(self, attribution, payload_bits, on_delivered) -> None:
        del attribution, on_delivered
        path = transfer_records.LinkPath.DECODER_TO_DECODER
        self.sends.append((path, payload_bits))


def _output(transfers) -> round_output.RoundStoreOutput:
    return round_output.RoundStoreOutput(
        transfers,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "Buffer 0",
    )


def _job() -> decoding_records.DecodeJob:
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=1
    )


def test_the_recording_transfers_fill_the_port():
    """The port is what a store's output holds, so a stand-in fills it."""
    transfers = _Transfers()
    assert isinstance(transfers, ports.WindowTransfers)


def test_a_moved_input_leaves_by_this_stores_own_path():
    transfers = _Transfers()
    output = _output(transfers)
    job = _job()
    landed = []
    delay = output.send_input(job, lambda: landed.append(True))
    assert delay == 3
    assert landed == [True]
    assert job.input_source_name == "Buffer 0"
    assert transfers.sends == [
        (transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER, 0)
    ]


def test_a_held_input_rides_no_link_and_lands_now():
    """The rounds never left, so no send is asked for and no delay runs."""
    engine = engine_module.Engine()
    transfers = _Transfers()
    output = _output(transfers)
    job = _job()
    landed = []
    delay = output.land_held_input(job, lambda: landed.append(engine.now))
    assert delay == 0
    assert landed == [0]
    assert transfers.sends == []
    assert job.input_source_name == "Buffer 0"
