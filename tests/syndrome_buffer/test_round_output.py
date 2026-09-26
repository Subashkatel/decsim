"""A store's outgoing port: what it sends, and what it lands itself.

Whoever executes a send is an end of that hop, and the end a decoder
input leaves from is the store that holds the rounds
(omnetpp src/sim/csimplemodule.cc:333-334, omnetpp-6.1.0;
gem5 packet.hh:424-431). The other half of that
rule is here too: an input the decoder reads in place rides no link, and
the store lands it without asking the fabric for anything.
"""

import pytest

import decsim.config as config
import decsim.engine as engine_module
import decsim.ports as ports
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.syndrome_buffer.ported_syndrome_buffer as ported_syndrome_buffer
import decsim.syndrome_buffer.round_output as round_output
import decsim.syndrome_buffer.settings as syndrome_buffer_settings
import decsim.syndrome_buffer.syndrome_buffer as syndrome_buffer_module
import tests.declared_run as declared_run

# the link every send in this file rides: 3 ticks, whenever it starts
LINK_DELAY_TICKS = 3


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

    def send_for_round(self, path, packet, payload_bits, on_delivered) -> None:
        del packet
        self.sends.append((path, payload_bits))
        on_delivered()

    def send_boundary(
        self, path, attribution, payload_bits, on_delivered
    ) -> None:
        del attribution, on_delivered
        self.sends.append((path, payload_bits))

    def send_region(self, path, region, on_delivered) -> int:
        del on_delivered
        self.sends.append((path, region.wire_bits))
        return 3


class _Link:
    """The fabric, answering what a send starting at a tick would pay."""

    def __init__(self) -> None:
        self.asked_at = []

    def expected_delay_ticks(self, path, payload_bits, now_ticks) -> int:
        del path, payload_bits
        self.asked_at.append(now_ticks)
        return LINK_DELAY_TICKS


def _store(engine, read_cycles=0) -> syndrome_buffer_module.SyndromeBuffer:
    """An unbounded store on a 10-tick clock, its read priced as asked."""
    clock = config.Clock(10)
    costs = syndrome_buffer_module.SyndromeBuffer.Settings(
        read_cycles=read_cycles
    )
    settings = syndrome_buffer_settings.SyndromeBufferSettings(
        clock=clock, row_settings=costs
    )
    return syndrome_buffer_module.SyndromeBuffer(settings, engine)


class _DecoderThatFormedNothing:
    """The placement port, whose decoder seat has formed no round yet."""

    clock = None

    def __init__(self) -> None:
        self.asked = []

    def forms_at(self, seat) -> bool:
        del seat
        return True

    def form_at(self, seat, fragments, round_before=()) -> tuple:
        del seat, round_before
        return fragments

    def needs_the_round_before(self, seat, operation_id, round_index) -> bool:
        self.asked.append((seat, operation_id, round_index))
        return True

    def cycles_at(self, seat, round_count) -> int:
        del seat, round_count
        return 0


def _output(
    engine, transfers, store, reads_in_place=False
) -> round_output.SyndromeBufferOutput:
    output = round_output.SyndromeBufferOutput(
        engine,
        transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER,
        "weak syndrome buffer",
        reads_in_place,
        "weak_decoder",
    )
    output.transfers = transfers
    output.store = store
    output.link = _Link()
    return output


def _fragment(round_index=1) -> round_records.RetainedSyndromeFragment:
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=(1, 0),
        size_bits=2,
        fragment_index=0,
    )


def _job(round_index=1) -> decoding_records.DecodeJob:
    fragment = _fragment(round_index)
    return decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=1, payloads=[fragment]
    )


def test_the_round_before_leaves_with_a_job_whose_decoder_needs_it():
    """Its bits ride the job's move, and the job carries its fragments."""
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    round_one = _fragment(1)
    stored = round_records.SyndromeRoundPacket(1, 1, (round_one,))
    store.accept_packed_round(stored, publication_tick=0)
    output = _output(engine, transfers, store)
    output.detection_events = _DecoderThatFormedNothing()
    job = _job(2)

    output.send_input(job, lambda: None)

    assert output.detection_events.asked == [("weak_decoder", 1, 2)]
    assert job.round_before == (_fragment(1),)
    assert transfers.sends == [(output.path, 4)]


def test_the_recording_transfers_fill_the_port():
    """The port is what a store's output holds, so a stand-in fills it."""
    transfers = _Transfers()
    assert isinstance(transfers, ports.WindowTransfers)


def test_a_moved_input_leaves_by_this_stores_own_path():
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    output = _output(engine, transfers, store)
    job = _job()
    landed = []
    delay = output.send_input(job, lambda: landed.append(True))
    assert delay == 3
    assert landed == [True]
    assert job.input_source_name == "weak syndrome buffer"
    assert transfers.sends == [
        (transfer_records.LinkPath.WEAK_BUFFER_TO_WEAK_DECODER, 2)
    ]


def test_an_input_leaves_when_the_stores_read_of_it_completes():
    """The read is paid when the bits leave, at dispatch, then the link.

    From tick 1 on a 10-tick clock, 3 read cycles end at the edge 40; the
    delay expected is that read and the link's 3 ticks asked at 40.
    """
    engine = engine_module.Engine()
    engine.now = 1
    transfers = _Transfers()
    store = _store(engine, read_cycles=3)
    output = _output(engine, transfers, store)
    job = _job()
    landed = []

    delay = output.send_input(job, lambda: landed.append(engine.now))
    sent_before_the_read_ends = list(transfers.sends)
    engine.run()

    assert delay == 39 + LINK_DELAY_TICKS
    assert output.link.asked_at == [40]
    assert sent_before_the_read_ends == []
    assert landed == [40]


def test_an_input_read_in_place_lands_at_the_reads_end_and_rides_no_link():
    """The unit reads the store's words: one read, priced once, no move."""
    engine = engine_module.Engine()
    engine.now = 1
    transfers = _Transfers()
    store = _store(engine, read_cycles=3)
    output = _output(engine, transfers, store, reads_in_place=True)
    job = _job()
    landed = []

    delay = output.send_input(job, lambda: landed.append(engine.now))
    engine.run()

    assert delay == 39
    assert landed == [40]
    assert transfers.sends == []


def test_a_held_input_rides_no_link_and_lands_now():
    """The rounds never left, so no send is asked for and no delay runs."""
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    output = _output(engine, transfers, store)
    job = _job()
    landed = []
    send = output.input_send_for(job, True)
    delay = send(lambda: landed.append(engine.now))
    assert delay == 0
    assert landed == [0]
    assert transfers.sends == []
    assert job.input_source_name == "weak syndrome buffer"


def test_a_store_names_itself_when_the_job_is_bound_and_not_when_it_sends():
    """A resubmitted job whose rounds never left runs no send.

    Its landing is land_held_input, so a store that named itself only
    inside the send would leave every observer of that job with no
    source at all.
    """
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    output = _output(engine, transfers, store)
    job = _job()

    output.input_send_for(job, False)

    assert job.input_source_name == "weak syndrome buffer"
    assert transfers.sends == []


@pytest.mark.parametrize("decoder_input", ["copy", "in_place"])
def test_read_cycles_delay_the_decode_from_dispatch_on(decoder_input):
    """Readiness, queueing and dispatch stand; the decode and frame move."""
    clocks = config.ClockSettings.from_yaml({"storage": 1.0})
    section = {"clock": "storage", "read_cycles": 3}
    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section,
        "weak_syndrome_buffer",
        clocks,
        ported_syndrome_buffer.SYNDROME_BUFFERS,
    )
    free = declared_run.weak_only_run(decoder_input=decoder_input)
    charged = declared_run.weak_only_run(
        weak_syndrome_buffer=settings, decoder_input=decoder_input
    )
    free_ticks = declared_run.reaction_ticks(free)
    charged_ticks = declared_run.reaction_ticks(charged)
    paired = zip(charged_ticks, free_ticks, strict=True)
    shifts = [charged_tick - free_tick for charged_tick, free_tick in paired]
    expected = 3 * settings.clock.period_ticks
    assert shifts == [0, 0, 0, expected, expected, expected]
