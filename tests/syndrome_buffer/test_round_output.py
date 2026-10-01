"""A store's outgoing port: what it sends, and what it lands itself.

Whoever executes a send is an end of that hop, and the end a decoder
input leaves from is the store that holds the rounds
(omnetpp src/sim/csimplemodule.cc:333-334, omnetpp-6.1.0;
gem5 packet.hh:424-431). The other half of that
rule is here too: an input the decoder reads in place rides no link, and
the store lands it without asking the fabric for anything. The raw
rounds a forming decoder needs before a job's first ride the job's one
read and one transfer, as one gem5 DMA request covers its whole range
(src/dev/dma_device.cc:195-207), and are priced as the read words and
bits they are (src/mem/simple_mem.cc:154).
"""

import pytest
import stim

import decsim.config as config
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
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


class _OneTable:
    """A source whose recipes are one circuit's, for every operation."""

    def __init__(self, table) -> None:
        self.table = table

    def formation_table(self, operation_id):
        """The one table."""
        del operation_id
        return self.table


def _lookback_placement():
    """The real placement at the weak decoder, over a two-round lookback.

    Ten rounds of one qubit; round r's detector is rec[-1] ^ rec[-3], so
    a decoder that joins at round 5 reads rounds 3 and 4 raw.
    """
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1]\n"
        "REPEAT 8 {\nM 0\nDETECTOR rec[-1] rec[-3]\n}\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    table = detector_formation.build_formation_table(
        circuit, 10, measurement_rounds=measurement_rounds
    )
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",)
    )
    source = _OneTable(table)
    return formation.SeatedFormation(source, settings)


def _ported_store(engine) -> syndrome_buffer_module.SyndromeBuffer:
    """A store at 10 MHz whose one read port moves a word a cycle.

    A word is two bits, one round of this file's fragments.
    """
    clocks = config.ClockSettings.from_yaml({"storage": 10})
    section = {
        "kind": "ported_syndrome_buffer",
        "clock": "storage",
        "word_bits": 2,
    }
    settings = syndrome_buffer_settings.SyndromeBufferSettings.from_yaml(
        section,
        "weak_syndrome_buffer",
        clocks,
        ported_syndrome_buffer.SYNDROME_BUFFERS,
    )
    return ported_syndrome_buffer.PortedSyndromeBuffer(settings, engine)


def _store_rounds(store, round_indices) -> None:
    for round_index in round_indices:
        fragment = _fragment(round_index)
        packet = round_records.SyndromeRoundPacket(1, round_index, (fragment,))
        store.accept_packed_round(packet, publication_tick=0)


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


def test_the_round_before_leaves_with_a_job_whose_decoder_reads_it():
    """Round 5 reads round 3, which rides the job's one move, and not 4."""
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    _store_rounds(store, (2, 3, 4))
    output = _output(engine, transfers, store)
    output.detection_events = _lookback_placement()
    job = _job(5)

    output.send_input(job, lambda: None)

    assert job.rounds_before == (_fragment(3),)
    assert transfers.sends == [(output.path, 4)]


def test_a_job_whose_first_round_reads_only_itself_carries_nothing_before():
    """Round 2's detector is rec[-1] alone: one round of two bits moves."""
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _store(engine)
    _store_rounds(store, (1,))
    output = _output(engine, transfers, store)
    output.detection_events = _lookback_placement()
    job = _job(2)

    output.send_input(job, lambda: None)

    assert job.rounds_before == ()
    assert transfers.sends == [(output.path, 2)]


def test_a_round_before_the_store_let_go_of_stops_the_read_by_name():
    """Round 5 reads round 3, which a program reaching further lost."""
    engine = engine_module.Engine()
    store = _store(engine)
    _store_rounds(store, (2, 4))
    transfers = _Transfers()
    output = _output(engine, transfers, store)
    output.detection_events = _lookback_placement()
    job = _job(5)

    with pytest.raises(RuntimeError) as refusal:
        output.send_input(job, lambda: None)

    assert str(refusal.value) == (
        "a job from round 5 of operation 1 reads round 3, which the weak "
        "syndrome buffer does not hold: its holds keep the rounds the "
        "program's declared fragments read, so a round_circuit that "
        "reads further back than they do cannot be formed at the "
        "weak_decoder"
    )


def test_the_round_before_is_read_in_the_jobs_one_port_booking():
    """Two words on one read port, a cycle each from tick 0, then the link.

    SimpleMemory is busy for the size it moves (gem5
    src/mem/simple_mem.cc:154), so round 3, which round 5 reads, costs
    its word in the same read as the job's own round, with no second
    request.
    """
    engine = engine_module.Engine()
    transfers = _Transfers()
    store = _ported_store(engine)
    _store_rounds(store, (3, 4, 5))
    output = _output(engine, transfers, store)
    output.detection_events = _lookback_placement()
    job = _job(5)

    delay = output.send_input(job, lambda: None)

    period_ticks = store.settings.clock.period_ticks
    assert delay == 2 * period_ticks + LINK_DELAY_TICKS


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


def test_an_idle_round_leaves_when_the_stores_read_of_it_completes():
    """A timing-only round leaves by a read of the store, like an input.

    gem5 prices an access where the memory serves it
    (src/mem/simple_mem.cc:154-174). From tick 1 on a 10-tick clock, 3
    read cycles end at the edge 40; the send starts there and the slot
    frees at its delivery.
    """
    engine = engine_module.Engine()
    engine.now = 1
    transfers = _Transfers()
    store = _store(engine, read_cycles=3)
    output = _output(engine, transfers, store)
    idle_round = round_records.QPUReadout(("idle", 1, 0), (0,), 1, size_bits=2)
    packet = round_records.SyndromeRoundPacket(("idle", 1, 0), 1, (idle_round,))
    route = round_records.SyndromePacketRoute.feedback_memory_round(1)
    packed = round_records.PackedRound(packet, route, 2)
    store.accept_packed_round(packet, publication_tick=None)
    delivered = []

    output.send_memory_round(packed, lambda: delivered.append(engine.now))
    sent_before_the_read_ends = list(transfers.sends)
    engine.run()

    assert sent_before_the_read_ends == []
    assert transfers.sends == [(output.path, 2)]
    assert delivered == [40]
    assert store.occupancy == 0


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
