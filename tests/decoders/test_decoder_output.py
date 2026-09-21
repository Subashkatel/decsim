"""The decoder side's outgoing sends: which link, how many bits, who commits.

A send is executed by an end of its hop
(omnetpp-6.1.0 src/sim/csimplemodule.cc:333-334;
gem5 coherent_xbar.cc:354-357 bills the port a packet left by), so the
correction that leaves a decoder for the Pauli frame leaves by the tier's
own output link, and the frame's priced write gates the commit.
"""

import functools

import pytest

import decsim.decoders.decoder_output as decoder_output_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records

REGION_LINK_TICKS = 5


class _Transfers:
    """A link that delivers after a fixed delay, recording every send."""

    def __init__(self, engine, delay_ticks: int) -> None:
        self.engine = engine
        self.delay_ticks = delay_ticks
        self.sent = []

    def send_for_window(
        self, path, window, _operation, request_key, payload_bits, on_delivered
    ):
        self.sent.append((path, window.key, request_key.tier, payload_bits))
        self.engine.schedule(self.delay_ticks, on_delivered)

    def send_for_job(
        self, path, job, *, payload_bits, request_key, on_delivered
    ):
        self.sent.append((path, job.label, request_key.tier, payload_bits))
        self.engine.schedule(self.delay_ticks, on_delivered)
        return self.delay_ticks


class _RegionTransfers:
    """A link that records the tick each region leaves on."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.left_at = []

    def send_region(self, path, region, on_delivered):
        del path, region, on_delivered
        self.left_at.append(self.engine.now)
        return REGION_LINK_TICKS


class _WeakStore:
    """A store whose every read ends a fixed time after it is booked."""

    def __init__(self, engine, read_ticks: int) -> None:
        self.engine = engine
        self.read_ticks = read_ticks
        self.read_keys = []

    def book_read(self, round_keys) -> int:
        self.read_keys.append(round_keys)
        return self.engine.now + self.read_ticks


class _Link:
    """The fabric, answering what a send starting at a tick would pay."""

    def __init__(self) -> None:
        self.asked_at = []

    def expected_delay_ticks(self, path, payload_bits, now_ticks) -> int:
        del path, payload_bits
        self.asked_at.append(now_ticks)
        return REGION_LINK_TICKS


class _Frame:
    def __init__(self, engine, commit_ticks: int) -> None:
        self.engine = engine
        self.commit_ticks = commit_ticks
        self.commits = []

    def commit_correction(
        self, *, window_key, logical_observables, request_key, on_committed
    ):
        del request_key
        self.commits.append((self.engine.now, window_key, logical_observables))
        self.engine.schedule(self.commit_ticks, on_committed)


def _window() -> window_records.Window:
    return window_records.Window(
        operation_id=4,
        window_index=1,
        commit_lo=4,
        commit_hi=6,
        buffer_hi=8,
        round_count=5,
    )


def _operation(patch_count: int) -> program_records.Operation:
    patches = tuple(range(patch_count))
    return program_records.Operation(4, "logical", (0,), patches=patches)


def _request_key(tier) -> window_records.DecoderRequestKey:
    return window_records.DecoderRequestKey(4, 1, tier, 0)


def test_each_tier_publishes_over_its_own_output_link():
    """A weak result rides weak_decoder_to_frame, a strong one its own."""
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    frame = _Frame(engine, 3)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    output.frame = frame
    window = _window()
    operation = _operation(1)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1,))
    tiers = window_records.DecoderTier
    for tier in (tiers.WEAK, tiers.STRONG):
        request_key = _request_key(tier)
        output.publish(window, operation, result, request_key, _ignore)
    engine.run()
    paths = []
    for path, _key, _tier, _bits in transfers.sent:
        paths.append(path)
    assert paths == [
        transfer_records.LinkPath.WEAK_DECODER_TO_FRAME,
        transfer_records.LinkPath.STRONG_DECODER_TO_FRAME,
    ]


def test_a_selection_is_sent_as_one_control_word():
    """A message that only names a request is still 8 bytes on the wire."""
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    weak_job = decoding_records.DecodeJob(
        operation_id=4, window_id=1, round_count=5, label="weak"
    )
    strong_key = _request_key(window_records.DecoderTier.STRONG)
    output.send_selection(weak_job, strong_key, _ignore)
    escalation_path = transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER
    tiers = window_records.DecoderTier
    assert transfers.sent == [(escalation_path, "weak", tiers.STRONG, 64)]


def test_a_result_is_one_bit_per_logical_observable():
    """A timing-only result stands for one observable per patch."""
    operation = _operation(3)
    payload_bits = decoder_output_module.result_payload_bits
    with_bits = decoding_records.DecodeResult(1, 0, logical_observables=(0, 1))
    assert payload_bits(with_bits, operation) == 2
    timing_only = decoding_records.DecodeResult(1, 0)
    assert payload_bits(timing_only, operation) == 3


def test_the_frames_write_gates_the_commit_and_a_frameless_run_does_not():
    """The correction is installed at the delivery, then charged."""
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    frame = _Frame(engine, 3)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    output.frame = frame
    committed = []
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1,))
    key = _request_key(window_records.DecoderTier.WEAK)
    window = _window()
    operation = _operation(1)
    on_committed = functools.partial(_note, committed, engine)
    output.publish(window, operation, result, key, on_committed)
    engine.run()
    assert frame.commits == [(4, (4, 1), (1,))]
    assert committed == [7]
    frameless = decoder_output_module.DecoderOutput(engine)
    frameless.transfers = transfers
    later = []
    on_later = functools.partial(_note, later, engine)
    frameless.publish(window, operation, result, key, on_later)
    engine.run()
    assert later == [11]


def _note(ticks: list, engine) -> None:
    """Record the tick a commit callback ran at."""
    ticks.append(engine.now)


def _ignore() -> None:
    pass


def _region() -> round_records.EscalatedRegion:
    """Rounds 2 and 3 of operation 4, each of one 8-bit fragment."""
    packets = []
    for round_index in (2, 3):
        fragment = round_records.RetainedSyndromeFragment(
            operation_id=4,
            patch_ids=(0,),
            round_index=round_index,
            bits=None,
            size_bits=8,
            fragment_index=0,
        )
        packet = round_records.SyndromeRoundPacket(4, round_index, (fragment,))
        packets.append(packet)
    strong_key = _request_key(window_records.DecoderTier.STRONG)
    return round_records.EscalatedRegion.of(strong_key, tuple(packets))


@pytest.mark.parametrize("read_ticks", [0, 30])
def test_an_escalated_region_leaves_when_its_weak_store_read_ends(read_ticks):
    """The read of the carried rounds is priced by the weak store, once."""
    engine = engine_module.Engine()
    engine.now = 100
    transfers = _RegionTransfers(engine)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    output.weak_store = _WeakStore(engine, read_ticks)
    output.link = _Link()
    region = _region()

    expected_delay = output.send_region(region, _ignore)
    engine.run()

    assert output.weak_store.read_keys == [((4, 2), (4, 3))]
    assert transfers.left_at == [100 + read_ticks]
    assert expected_delay == read_ticks + REGION_LINK_TICKS
