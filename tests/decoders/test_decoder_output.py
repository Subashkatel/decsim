"""The decoder side's outgoing sends: which link, how many bits, who commits.

A send is executed by an end of its hop (OMNeT++ cSimpleModule.cc:334;
gem5 coherent_xbar.hh:210-231 bills the port a packet left by), so the
correction that leaves a decoder for the Pauli frame leaves by the tier's
own output link, and the frame's priced write gates the commit.
"""

import functools

import decsim.decoders.decoder_output as decoder_output_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.program as program_records
import decsim.records.transfers as transfer_records
import decsim.records.windows as window_records


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
    output = decoder_output_module.DecoderOutput(transfers, frame)
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
    output = decoder_output_module.DecoderOutput(transfers, frame)
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
    frameless = decoder_output_module.DecoderOutput(transfers, None)
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
