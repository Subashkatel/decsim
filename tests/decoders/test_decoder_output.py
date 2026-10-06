"""The decoder side's outgoing sends: which link, how many bits, who commits.

A send is executed by an end of its hop
(omnetpp-6.1.0 src/sim/csimplemodule.cc:333-334;
gem5 coherent_xbar.cc:354-357 bills the port a packet left by), so the
correction that leaves a decoder for the Pauli frame leaves by its
route's hops, and the frame's priced write gates the commit. Through the
weak chip, the chip's frame update is one cycle (Yang et al. 2605.04892
Table I).
"""

import functools

import numpy
import pytest
import scipy.sparse

import decsim.config as config
import decsim.decoders.decoder_output as decoder_output_module
import decsim.engine as engine_module
import decsim.records.decoding as decoding_records
import decsim.records.fault_model_contracts as fault_models
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
        self.sent_ticks = []

    def send_for_window(
        self, path, window, _operation, request_key, payload_bits, on_delivered
    ):
        self.sent.append((path, window.key, request_key.tier, payload_bits))
        self.sent_ticks.append(self.engine.now)
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
        self.asked = []

    def expected_delay_ticks(self, path, payload_bits, now_ticks) -> int:
        del path
        self.asked.append((now_ticks, payload_bits))
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


def _published_bits(tier, logical_observables: tuple) -> int:
    """The bits one answer of that tier puts on its output link."""
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    result = decoding_records.DecodeResult(
        4, 1, logical_observables=logical_observables
    )
    request_key = _request_key(tier)
    window = _window()
    operation = _operation(1)
    output.publish(window, operation, result, request_key, _ignore)
    ((_path, _key, _tier, payload_bits),) = transfers.sent
    return payload_bits


def test_each_tier_publishes_over_its_own_output_link():
    """A weak result rides weak_decoder_to_frame, a direct strong one its own.

    The direct route is the default: the strong host sends home itself.
    """
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    frame = _Frame(engine, 3)
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = transfers
    output.frame = frame
    window = _window()
    operation = _operation(1)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1,))
    weak_key = _request_key(window_records.DecoderTier.WEAK)
    strong_key = _request_key(window_records.DecoderTier.STRONG)
    output.publish(window, operation, result, weak_key, _ignore)
    output.publish(window, operation, result, strong_key, _ignore)
    engine.run()
    sends = [(path, bits) for path, _key, _tier, bits in transfers.sent]
    assert sends == [
        (transfer_records.LinkPath.WEAK_DECODER_TO_FRAME, 1),
        (transfer_records.LinkPath.STRONG_DECODER_TO_FRAME, 64 + 1),
    ]


def test_a_strong_answer_is_its_flip_behind_the_requests_name():
    """It crosses the wall out of order, so it says which request it answers."""
    one_flip = (1,)
    strong = window_records.DecoderTier.STRONG
    assert _published_bits(strong, one_flip) == 64 + 1


def test_a_weak_answer_is_its_flip_alone():
    """It stays on the board with its frame, and no name is priced there."""
    one_flip = (1,)
    weak = window_records.DecoderTier.WEAK
    assert _published_bits(weak, one_flip) == 1


def test_a_strong_answer_through_the_weak_chip_commits_there_then_goes_home():
    """Down with its name, one chip cycle later home as its flip alone.

    The frame takes one write, and the commit is heard once.
    """
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    frame = _Frame(engine, 3)
    clock = config.Clock(4)
    output = decoder_output_module.DecoderOutput(
        engine, strong_answer_route="through_weak_chip", clock=clock
    )
    output.transfers = transfers
    output.frame = frame
    committed = []
    on_committed = functools.partial(_note, committed, engine)
    result = decoding_records.DecodeResult(4, 1, logical_observables=(1,))
    strong = window_records.DecoderTier.STRONG
    strong_key = _request_key(strong)
    window = _window()
    operation = _operation(1)
    output.publish(window, operation, result, strong_key, on_committed)
    engine.run()
    paths = transfer_records.LinkPath
    assert transfers.sent == [
        (paths.STRONG_DECODER_TO_WEAK_DECODER, (4, 1), strong, 64 + 1),
        (paths.WEAK_DECODER_TO_FRAME, (4, 1), strong, 1),
    ]
    assert transfers.sent_ticks == [0, 8]
    assert frame.commits == [(12, (4, 1), (1,))]
    assert committed == [15]


def _selection_bits(route: str, weak_job) -> int:
    """The bits one selection puts on the escalation hop under a route."""
    engine = engine_module.Engine()
    transfers = _Transfers(engine, 4)
    output = decoder_output_module.DecoderOutput(
        engine, strong_answer_route=route
    )
    output.transfers = transfers
    strong_key = _request_key(window_records.DecoderTier.STRONG)
    output.send_selection(weak_job, strong_key, _ignore)
    ((path, _label, _tier, payload_bits),) = transfers.sent
    assert path is transfer_records.LinkPath.WEAK_DECODER_TO_STRONG_DECODER
    return payload_bits


def _timing_only_job(patch_ids: tuple) -> decoding_records.DecodeJob:
    """A weak job with no error model, reading one round of those patches."""
    fragment = round_records.RetainedSyndromeFragment(
        operation_id=4,
        patch_ids=patch_ids,
        round_index=2,
        bits=None,
        size_bits=8,
        fragment_index=0,
    )
    return decoding_records.DecodeJob(
        operation_id=4, window_id=1, round_count=5, payloads=[fragment]
    )


def test_a_selection_through_the_weak_chip_is_the_requests_name_alone():
    """The chip keeps its crossing commit, so only the 8-byte name goes up."""
    weak_job = _timing_only_job((0,))
    assert _selection_bits("through_weak_chip", weak_job) == 64


def test_a_direct_selection_carries_a_crossing_bit_per_observable():
    """The host joins the crossing commit, so it rides with the name.

    The count is the window model's observables, here two.
    """
    observables = scipy.sparse.csc_matrix([[0], [0]], dtype=numpy.uint8)
    check = scipy.sparse.csc_matrix([[1]], dtype=numpy.uint8)
    placed = fault_models.PlacedFaultModel(
        representation=fault_models.FaultRepresentation.PHYSICAL,
        check=check,
        priors=[0.1],
        observables=observables,
        owned=[True],
        source_fault_ids=[0],
        boundary_flips={0: [0]},
    )
    model = fault_models.WindowErrorModel(
        detector_ids=(0,),
        detector_coordinates=None,
        defect_positions={0: (4, 0)},
        first_commit_round=4,
        graphlike_faults=None,
        physical_faults=placed,
    )
    weak_job = decoding_records.DecodeJob(
        operation_id=4, window_id=1, round_count=5, detector_error_model=model
    )
    assert _selection_bits("direct", weak_job) == 64 + 2


def test_a_direct_timing_only_selection_carries_a_bit_per_patch():
    """With no model, as result_payload_bits: one observable per patch."""
    weak_job = _timing_only_job((0, 1, 2))
    assert _selection_bits("direct", weak_job) == 64 + 3


def test_a_model_with_no_fault_view_selects_as_timing_only():
    """A run that asks for no fault view still builds a model, holding none."""
    weak_job = _timing_only_job((0, 1))
    weak_job.detector_error_model = fault_models.WindowErrorModel(
        detector_ids=(0,),
        detector_coordinates=None,
        defect_positions={0: (4, 0)},
        first_commit_round=4,
        graphlike_faults=None,
        physical_faults=None,
    )
    assert _selection_bits("direct", weak_job) == 64 + 2


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


def test_a_region_that_waits_on_its_read_is_priced_as_its_message():
    """The link is asked about the name and the rounds, from the read's end."""
    engine = engine_module.Engine()
    engine.now = 100
    output = decoder_output_module.DecoderOutput(engine)
    output.transfers = _RegionTransfers(engine)
    output.weak_store = _WeakStore(engine, 30)
    output.link = _Link()
    region = _region()

    output.send_region(region, _ignore)

    assert output.link.asked == [(130, 64 + 16)]
