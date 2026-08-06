#==================================================================
# TESTS FOR SYNDROME PAYLOAD WIRE SIZES
# Bandwidth-aware links can only price a payload whose size_bits is set,
# so every device that fills in bits must also report size_bits ==
# len(bits) -- aggregate, per-patch, idle-round, and Stim-backed alike.
# The aggregate fake device must also count PHYSICAL patches
# (op.patches), not logical qubits, when the two differ.
#==================================================================
import pytest

from decsim.codes import SurfaceCodeModel
from decsim.devices import ClockedDevice, SyndromeBitDevice, TimingOnlyDevice
from decsim.message import Operation, SyndromePayload


def test_aggregate_fake_payload_reports_its_bit_count():
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), max_bits=5)
    (payload,) = device.round_payloads(Operation(0, "mem", (0,)), 1)
    assert len(payload.bits) == 5
    assert payload.size_bits == 5


def test_per_patch_fake_payloads_report_their_bit_counts():
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), max_bits=5,
                               per_patch=True)
    payloads = device.round_payloads(Operation(0, "mem", (0, 1)), 1)
    assert len(payloads) == 2
    for payload in payloads:
        assert payload.size_bits == len(payload.bits) == 5


def test_clocked_device_relay_does_not_mutate_source_fragment_counts():
    class RecordingController:
        def __init__(self):
            self.payloads = []

        def relay_syndrome(self, payload, route):
            self.payloads.append(payload)

    controller = RecordingController()
    clocked = ClockedDevice(
        engine=None,
        device=None,
        controller=controller,
        round_count_by_operation_id={},
    )
    source_payloads = [
        SyndromePayload(0, "north", 1),
        SyndromePayload(0, "south", 1),
    ]

    clocked.relay_payloads(
        source_payloads, Operation(0, "round", ("north", "south"))
    )

    assert [payload.n_fragments for payload in source_payloads] == [1, 1]
    assert [payload.n_fragments for payload in controller.payloads] == [2, 2]
    assert all(
        relayed is not source
        for relayed, source in zip(controller.payloads, source_payloads)
    )


def test_explicit_fragment_slot_rejects_multiple_payloads_before_relay():
    class RecordingController:
        def __init__(self):
            self.payloads = []

        def relay_syndrome(self, payload, route):
            self.payloads.append(payload)

    controller = RecordingController()
    clocked = ClockedDevice(
        engine=None, device=None, controller=controller,
        round_count_by_operation_id={},
    )
    operation = Operation(
        0, "declared fragment", (0,),
        syndrome_fragment_index=0, syndrome_fragment_count=2,
    )
    with pytest.raises(ValueError, match="one payload"):
        clocked.relay_payloads(
            [SyndromePayload(0, 0, 1), SyndromePayload(0, 1, 1)],
            operation,
        )
    assert controller.payloads == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({
            "emits_detector_data": False, "stream_id": 7, "stream_offset": 0,
            "syndrome_fragment_index": 1, "syndrome_fragment_count": 2,
        }, "emitter"),
        ({
            "stream_offset": 0,
            "syndrome_fragment_index": 1, "syndrome_fragment_count": 2,
        }, "stream_id"),
        ({
            "stream_id": 7,
            "syndrome_fragment_index": 1, "syndrome_fragment_count": 2,
        }, "stream_offset"),
        ({"stream_id": 7, "stream_offset": 0}, "fragment slot"),
    ],
)
def test_malformed_stream_finalizer_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Operation(0, "bad finalizer", (0,),
                  finalizes_stream_round=True, **kwargs)


def test_nonphysical_devices_fail_closed_on_terminal_finalization():
    operation = Operation(
        0, "terminal", (0,), stream_id=7, stream_offset=0,
        finalizes_stream_round=True,
        syndrome_fragment_index=1, syndrome_fragment_count=2,
    )
    for device in (
        TimingOnlyDevice(),
        SyndromeBitDevice(SurfaceCodeModel(d=3)),
    ):
        with pytest.raises(ValueError, match="cannot finalize"):
            device.finalize_stream_round(operation)


def test_idle_round_fake_payload_reports_its_bit_count():
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), max_bits=5)
    (payload,) = device.idle_round_payloads(Operation(0, "mem", (0,)),
                                            stream_id=0, global_round=1,
                                            patch=0)
    assert payload.size_bits == len(payload.bits) == 5


def test_aggregate_fake_device_counts_physical_patches_not_qubits():
    # one physical d=3 patch (8 syndrome bits/round) encodes two logical
    # qubits; the payload must carry 8 bits, not 16
    device = SyndromeBitDevice(SurfaceCodeModel(d=3), max_bits=64)
    op = Operation(0, "encoded pair", (0, 1), patches=(10,))
    (payload,) = device.round_payloads(op, 1)
    assert payload.patch_id == 10
    assert len(payload.bits) == 8
    assert payload.size_bits == 8


def test_fake_device_run_seed_binding_is_atomic_and_replayable():
    code = SurfaceCodeModel(d=3)
    operation = Operation(0, "mem", (0,))
    first = SyndromeBitDevice(code)
    reservation = first.reserve_run_seed(41)
    assert (reservation.proposed_seed_source, reservation.proposed_seed) == ("derived", 41)
    first.commit_run_seed(reservation)

    second = SyndromeBitDevice(code)
    second_reservation = second.reserve_run_seed(41)
    second.commit_run_seed(second_reservation)

    assert first.round_payloads(operation, 1)[0].bits == \
        second.round_payloads(operation, 1)[0].bits


def test_fake_device_rng_state_has_no_public_bypass():
    device = SyndromeBitDevice(SurfaceCodeModel(d=3))

    assert not hasattr(device, "rng")


def test_fake_device_explicit_seed_conflicts_and_direct_use_is_monotone():
    code = SurfaceCodeModel(d=3)
    explicit = SyndromeBitDevice(code, seed=0)
    with pytest.raises(ValueError, match=r"SyndromeBitDevice.*explicit seed"):
        explicit.reserve_run_seed(41)

    used = SyndromeBitDevice(code)
    used.round_payloads(Operation(0, "mem", (0,)), 1)
    with pytest.raises(ValueError, match=r"SyndromeBitDevice.*already used"):
        used.reserve_run_seed(41)


def test_stim_payloads_report_their_bit_counts():
    stim = pytest.importorskip("stim")
    from decsim.adapters.stim_device import StimDevice
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=0.001)
    op = Operation(0, "mem", (0,), circuit=circuit)
    device = StimDevice(seed=0)
    device.begin_operation(op, 3)
    for round_index in (1, 2, 3):
        for payload in device.round_payloads(op, round_index):
            assert payload.size_bits == len(payload.bits)
    (idle,) = device.idle_round_payloads(op, stream_id=0, global_round=1,
                                         patch=0)
    assert idle.size_bits == len(idle.bits)
