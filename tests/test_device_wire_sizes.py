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
from decsim.devices import SyndromeBitDevice
from decsim.message import Operation


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


def test_stim_payloads_report_their_bit_counts():
    stim = pytest.importorskip("stim")
    from decsim.adapters.stim_device import StimDevice
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=3, rounds=3,
        after_clifford_depolarization=0.001)
    op = Operation(0, "mem", (0,), circuit=circuit)
    device = StimDevice(seed=0)
    device.begin_operation(op)
    for round_index in (1, 2, 3):
        for payload in device.round_payloads(op, round_index):
            assert payload.size_bits == len(payload.bits)
    (idle,) = device.idle_round_payloads(op, stream_id=0, global_round=1,
                                         patch=0)
    assert idle.size_bits == len(idle.bits)
