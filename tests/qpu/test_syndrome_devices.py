"""The circuit-less syndrome sources shape their payloads by the code card.

Sources: the SyndromeDevice seam in decsim/protocols.py; the rotated
surface code's d*d - 1 stabilizers per round (Stim's generated circuit,
src/stim/gen/gen_surface_code.cc).
"""

import pytest

from decsim.message import Operation
from decsim.qpu.code_geometry import SurfaceCodeModel
from decsim.qpu.syndrome_devices import SyndromeBitDevice, TimingOnlyDevice


def first_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    return payloads[0]


def test_a_timing_only_round_carries_no_bits_and_names_its_patch():
    device = TimingOnlyDevice()
    operation = Operation(id=4, name="memory", qubits=(2,), patches=(7,))
    payload = first_payload(device, operation, 3)
    assert (payload.operation_id, payload.patch_id, payload.round_index) == (
        4,
        7,
        3,
    )
    assert payload.bits is None


def test_a_stream_segment_reports_its_stream_and_global_round():
    device = TimingOnlyDevice()
    operation = Operation(
        id=4, name="tail", qubits=(2,), stream_id="s", stream_offset=6
    )
    payload = first_payload(device, operation, 2)
    assert (payload.operation_id, payload.round_index) == ("s", 8)


def test_fake_bits_are_as_wide_as_the_syndrome_up_to_the_cap():
    code = SurfaceCodeModel(distance=3)
    device = SyndromeBitDevice(code, seed=1)
    operation = Operation(id=1, name="memory", qubits=(0,))
    payload = first_payload(device, operation, 1)
    assert len(payload.bits) == 8
    assert payload.size_bits == 8
    assert payload.code == "rotated surface code (d=3)"
    wide_code = SurfaceCodeModel(distance=5)
    capped = SyndromeBitDevice(wide_code, seed=1, max_bit_count=10)
    capped_payload = first_payload(capped, operation, 1)
    assert len(capped_payload.bits) == 10


def test_one_payload_per_patch_when_asked():
    code = SurfaceCodeModel(distance=3)
    device = SyndromeBitDevice(code, seed=1, one_payload_per_patch=True)
    operation = Operation(id=1, name="merge", qubits=(0, 1), patches=(5, 6))
    payloads = device.round_payloads(operation, 1)
    patches = [payload.patch_id for payload in payloads]
    assert patches == [5, 6]


def test_neither_source_can_finalize_a_stream_round():
    operation = Operation(id=1, name="tail", qubits=(0,))
    timing_only = TimingOnlyDevice()
    with pytest.raises(ValueError, match="finalize"):
        timing_only.finalize_stream_round(operation, 3)
    code = SurfaceCodeModel(distance=3)
    fake_bits = SyndromeBitDevice(code, seed=1)
    with pytest.raises(ValueError, match="finalize"):
        fake_bits.finalize_stream_round(operation, 3)
