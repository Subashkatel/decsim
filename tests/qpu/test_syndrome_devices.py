"""The circuit-less syndrome sources shape their payloads by the code card.

Sources: the SyndromeDevice seam in decsim/ports.py; the rotated
surface code's d*d - 1 stabilizers per round (Stim's generated circuit,
src/stim/gen/gen_surface_code.cc).
"""

import pytest

import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.program as program_records
import decsim.records.rounds as round_records


def first_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    return payloads[0]


def test_a_timing_only_round_carries_no_bits_and_names_its_patch():
    device = syndrome_devices.TimingOnlyDevice()
    operation = program_records.Operation(
        id=4, name="memory", qubits=(2,), patches=(7,)
    )
    payload = first_payload(device, operation, 3)
    assert payload == round_records.QPUReadout(4, 7, 3)


def test_a_stream_segment_reports_its_stream_and_global_round():
    device = syndrome_devices.TimingOnlyDevice()
    operation = program_records.Operation(
        id=4, name="tail", qubits=(2,), stream_id="s", stream_offset=6
    )
    payload = first_payload(device, operation, 2)
    assert payload == round_records.QPUReadout("s", 2, 8)


def test_fake_bits_are_as_wide_as_the_syndrome():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    payload = first_payload(device, operation, 1)
    assert len(payload.bits) == 8
    assert payload.size_bits == 8
    assert payload.code == "rotated surface code (d=3)"


def test_fake_bits_are_capped_at_max_bit_count():
    wide_code = code_geometry.SurfaceCodeModel(distance=5)
    capped = syndrome_devices.SyndromeBitDevice(
        wide_code, seed=1, max_bit_count=10
    )
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    payload = first_payload(capped, operation, 1)
    assert len(payload.bits) == 10


def test_a_cap_above_the_syndrome_leaves_its_width():
    code = code_geometry.SurfaceCodeModel(distance=3)
    roomy = syndrome_devices.SyndromeBitDevice(code, seed=1, max_bit_count=100)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    payload = first_payload(roomy, operation, 1)
    assert len(payload.bits) == 8


def test_one_payload_per_patch_when_asked():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(
        code, seed=1, one_payload_per_patch=True
    )
    operation = program_records.Operation(
        id=1, name="merge", qubits=(0, 1), patches=(5, 6)
    )
    payloads = device.round_payloads(operation, 1)
    assert len(payloads) == 2
    assert payloads[0].patch_id == 5
    assert payloads[1].patch_id == 6
    assert payloads[0].operation_id == 1
    assert payloads[1].operation_id == 1


def test_the_timing_only_source_cannot_finalize_a_stream_round():
    operation = program_records.Operation(id=1, name="tail", qubits=(0,))
    timing_only = syndrome_devices.TimingOnlyDevice()
    with pytest.raises(ValueError, match="finalize"):
        timing_only.finalize_stream_round(operation, 3)


def test_the_fake_bit_source_cannot_finalize_a_stream_round():
    operation = program_records.Operation(id=1, name="tail", qubits=(0,))
    code = code_geometry.SurfaceCodeModel(distance=3)
    fake_bits = syndrome_devices.SyndromeBitDevice(code, seed=1)
    with pytest.raises(ValueError, match="finalize"):
        fake_bits.finalize_stream_round(operation, 3)
