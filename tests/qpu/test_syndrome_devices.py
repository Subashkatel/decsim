"""The circuit-less syndrome sources shape their payloads by the code card.

Sources: the SyndromeSource port in decsim/ports.py; the rotated
surface code's d*d - 1 stabilizers per round (Stim's generated circuit,
src/stim/gen/gen_surface_code.cc; Barber et al. 2309.05558 lines
947-951). A timing-only round states that size and no values, the way
gem5's packet trace records a size and no data (gem5
src/proto/packet.proto).
"""

import dataclasses
import random

import pytest
import stim

import decsim.detector_error_model.detector_formation as detector_formation
import decsim.ports as ports
import decsim.qpu.code_geometry as code_geometry
import decsim.qpu.syndrome_devices as syndrome_devices
import decsim.records.program as program_records
import decsim.records.rounds as round_records
import decsim.seeding as seeding


def first_payload(device, operation, round_index):
    payloads = device.round_payloads(operation, round_index)
    return payloads[0]


def test_a_timing_only_round_states_the_codes_width_and_no_values():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(
        id=4, name="memory", qubits=(2,), patches=(7,)
    )
    payload = first_payload(device, operation, 3)
    assert payload == round_records.QPUReadout(
        4, (7,), 3, size_bits=8, event_bits=8
    )
    assert payload.bits is None


def test_a_timing_only_round_of_two_patches_is_two_patches_wide():
    code = code_geometry.SurfaceCodeModel(distance=5)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(
        id=4, name="merge", qubits=(2, 3), patches=(7, 9)
    )
    payload = first_payload(device, operation, 1)
    assert payload.size_bits == 48


def test_a_timing_only_idle_round_is_one_patch_wide():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(id=4, name="idle", qubits=(2,))
    (payload,) = device.idle_round_payloads(
        operation, "s", 5, is_final=False, round_period_ticks=1
    )
    assert payload == round_records.QPUReadout(
        "s", (2,), 5, size_bits=8, event_bits=8
    )


def test_a_stream_segment_reports_its_stream_and_global_round():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(
        id=4, name="tail", qubits=(2,), stream_id="s", stream_offset=6
    )
    payload = first_payload(device, operation, 2)
    assert payload == round_records.QPUReadout(
        "s", (2,), 8, size_bits=8, event_bits=8
    )


def test_fake_bits_grow_with_the_codes_distance():
    """A d = 5 patch reads 24 stabilizers a round, so 24 bits."""
    wide_code = code_geometry.SurfaceCodeModel(distance=5)
    device = syndrome_devices.SyndromeBitDevice(wide_code, seed=1)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    payload = first_payload(device, operation, 1)
    assert len(payload.bits) == 24
    assert payload.size_bits == 24


def test_a_rounds_fake_bits_do_not_depend_on_the_rounds_drawn_before_it():
    """Another operation's rounds, drawn first, leave this round's bits.

    Which operation's round the clock draws first is decided by other
    components' timing; the Stim source keeps each stream's shot under
    its own substream for the same reason (seeding.substream_seed).
    """
    code = code_geometry.SurfaceCodeModel(distance=3)
    first = program_records.Operation(id=1, name="memory", qubits=(0,))
    second = program_records.Operation(id=2, name="memory", qubits=(1,))
    alone = syndrome_devices.SyndromeBitDevice(code, seed=1)
    after_another = syndrome_devices.SyndromeBitDevice(code, seed=1)
    after_another.round_payloads(second, 1)
    payload_alone = first_payload(alone, first, 1)
    payload_after_another = first_payload(after_another, first, 1)
    assert payload_after_another.bits == payload_alone.bits


def test_a_fake_bit_payload_draws_under_its_stream_round_and_patches():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    operation = program_records.Operation(
        id=4, name="memory", qubits=(0,), patches=(7,)
    )
    segment = dataclasses.replace(operation, stream_id="s", stream_offset=2)
    payload_seed = seeding.substream_seed(1, ("s", 5, 7))
    generator = random.Random(payload_seed)
    oracle_bits = [generator.randint(0, 1) for _ in range(8)]
    payload = first_payload(device, segment, 3)
    assert payload.bits == oracle_bits


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
    assert payloads[0].patch_ids == (5,)
    assert payloads[1].patch_ids == (6,)
    assert payloads[0].operation_id == 1
    assert payloads[1].operation_id == 1


def last_syndrome_fragment() -> program_records.Operation:
    """Round 5 of stream "s" split in two: the checks first, of two slots."""
    return program_records.Operation(
        id=8,
        name="last syndrome",
        qubits=(2,),
        patches=(7,),
        stream_id="s",
        stream_offset=4,
        syndrome_fragment_index=0,
        syndrome_fragment_count=2,
    )


def data_readout_fragment() -> program_records.Operation:
    """The terminal fragment that finalizes round 5 of stream "s"."""
    return program_records.Operation(
        id=9,
        name="data readout",
        qubits=(2,),
        patches=(7,),
        stream_id="s",
        stream_offset=4,
        finalizes_stream_round=True,
        syndrome_fragment_index=1,
        syndrome_fragment_count=2,
    )


def test_a_last_round_split_in_two_keeps_the_readout_for_its_fragment():
    """The checks fragment is 8 bits; the finalizer brings the 9 data bits."""
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    checks = last_syndrome_fragment()
    device.begin_operation(checks, 1, 5, round_period_ticks=1)

    payload = first_payload(device, checks, 1)

    assert payload == round_records.QPUReadout(
        "s", (7,), 5, size_bits=8, event_bits=8
    )


def test_the_timing_only_finalizer_emits_the_data_readout_as_a_fragment():
    """9 data qubits raw, closing the last 4 of the round's 12 events."""
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    readout = data_readout_fragment()

    (payload,) = device.finalize_stream_round(readout, 5)

    assert payload == round_records.QPUReadout(
        "s", (7,), 5, size_bits=9, event_bits=4
    )


def test_a_fake_bit_last_round_split_in_two_draws_the_checks_alone():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    checks = last_syndrome_fragment()
    device.begin_operation(checks, 1, 5, round_period_ticks=1)

    payload = first_payload(device, checks, 1)

    assert len(payload.bits) == 8
    assert payload.event_bits == 8


def test_the_fake_bit_finalizer_draws_the_data_readout():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    readout = data_readout_fragment()

    (payload,) = device.finalize_stream_round(readout, 5)

    assert len(payload.bits) == 9
    assert payload.round_index == 5
    assert payload.event_bits == 4


def test_the_fake_bit_device_names_its_code_card_as_its_seed_child():
    """The card shapes every payload, so the seed walk reaches it here."""
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    children = device.run_seed_children()
    assert len(children) == 1
    assert children[0].child is code


def test_a_circuit_less_source_names_a_model_source_that_builds_nothing():
    """The model port is its own fact: no circuit, no model, and no methods.

    sinter derives a decoder's model from the circuit only when one is
    given (sinter/_data/_task.py lines 71 and 87-89); a source with no
    circuit names the shared component that answers with nothing.
    """
    code = code_geometry.SurfaceCodeModel(distance=3)
    source = syndrome_devices.TimingOnlyDevice(code)

    models = source.window_model_source()

    assert models is syndrome_devices.NO_WINDOW_MODELS
    assert isinstance(models, ports.WindowModelSource)
    assert not isinstance(source, ports.WindowModelSource)


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("round_count", [1, 2, 4])
def test_every_rounds_widths_are_those_of_stims_memory(distance, round_count):
    """Raw and event widths per round, as Stim's own memory lays them out.

    The referent is the formation table of stim.Circuit.generated's
    rotated memory: its packet widths are the raw outcomes a round reads
    out, and its detectors per round the events a seat forms.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=round_count,
        distance=distance,
    )
    table = detector_formation.build_formation_table(circuit, round_count)
    code = code_geometry.SurfaceCodeModel(distance=distance)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    device.begin_operation(
        operation, round_count, round_count, round_period_ticks=1
    )

    stated = _stated_widths(device, operation, round_count)

    assert stated == _stims_widths(table)


def test_a_final_idle_round_reads_out_the_data_qubits():
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.TimingOnlyDevice(code)
    operation = program_records.Operation(id=4, name="idle", qubits=(2,))

    (payload,) = device.idle_round_payloads(
        operation, "s", 5, is_final=True, round_period_ticks=1
    )

    assert payload.size_bits == 17
    assert payload.event_bits == 12


def test_fake_bits_are_as_wide_as_the_raw_round_they_stand_for():
    """The last round's 8 checks and 9 data qubits: 17 random bits."""
    code = code_geometry.SurfaceCodeModel(distance=3)
    device = syndrome_devices.SyndromeBitDevice(code, seed=1)
    operation = program_records.Operation(id=1, name="memory", qubits=(0,))
    device.begin_operation(operation, 2, 2, round_period_ticks=1)

    payload = first_payload(device, operation, 2)

    assert len(payload.bits) == 17
    assert payload.event_bits == 12


def _stated_widths(device, operation, round_count) -> list:
    """Each round's (raw, event) widths as the device states them."""
    widths = []
    after_last_round = round_count + 1
    for round_index in range(1, after_last_round):
        payload = first_payload(device, operation, round_index)
        widths.append((payload.size_bits, payload.event_bits))
    return widths


def _stims_widths(table) -> list:
    """Each round's (raw, event) widths off Stim's formation table."""
    widths = []
    after_last_round = table.round_count + 1
    for round_index in range(1, after_last_round):
        raw_bits = table.packet_width_by_round[round_index]
        detectors = table.detectors_of_round(round_index)
        widths.append((raw_bits, len(detectors)))
    return widths
