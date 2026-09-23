"""Where a round's outcomes become events: the three rows and the former.

The rows are placements of one conversion: Google's workstation
(2408.13687 lines 474-476), the decoder's own chip (Maurer 2510.21600
lines 235-237) ahead of its store, and the decoder side (Caune et al.
2410.05202 lines 1252-1256, LILLIPUT 2108.06569 lines 499-510), and the
values are a parity of the raw outcomes in every row. The former's
order law is LILLIPUT's: a detector compares a round against the one
before it, so the rounds of an operation are formed in order. The
referent for the values is Stim's own converter,
stim.Circuit.compile_m2d_converter.
"""

import dataclasses
from collections.abc import Callable

import pytest
import stim

import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.ports as ports
import decsim.records.rounds as round_records


class DeviceTable:
    """A formation table: the round's events are its bits, ones first."""

    def __init__(self):
        self.asked = []

    def form_round(self, operation_id, round_index, raw_bits):
        """One round's detection events."""
        self.asked.append((operation_id, round_index, tuple(raw_bits)))
        return (round_index % 2, 1, 0)


def fragment(round_index, bits=(1, 0, 1, 1)):
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=round_index,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
    )


def test_the_controller_row_sends_the_events_at_their_own_width():
    table = DeviceTable()
    row = formation.ControllerSideFormation(table, 0)
    raw = fragment(1)

    (leaving,) = row.form_before_departure((raw,))

    assert leaving.bits == (1, 1, 0)
    assert leaving.size_bits == 3


def test_the_decoder_row_sends_the_raw_outcomes_at_their_own_width():
    table = DeviceTable()
    row = formation.DecoderSideFormation(table, 0)
    raw = fragment(1)

    (leaving,) = row.form_before_departure((raw,))

    assert leaving.bits == (1, 0, 1, 1)
    assert leaving.size_bits == 4


def test_the_weak_syndrome_buffer_row_sends_raw_and_stores_the_events():
    table = DeviceTable()
    row = formation.WeakSyndromeBufferSideFormation(table, 0)
    raw = fragment(1)

    (leaving,) = row.form_before_departure((raw,))
    (stored,) = row.form_before_storage((leaving,))

    assert leaving.bits == (1, 0, 1, 1)
    assert stored.bits == (1, 1, 0)
    assert stored.size_bits == 3


def test_the_controller_and_decoder_rows_store_the_round_as_it_landed():
    table = DeviceTable()
    at_the_controller = formation.ControllerSideFormation(table, 0)
    at_the_decoder = formation.DecoderSideFormation(table, 0)
    landed = fragment(1)

    (from_the_controller,) = at_the_controller.form_before_storage((landed,))
    (for_the_decoder,) = at_the_decoder.form_before_storage((landed,))

    assert from_the_controller is landed
    assert for_the_decoder is landed
    assert table.asked == []


def test_only_the_decoder_row_hands_a_former_to_the_decoder_side():
    table = DeviceTable()

    at_the_controller = formation.ControllerSideFormation(table, 0)
    at_the_decoder = formation.DecoderSideFormation(table, 0)
    at_the_weak_syndrome_buffer = formation.WeakSyndromeBufferSideFormation(
        table, 0
    )

    assert at_the_controller.decoder_side_former() is None
    assert at_the_weak_syndrome_buffer.decoder_side_former() is None
    assert at_the_decoder.decoder_side_former() is not None


def test_a_source_with_no_formation_table_forms_nothing_either_way():
    at_the_controller = formation.ControllerSideFormation(None, 0)
    at_the_decoder = formation.DecoderSideFormation(None, 0)
    raw = fragment(1)

    (kept,) = at_the_controller.form_before_departure((raw,))

    assert kept.bits == (1, 0, 1, 1)
    assert at_the_decoder.decoder_side_former() is None


def test_a_timing_only_round_is_handed_on_as_it_is():
    """A round with no bits has no outcomes to form events from."""
    table = DeviceTable()
    row = formation.ControllerSideFormation(table, 0)
    timing_only = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=None,
        size_bits=12,
        fragment_index=0,
    )

    (kept,) = row.form_before_departure((timing_only,))

    assert kept is timing_only


def test_a_controller_charge_under_the_weak_syndrome_buffer_row_is_refused():
    """The controller cannot be charged for work the chip does."""
    table = DeviceTable()

    with pytest.raises(ValueError) as refusal:
        formation.WeakSyndromeBufferSideFormation(table, 250)

    sentence = str(refusal.value)
    assert "controller.detection_event_cycles_per_round" in sentence
    assert "weak_syndrome_buffer.detection_event_cycles_per_round" in sentence


def test_a_controller_charge_under_the_decoder_row_is_refused():
    """The controller cannot be charged for work it does not do."""
    table = DeviceTable()

    with pytest.raises(ValueError) as refusal:
        formation.DecoderSideFormation(table, 250)

    sentence = str(refusal.value)
    assert "controller.detection_event_cycles_per_round" in sentence
    assert "detection_event_latency_cycles, or write null" in sentence


def test_the_former_forms_each_round_once_and_remembers_it():
    """Both tiers read one round's events; the table is fed once."""
    table = DeviceTable()
    former = formation.RememberedDetectionEvents(table)

    first = former.form_round(1, 1, (1, 0, 1, 1))
    second = former.form_round(1, 1, (1, 0, 1, 1))

    assert first == (1, 1, 0)
    assert second == (1, 1, 0)
    assert table.asked == [(1, 1, (1, 0, 1, 1))]


def test_the_rounds_of_an_operation_are_formed_in_order():
    table = DeviceTable()
    former = formation.RememberedDetectionEvents(table)

    former.form_round(1, 1, (1, 0, 1, 1))
    former.form_round(1, 2, (0, 0, 1, 1))

    assert table.asked == [
        (1, 1, (1, 0, 1, 1)),
        (1, 2, (0, 0, 1, 1)),
    ]


def test_a_round_asked_for_before_the_one_before_it_is_refused():
    """The table no longer holds the packet that round is compared against."""
    table = DeviceTable()
    former = formation.RememberedDetectionEvents(table)
    former.form_round(1, 1, (1, 0, 1, 1))

    with pytest.raises(RuntimeError) as refusal:
        former.form_round(1, 3, (0, 1, 1, 0))

    sentence = str(refusal.value)
    assert "is at round 1 and was asked for round 3" in sentence
    assert "LILLIPUT 2108.06569 lines 499-510" in sentence


def test_each_operation_is_formed_from_its_own_first_round():
    table = DeviceTable()
    former = formation.RememberedDetectionEvents(table)
    former.form_round(1, 1, (1, 0, 1, 1))
    former.form_round(1, 2, (1, 0, 1, 1))

    events = former.form_round(2, 1, (0, 1, 0, 1))

    assert events == (1, 1, 0)


@pytest.mark.parametrize(
    "placement",
    [
        formation.ControllerSideFormation,
        formation.WeakSyndromeBufferSideFormation,
    ],
)
def test_joint_round_is_formed_once_in_measurement_order(
    placement: Callable[..., ports.DetectionEventPlacement],
) -> None:
    table = DeviceTable()
    row = placement(table, 0)
    first = fragment(1, bits=(1,))
    middle = fragment(1, bits=(0,))
    middle = dataclasses.replace(middle, patch_ids=("other",), fragment_index=1)
    last = fragment(1, bits=(1,))
    last = dataclasses.replace(last, fragment_index=2)

    leaving = row.form_before_departure((last, first, middle))
    (formed,) = row.form_before_storage(leaving)

    assert table.asked == [(1, 1, (1, 0, 1))]
    assert formed.patch_ids == (0, "other")
    assert formed.bits == (1, 1, 0)
    assert formed.size_bits == 3


def test_rounds_split_into_fragments_form_stims_events_on_sampled_shots():
    """The controller row, fed each round as three shuffled fragments.

    The first round compares against the reset and the last folds in
    the data readout, so a four-round memory covers all three layers.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=4,
        distance=3,
        before_measure_flip_probability=0.05,
        after_clifford_depolarization=0.05,
    )
    sampler = circuit.compile_sampler(seed=11)
    measurements = sampler.sample(32)
    converter = circuit.compile_m2d_converter()
    expected = converter.convert(
        measurements=measurements, append_observables=False
    )

    formed = _formed_at_the_controller(circuit, 4, measurements)

    assert formed == _as_rows(expected)
    assert any(any(row) for row in formed)


class _StimFormer:
    """StimDevice.form_round over one shot's formation table."""

    def __init__(self, table: detector_formation.FormationTable) -> None:
        self.former = detector_formation.StreamingDetectorFormer(table)

    def form_round(self, operation_id, round_index, raw_bits):
        del operation_id
        events, _ = self.former.feed_packet(round_index, raw_bits)
        return tuple(value for _, value in events)


def _formed_at_the_controller(circuit, round_count, measurements) -> list:
    table = detector_formation.build_formation_table(circuit, round_count)
    rows = []
    after_last_round = round_count + 1
    for shot in measurements:
        former = _StimFormer(table)
        row = formation.ControllerSideFormation(former, 0)
        packets = detector_formation.split_measurements_into_packets(
            table, shot
        )
        events = []
        for round_index in range(1, after_last_round):
            fragments = _in_three_shuffled_fragments(
                round_index, packets[round_index]
            )
            (leaving,) = row.form_before_departure(fragments)
            events.extend(leaving.bits)
        rows.append(tuple(events))
    return rows


def _in_three_shuffled_fragments(round_index, packet) -> tuple:
    third = len(packet) // 3
    edges = (0, third, 2 * third, len(packet))
    fragments = []
    for index in range(3):
        start = edges[index]
        end = edges[index + 1]
        bits = tuple(packet[start:end])
        part = fragment(round_index, bits=bits)
        part = dataclasses.replace(part, fragment_index=index)
        fragments.append(part)
    return (fragments[2], fragments[0], fragments[1])


def _as_rows(events) -> list:
    rows = []
    for shot in events:
        bits = tuple(int(bit) for bit in shot)
        rows.append(bits)
    return rows
