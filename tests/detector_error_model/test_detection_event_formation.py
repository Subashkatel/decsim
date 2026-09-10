"""Where a round's outcomes become events: the two rows and the former.

The rows are the two published placements of one conversion, Google's
workstation (2408.13687 lines 474-476) against the decoder side (Caune
et al. 2410.05202 lines 1252-1256, LILLIPUT 2108.06569 lines 499-510),
and the values are a parity of the raw outcomes either way. The former's
order law is LILLIPUT's: a detector compares a round against the one
before it, so the rounds of an operation are formed in order.
"""

import pytest

import decsim.detector_error_model.detection_event_formation as formation
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
        patch_id=0,
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


def test_only_the_decoder_row_hands_a_former_to_the_decoder_side():
    table = DeviceTable()

    at_the_controller = formation.ControllerSideFormation(table, 0)
    at_the_decoder = formation.DecoderSideFormation(table, 0)

    assert at_the_controller.decoder_side_former() is None
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
        patch_id=0,
        round_index=1,
        bits=None,
        size_bits=12,
        fragment_index=0,
    )

    (kept,) = row.form_before_departure((timing_only,))

    assert kept is timing_only


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
