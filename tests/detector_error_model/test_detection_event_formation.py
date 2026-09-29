"""The former seated where detection_events.formed_at names.

The seats are placements of one conversion: Google's workstation
(2408.13687 lines 474-476), the decoder chip's input path (Maurer
2510.21600 lines 234-236) and the decoder itself (Caune et al.
2410.05202 lines 1252-1255, LILLIPUT 2108.06569 lines 500-509), and the
values are a parity of the raw outcomes at every seat. A detector
compares a round against the one before it (LILLIPUT lines 499-510), so
each seat forms from the packets it holds. The referent for the values
is Stim's own converter, stim.Circuit.compile_m2d_converter.
"""

import dataclasses

import pytest
import stim

import decsim.config as config
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.records.rounds as round_records

# a distance-3 memory: 8 raw bits a round, 17 on the last; 4 events on
# the first round, 8 in the bulk and 12 on the last
CIRCUIT = stim.Circuit.generated(
    "surface_code:rotated_memory_z", rounds=4, distance=3
)
TABLE = detector_formation.build_formation_table(CIRCUIT, 4)


class CircuitSource:
    """A source whose recipes are one circuit's, for every operation."""

    def __init__(self, table=TABLE):
        self.table = table

    def formation_table(self, operation_id):
        """The one table."""
        del operation_id
        return self.table


class CountingDetector:
    """A burst detector that records every round it is shown."""

    def __init__(self):
        self.observed = []

    def observe_round(self, operation_id, round_index, events):
        """One round's events."""
        self.observed.append((operation_id, round_index, tuple(events)))


FORMER_CLOCK = config.Clock(4000)


def seated(formed_at, latency_cycles=0, cycles_per_round=0):
    settings = event_settings.DetectionEventSettings(
        formed_at=formed_at,
        clock=FORMER_CLOCK,
        latency_cycles=latency_cycles,
        cycles_per_round=cycles_per_round,
    )
    source = CircuitSource()
    return formation.SeatedFormation(source, settings)


def rounds(*round_indices, operation_id=1) -> tuple:
    """One zero fragment of each round, in the order given."""
    fragments = []
    for round_index in round_indices:
        carried = fragment(round_index, operation_id=operation_id)
        fragments.append(carried)
    return tuple(fragments)


def fragment(round_index, bits=None, operation_id=1):
    if bits is None:
        width = TABLE.packet_width_by_round[round_index]
        bits = (0,) * width
    return round_records.RetainedSyndromeFragment(
        operation_id=operation_id,
        patch_ids=(0,),
        round_index=round_index,
        bits=tuple(bits),
        size_bits=len(bits),
        fragment_index=0,
    )


def test_a_seat_not_in_the_list_hands_the_round_on_as_it_came():
    placement = seated(("controller",))
    raw = (fragment(1),)

    leaving = placement.form_at("weak_syndrome_buffer", raw)

    assert leaving is raw
    assert not placement.forms_at("weak_syndrome_buffer")


def test_a_seat_outside_a_decoder_sends_the_events_at_their_own_width():
    placement = seated(("weak_syndrome_buffer",))
    raw = rounds(1)

    (first,) = placement.form_at("weak_syndrome_buffer", raw)

    assert first.bits == (0, 0, 0, 0)
    assert first.size_bits == 4


def test_a_decoder_seat_keeps_the_width_that_landed():
    """The unit's input memory was written the raw round."""
    placement = seated(("weak_decoder",))
    raw = rounds(1)

    (first,) = placement.form_at("weak_decoder", raw)

    assert first.bits == (0, 0, 0, 0)
    assert first.size_bits == 8


def test_a_timing_only_round_is_handed_on_as_it_is():
    """A round with no bits has no outcomes to form events from."""
    placement = seated(("controller",))
    timing_only = round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=None,
        size_bits=8,
        fragment_index=0,
    )

    (kept,) = placement.form_at("controller", (timing_only,))

    assert kept is timing_only


def test_a_source_with_no_recipes_and_no_stated_width_forms_nothing():
    placement = no_recipes_at("controller")
    raw = rounds(1)

    leaving = placement.form_at("controller", raw)

    assert leaving == raw


def stated(bits=None, size_bits=17, event_bits=12):
    """A circuit-less source's last round: its raw and its events' width."""
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=4,
        bits=bits,
        size_bits=size_bits,
        fragment_index=0,
        event_bits=event_bits,
    )


def no_recipes_at(seat):
    settings = event_settings.DetectionEventSettings(formed_at=(seat,))
    return formation.SeatedFormation(None, settings)


def test_a_circuit_less_round_leaves_a_store_seat_at_its_stated_width():
    placement = no_recipes_at("weak_syndrome_buffer")
    raw = (stated(),)

    (formed,) = placement.form_at("weak_syndrome_buffer", raw)

    assert formed.size_bits == 12
    assert formed.event_bits is None


def test_a_circuit_less_round_keeps_its_landed_width_at_a_decoder_seat():
    placement = no_recipes_at("weak_decoder")
    raw = (stated(),)

    leaving = placement.form_at("weak_decoder", raw)

    assert leaving == raw


def timing_only_round():
    """A round with no bits, eight wide on the wire."""
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
        round_index=1,
        bits=None,
        size_bits=8,
        fragment_index=0,
    )


WIDTH_CASES = (
    (seated(("controller",)), "weak_syndrome_buffer", rounds(1), 8),
    (seated(("weak_syndrome_buffer",)), "weak_syndrome_buffer", rounds(1), 4),
    (seated(("weak_decoder",)), "weak_decoder", rounds(1), 8),
    (
        seated(("weak_syndrome_buffer",)),
        "weak_syndrome_buffer",
        (timing_only_round(),),
        8,
    ),
    (
        no_recipes_at("weak_syndrome_buffer"),
        "weak_syndrome_buffer",
        (stated(),),
        12,
    ),
    (no_recipes_at("weak_decoder"), "weak_decoder", (stated(),), 17),
)


@pytest.mark.parametrize(
    ("placement", "seat", "fragments", "width"), WIDTH_CASES
)
def test_the_width_a_seat_answers_is_the_width_it_forms_the_round_to(
    placement, seat, fragments, width
):
    """A store reserves this width before the round lands.

    gem5 makes room for a block at the size its compressor will store it
    at (src/mem/cache/base.cc:1678-1698).
    """
    asked = placement.width_at(seat, fragments)
    formed = placement.form_at(seat, fragments)

    assert asked == width
    assert round_records.fragment_wire_bits(formed) == width


def test_asking_a_width_forms_nothing_and_holds_nothing():
    detector = CountingDetector()
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_syndrome_buffer",)
    )
    source = CircuitSource()
    placement = formation.SeatedFormation(
        source, settings, "weak_syndrome_buffer", detector
    )
    second_round = rounds(2)

    placement.width_at("weak_syndrome_buffer", second_round)

    assert detector.observed == []
    assert placement.needs_the_round_before("weak_syndrome_buffer", 1, 2)


def test_fake_bits_formed_are_the_first_of_the_random_bits():
    """Random bits carry no parity, so the events are their first bits."""
    placement = no_recipes_at("controller")
    random_bits = (1, 0) * 8 + (1,)
    raw = (stated(bits=random_bits),)

    (formed,) = placement.form_at("controller", raw)

    assert formed.bits == random_bits[:12]


def test_a_round_the_seat_formed_before_is_answered_from_what_it_holds():
    """Two windows of one tier read one round; it is formed once."""
    detector = CountingDetector()
    settings = event_settings.DetectionEventSettings(
        formed_at=("weak_decoder",)
    )
    source = CircuitSource()
    placement = formation.SeatedFormation(
        source, settings, "weak_decoder", detector
    )
    first_job = rounds(1, 2)
    second_job = rounds(2, 3)

    placement.form_at("weak_decoder", first_job)
    placement.form_at("weak_decoder", second_job)

    observed_rounds = [round_index for _, round_index, _ in detector.observed]
    assert observed_rounds == [1, 2, 3]


def test_each_seat_forms_from_the_packets_it_was_given():
    """The strong decoder never saw round 1, so it cannot form round 2."""
    placement = seated(("weak_decoder", "strong_decoder"))
    first_two = rounds(1, 2)
    second = rounds(2)
    placement.form_at("weak_decoder", first_two)

    with pytest.raises(RuntimeError) as refusal:
        placement.form_at("strong_decoder", second)

    sentence = str(refusal.value)
    assert "round 2 reads round 1, which this former does not hold" in sentence
    assert "LILLIPUT 2108.06569 lines 499-510" in sentence


def test_a_seat_given_the_round_before_forms_the_round_after_it():
    """The round before is held, not formed and not handed on."""
    placement = seated(("strong_decoder",))
    third = rounds(3)
    second = rounds(2)

    formed = placement.form_at("strong_decoder", third, second)

    assert [carried.round_index for carried in formed] == [3]
    assert formed[0].bits == (0,) * 8


def test_a_seat_that_has_not_formed_a_round_needs_the_one_before():
    placement = seated(("weak_decoder",))

    assert placement.needs_the_round_before("weak_decoder", 1, 3)


def test_a_seat_needs_nothing_before_a_round_it_formed():
    placement = seated(("weak_decoder",))
    first_two = rounds(1, 2)
    placement.form_at("weak_decoder", first_two)

    assert not placement.needs_the_round_before("weak_decoder", 1, 2)


def test_an_operations_first_round_needs_nothing_before_it():
    """It compares against the reset (LILLIPUT 2108.06569 lines 499-510)."""
    placement = seated(("weak_decoder",))

    assert not placement.needs_the_round_before("weak_decoder", 1, 1)


def test_a_seat_that_does_not_form_needs_nothing():
    placement = seated(("controller",))

    assert not placement.needs_the_round_before("weak_decoder", 1, 3)


def test_a_seat_reports_the_raw_packets_its_recipes_still_read():
    """max_record_span + 1 packets: two rounds of eight raw bits here.

    Maurer 2510.21600 Algorithm 2 (lines 760-770) keeps the running
    syndrome a detector compares against; the seat holds that and no
    more.
    """
    placement = seated(("weak_decoder",))
    reported = []

    def state_held(seat, operation_id, bits) -> None:
        del seat, operation_id
        reported.append(bits)

    placement.trace.state_held.connect(state_held)
    first_three = rounds(1, 2, 3)

    placement.form_at("weak_decoder", first_three)

    assert TABLE.max_record_span == 1
    assert reported == [8, 16, 16]


def test_each_operation_is_formed_from_its_own_first_round():
    placement = seated(("controller",))
    first = rounds(1)
    second = rounds(2)
    other_first = rounds(1, operation_id=2)
    placement.form_at("controller", first)
    placement.form_at("controller", second)

    (other,) = placement.form_at("controller", other_first)

    assert other.bits == (0, 0, 0, 0)


@pytest.mark.parametrize("seat", ["controller", "weak_syndrome_buffer"])
def test_a_joint_round_is_formed_once_in_measurement_order(seat):
    placement = seated((seat,))
    first = fragment(1, bits=(1, 0, 0))
    middle = fragment(1, bits=(0, 1, 0))
    middle = dataclasses.replace(middle, patch_ids=("other",), fragment_index=1)
    last = fragment(1, bits=(0, 0))
    last = dataclasses.replace(last, fragment_index=2)

    (formed,) = placement.form_at(seat, (last, first, middle))

    assert formed.patch_ids == (0, "other")
    assert len(formed.bits) == 4
    assert formed.size_bits == 4


def test_forming_rounds_together_costs_the_latency_once_and_the_rate_after():
    """Yang 2605.04892 lines 1273-1275: a pipelined stage, 5 then 1 a round."""
    placement = seated(("weak_decoder",), latency_cycles=5, cycles_per_round=1)

    assert placement.cycles_at("weak_decoder", 1) == 5
    assert placement.cycles_at("weak_decoder", 4) == 8
    assert placement.cycles_at("weak_decoder", 0) == 0
    assert placement.cycles_at("controller", 4) == 0


@pytest.mark.parametrize("seat", event_settings.SEATS)
def test_every_seat_forms_stims_events_on_sampled_shots(seat):
    """Each round split into three shuffled fragments, at every seat.

    The first round compares against the reset and the last folds in
    the data readout, so a four-round memory covers all three layers.
    The rounds come as two windows out of order, the later first, and
    overlapping, so a seat forms a round from the round before it that
    it is given and answers a round it formed from what it remembers.
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

    formed = _formed_at(seat, circuit, measurements)

    assert formed == _as_rows(expected)
    assert any(any(row) for row in formed)


def _formed_at(seat, circuit, measurements) -> list:
    table = detector_formation.build_formation_table(circuit, 4)
    settings = event_settings.DetectionEventSettings(formed_at=(seat,))
    rows = []
    for shot in measurements:
        source = CircuitSource(table)
        placement = formation.SeatedFormation(source, settings)
        packets = detector_formation.split_measurements_into_packets(
            table, shot
        )
        row = _one_shot_formed_at(placement, seat, packets)
        rows.append(row)
    return rows


def _one_shot_formed_at(placement, seat, packets) -> tuple:
    """One shot's events, its rounds given as two windows out of order."""
    events_by_round = {}
    for window in ((3, 4), (1, 2, 3)):
        fragments = _window_fragments(window, packets)
        before = _round_before(placement, seat, window[0], packets)
        formed = placement.form_at(seat, fragments, before)
        for carried in formed:
            events_by_round[carried.round_index] = carried.bits
    row = []
    for round_index in (1, 2, 3, 4):
        row.extend(events_by_round[round_index])
    return tuple(row)


def _round_before(placement, seat, first_round, packets) -> tuple:
    """The raw round before first_round, when the seat needs it."""
    if not placement.needs_the_round_before(seat, 1, first_round):
        return ()
    round_before = first_round - 1
    return _in_three_shuffled_fragments(round_before, packets[round_before])


def _window_fragments(window, packets) -> tuple:
    fragments = []
    for round_index in window:
        shuffled = _in_three_shuffled_fragments(
            round_index, packets[round_index]
        )
        fragments.extend(shuffled)
    return tuple(fragments)


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
