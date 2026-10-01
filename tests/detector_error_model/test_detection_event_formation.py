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


def lookback_circuit(error_probability=0.0) -> tuple:
    """One qubit measured ten rounds; round r's detector is rec[-1] ^ rec[-3].

    Rounds 1 and 2 compare against the reset. Returns the circuit and its
    packet schedule, one measurement a round.
    """
    noisy_measurement = (
        f"X_ERROR({error_probability}) 0\nM({error_probability}) 0\n"
    )
    circuit = stim.Circuit(
        "R 0\n"
        f"{noisy_measurement}DETECTOR rec[-1]\n"
        f"{noisy_measurement}DETECTOR rec[-1]\n"
        "REPEAT 8 {\n"
        f"{noisy_measurement}DETECTOR rec[-1] rec[-3]\n"
        "}\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    return circuit, measurement_rounds


def lookback_seated(formed_at):
    circuit, measurement_rounds = lookback_circuit()
    table = detector_formation.build_formation_table(
        circuit, 10, measurement_rounds=measurement_rounds
    )
    settings = event_settings.DetectionEventSettings(formed_at=formed_at)
    source = CircuitSource(table)
    return formation.SeatedFormation(source, settings)


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
    needed_rounds = placement.rounds_needed_before(
        "weak_syndrome_buffer", 1, 2, 2
    )
    assert needed_rounds == (1,)


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


def test_a_seat_that_has_not_formed_a_surface_code_round_needs_one_before():
    """A bulk detector compares a round against the one before it."""
    placement = seated(("weak_decoder",))

    assert placement.rounds_needed_before("weak_decoder", 1, 3, 3) == (2,)


def test_a_seat_is_given_the_round_its_detector_reads_and_none_between():
    """Round 5's detector is rec[-1] XOR rec[-3], rounds 5 and 3."""
    placement = lookback_seated(("strong_decoder",))

    assert placement.rounds_needed_before("strong_decoder", 1, 5, 5) == (3,)


def reach_growing_circuit(error_probability=0.0) -> tuple:
    """Ten rounds of rec[-1], then rec[-1] ^ rec[-4] from round 4 on.

    A read from round 3 reads only round 3 for its first round, while
    its round 4 reads round 1. Returns the circuit and its packet
    schedule, one measurement a round.
    """
    noisy_measurement = (
        f"X_ERROR({error_probability}) 0\nM({error_probability}) 0\n"
    )
    circuit = stim.Circuit(
        "R 0\n"
        "REPEAT 3 {\n"
        f"{noisy_measurement}DETECTOR rec[-1]\n"
        "}\n"
        "REPEAT 7 {\n"
        f"{noisy_measurement}DETECTOR rec[-1] rec[-4]\n"
        "}\n"
        "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    measurement_rounds = {index: index + 1 for index in range(10)}
    return circuit, measurement_rounds


def reach_growing_seated(formed_at):
    circuit, measurement_rounds = reach_growing_circuit()
    table = detector_formation.build_formation_table(
        circuit, 10, measurement_rounds=measurement_rounds
    )
    settings = event_settings.DetectionEventSettings(formed_at=formed_at)
    source = CircuitSource(table)
    return formation.SeatedFormation(source, settings)


def test_a_seat_is_given_the_round_a_later_round_of_its_read_reads():
    """Round 3 reads only itself; round 4 of the same read reads round 1."""
    placement = reach_growing_seated(("strong_decoder",))

    assert placement.rounds_needed_before("strong_decoder", 1, 3, 4) == (1,)


def test_a_seat_is_not_given_a_round_it_formed():
    """It formed round 2, so round 3's detectors read the packet it kept."""
    placement = seated(("weak_decoder",))
    first_two = rounds(1, 2)
    placement.form_at("weak_decoder", first_two)

    assert placement.rounds_needed_before("weak_decoder", 1, 3, 3) == ()


def test_a_seat_is_not_given_a_round_it_was_given_raw_before():
    """Round 3 read round 1 and was given rounds 1 and 2; round 4 reads 2.

    The seat keeps round 2's packet while round 4 is unformed there, so
    a read from round 4 brings nothing before it.
    """
    placement = lookback_seated(("strong_decoder",))
    third = (fragment(3, bits=(0,)),)
    first_two = (fragment(1, bits=(0,)), fragment(2, bits=(0,)))
    placement.form_at("strong_decoder", third, first_two)

    assert placement.rounds_needed_before("strong_decoder", 1, 4, 4) == ()


def test_a_round_whose_detectors_read_only_itself_needs_nothing_before():
    """Round 2's detector compares against the reset: rec[-1] alone."""
    placement = lookback_seated(("strong_decoder",))

    assert placement.rounds_needed_before("strong_decoder", 1, 2, 2) == ()


def test_a_seat_given_fewer_rounds_than_it_needs_refuses_the_round():
    placement = lookback_seated(("strong_decoder",))
    fifth = (fragment(5, bits=(0,)),)
    fourth = (fragment(4, bits=(0,)),)

    with pytest.raises(RuntimeError, match="round 5 reads round 3"):
        placement.form_at("strong_decoder", fifth, fourth)


def test_a_seat_needs_nothing_before_a_round_it_formed():
    placement = seated(("weak_decoder",))
    first_two = rounds(1, 2)
    placement.form_at("weak_decoder", first_two)

    assert placement.rounds_needed_before("weak_decoder", 1, 2, 2) == ()


def test_an_operations_first_round_needs_nothing_before_it():
    """It compares against the reset (LILLIPUT 2108.06569 lines 499-510)."""
    placement = seated(("weak_decoder",))

    assert placement.rounds_needed_before("weak_decoder", 1, 1, 1) == ()


def test_a_seat_that_does_not_form_needs_nothing():
    placement = seated(("controller",))

    assert placement.rounds_needed_before("weak_decoder", 1, 3, 3) == ()


def test_a_source_with_no_recipes_needs_nothing_before_a_round():
    placement = no_recipes_at("weak_decoder")

    assert placement.rounds_needed_before("weak_decoder", 1, 3, 3) == ()


def test_a_seat_reports_the_raw_packets_its_recipes_still_read():
    """Two packets of eight raw bits: a round and the one it compares to.

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

    assert reported == [8, 16, 16]


def test_a_round_that_left_the_store_lets_its_seat_go_of_what_it_read():
    """Round 2 formed at another seat: no read forms it here after."""
    placement = seated(("weak_decoder", "strong_decoder"))
    first = rounds(1)
    placement.form_at("strong_decoder", first)

    placement.retire_round((1, 2))

    history = placement.history_by_seat["strong_decoder"]
    assert history.former_by_operation[1].packets == {}


def test_a_seat_that_joins_late_knows_the_rounds_that_left_before_it():
    """Round 1 is read by rounds 2 and 3; round 2 left before this seat formed.

    The strong seat forms round 3 from round 1 given raw, and round 2's
    read of round 1 was done already, so it holds nothing after.
    """
    placement = _twice_read_seated(("strong_decoder",))
    placement.retire_round((1, 2))
    third = (fragment(3, bits=(0,)),)
    first = (fragment(1, bits=(0,)),)

    placement.form_at("strong_decoder", third, first)

    history = placement.history_by_seat["strong_decoder"]
    assert history.former_by_operation[1].packets == {}


def test_a_seat_keeps_nothing_of_the_rounds_that_left_the_store():
    """Rounds retire out of order; the watermark closes over all three."""
    placement = seated(("weak_decoder",))
    three_rounds = rounds(1, 2, 3)
    placement.form_at("weak_decoder", three_rounds)

    placement.retire_round((1, 1))
    placement.retire_round((1, 3))
    placement.retire_round((1, 2))

    history = placement.history_by_seat["weak_decoder"]
    done_rounds = history.done_by_operation[1]
    assert history.events_by_round == {}
    assert done_rounds.through == 3
    assert done_rounds.above == set()


def test_a_seat_still_holding_a_raw_round_at_the_end_is_named():
    """Round 1 waits for round 2, which never formed here nor left a store."""
    placement = seated(("strong_decoder",))
    first = rounds(1)
    placement.form_at("strong_decoder", first)

    with pytest.raises(RuntimeError, match="strong_decoder seat still holds"):
        placement.check_settled()


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


def _noisy_surface_code() -> tuple:
    """Four rounds, the three layers: reset, bulk, folded data readout."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=4,
        distance=3,
        before_measure_flip_probability=0.05,
        after_clifford_depolarization=0.05,
    )
    return circuit, 4, None


def _noisy_lookback_memory() -> tuple:
    """Ten rounds whose detector reads two rounds back: rec[-1] ^ rec[-3]."""
    circuit, measurement_rounds = lookback_circuit(0.1)
    return circuit, 10, measurement_rounds


def _noisy_reach_growing_memory() -> tuple:
    """Ten rounds whose later rounds reach further back than round 3."""
    circuit, measurement_rounds = reach_growing_circuit(0.1)
    return circuit, 10, measurement_rounds


def _noisy_color_code(distance, round_count) -> tuple:
    """Stim's XYZ color code memory, whose detectors read two rounds back."""
    circuit = stim.Circuit.generated(
        "color_code:memory_xyz",
        rounds=round_count,
        distance=distance,
        before_measure_flip_probability=0.05,
        after_clifford_depolarization=0.05,
    )
    return circuit, round_count, None


FORMATION_CIRCUITS = {
    "surface_code_d3": _noisy_surface_code(),
    "lookback_two_rounds": _noisy_lookback_memory(),
    "color_code_d3": _noisy_color_code(3, 6),
    "color_code_d5": _noisy_color_code(5, 8),
    "reach_growing": _noisy_reach_growing_memory(),
}


@pytest.mark.parametrize("circuit_name", sorted(FORMATION_CIRCUITS))
@pytest.mark.parametrize("seat", event_settings.SEATS)
def test_every_seat_forms_stims_events_on_sampled_shots(seat, circuit_name):
    """Each round split into three shuffled fragments, at every seat.

    The rounds come as two windows out of order, the later first, and
    overlapping, so a seat forms a round from the raw rounds before it
    that it is given, as many as rounds_needed_before says, and answers
    a round it formed from what it remembers. The circuits reach one
    round back (the surface code), two (the lookback memory and Stim's
    color code) and, from round 4 of a read starting at round 3, three
    (the reach-growing memory), so the rounds the whole read reaches,
    not a fixed count, are what the seat needs.
    """
    circuit, round_count, measurement_rounds = FORMATION_CIRCUITS[circuit_name]
    table = detector_formation.build_formation_table(
        circuit, round_count, measurement_rounds=measurement_rounds
    )
    sampler = circuit.compile_sampler(seed=11)
    measurements = sampler.sample(32)
    converter = circuit.compile_m2d_converter()
    expected = converter.convert(
        measurements=measurements, append_observables=False
    )

    formed = _formed_at(seat, table, measurements)

    assert formed == _as_rows(expected)
    assert any(any(row) for row in formed)


def _formed_at(seat, table, measurements) -> list:
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
    last_round = len(packets)
    after_last_round = last_round + 1
    later_window = range(3, after_last_round)
    earlier_window = range(1, last_round)
    events_by_round = {}
    for window in (later_window, earlier_window):
        fragments = _window_fragments(window, packets)
        before = _rounds_before(placement, seat, window, packets)
        formed = placement.form_at(seat, fragments, before)
        for carried in formed:
            events_by_round[carried.round_index] = carried.bits
    row = []
    for round_index in sorted(events_by_round):
        row.extend(events_by_round[round_index])
    return tuple(row)


def _rounds_before(placement, seat, window, packets) -> tuple:
    """The raw rounds before the window the seat says it needs."""
    first_round = window[0]
    last_round = window[-1]
    rounds_before = placement.rounds_needed_before(
        seat, 1, first_round, last_round
    )
    return _window_fragments(rounds_before, packets)


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


def _twice_read_seated(formed_at) -> formation.SeatedFormation:
    """Three one-bit rounds; rounds 2 and 3 each read round 1."""
    circuit = stim.Circuit(
        "R 0\nM 0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1] rec[-2]\n"
        "M 0\nDETECTOR rec[-1] rec[-3]"
    )
    table = detector_formation.build_formation_table(
        circuit, 3, measurement_rounds={0: 1, 1: 2, 2: 3}
    )
    settings = event_settings.DetectionEventSettings(formed_at=formed_at)
    source = CircuitSource(table)
    return formation.SeatedFormation(source, settings)
