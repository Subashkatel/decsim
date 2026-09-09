"""One tier's event-detection logic and the stage that prices it.

The cost is Yang et al. 2605.04892 lines 1273-1275, "All variables are
stored in FPGA registers, enabling fully pipelined operation. The total
latency of the preprocessing stage for syndrome calculation is fixed at
20 ns (5 FPGA clock cycles)", counted inside "Subtotal (decoder) 148" in
their Table I (lines 1049-1052); the block sits in front of the decoder
core, as LILLIPUT's Event Detection Logic does (2108.06569 lines
499-510). Fully pipelined means one round a clock after the first, the
rate LILLIPUT's FIFO reads at (2108.06569 line 593), so n rounds cost
the latency plus n - 1. The laws here are the tier's: a round it has
already formed is neither formed nor charged again, a round is charged
to the job that first asks for it, and a job that never runs gives its
rounds back.
"""

import decsim.decoders.detection_events as detection_events
import decsim.detector_error_model.detection_event_formation as event_formation
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records

YANG_LATENCY_CYCLES = 5
ONE_ROUND_A_CLOCK = 1


class Former:
    """A former: one round's events are its round index, once each."""

    def __init__(self):
        self.asked = []

    def form_round(self, operation_id, round_index, raw_bits):
        """One round's detection events."""
        self.asked.append((operation_id, round_index))
        del raw_bits
        return (round_index, 0)


def fragment(round_index, bits=(1, 0, 1, 1)):
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_id=0,
        round_index=round_index,
        bits=bits,
        size_bits=len(bits),
        fragment_index=0,
    )


def job(round_indices):
    """One window's job over those rounds, before its input lands."""
    payloads = []
    for round_index in round_indices:
        carried = fragment(round_index)
        payloads.append(carried)
    return decoding_records.DecodeJob(
        operation_id=1,
        window_id=0,
        round_count=len(payloads),
        payloads=payloads,
    )


def stage(formation):
    return detection_events.DetectionEventFormationStage(
        "detection_event_formation",
        cycles_per_job=YANG_LATENCY_CYCLES,
        cycles_per_round=ONE_ROUND_A_CLOCK,
        formation=formation,
    )


def test_a_formed_round_carries_its_events_at_their_own_width():
    former = Former()
    formation = detection_events.TierFormation(former)
    reading_two_rounds = job([1, 2])

    formed = formation.form(reading_two_rounds.payloads)

    assert [carried.bits for carried in formed] == [(1, 0), (2, 0)]
    assert [carried.size_bits for carried in formed] == [2, 2]


def test_a_round_two_windows_read_is_formed_once():
    """The shipped former remembers it; the tier reads the same events."""
    table = Former()
    remembered = event_formation.RememberedDetectionEvents(table)
    formation = detection_events.TierFormation(remembered)
    first = job([1, 2, 3])
    second = job([2, 3, 4])

    formed_first = formation.form(first.payloads)
    formed_second = formation.form(second.payloads)

    assert table.asked == [(1, 1), (1, 2), (1, 3), (1, 4)]
    assert formed_first[1].bits == formed_second[0].bits


def test_a_window_pays_the_stages_latency_then_one_round_a_clock():
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    reading_six_rounds = job([1, 2, 3, 4, 5, 6])

    cycles = formation_stage.cycles_for(reading_six_rounds)

    assert cycles == 10


def test_one_round_costs_the_stages_latency_and_nothing_more():
    """A pipelined stage's fixed latency is one round's way through it."""
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    reading_one_round = job([1])

    cycles = formation_stage.cycles_for(reading_one_round)

    assert cycles == 5


def test_an_overlapping_window_pays_only_for_the_rounds_it_brings():
    """5 + (r - b - 1): the b rounds it shares are already formed."""
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    first = job([1, 2, 3, 4, 5, 6])
    second = job([4, 5, 6, 7, 8, 9])

    first_cycles = formation_stage.cycles_for(first)
    second_cycles = formation_stage.cycles_for(second)

    assert first_cycles == 10
    assert second_cycles == 7


def test_the_second_tier_charges_its_own_rounds():
    """Two tiers read two stores, so each forms what it reads."""
    former = Former()
    weak = detection_events.TierFormation(former)
    strong = detection_events.TierFormation(former)
    weak_stage = stage(weak)
    strong_stage = stage(strong)
    weak_job = job([1, 2, 3])
    strong_job = job([1, 2, 3])

    weak_cycles = weak_stage.cycles_for(weak_job)
    strong_cycles = strong_stage.cycles_for(strong_job)

    assert weak_cycles == 7
    assert strong_cycles == 7
    assert former.asked == []


def test_the_rounds_a_job_pays_for_are_frozen_at_the_first_ask():
    """The dispatcher's prediction and the stage's walk read one number."""
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    reading_three_rounds = job([1, 2, 3])

    at_dispatch = formation_stage.cycles_for(reading_three_rounds)
    at_the_walk = formation_stage.cycles_for(reading_three_rounds)

    assert at_dispatch == 7
    assert at_the_walk == 7


def test_the_stage_reports_the_rounds_it_formed():
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    reading_two_rounds = job([1, 2])

    round_keys = formation_stage.formed_round_keys(reading_two_rounds)

    assert round_keys == ((1, 1), (1, 2))


def test_a_job_that_carries_no_rounds_forms_and_pays_nothing():
    former = Former()
    formation = detection_events.TierFormation(former)
    formation_stage = stage(formation)
    windowless = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=0
    )

    cycles = formation_stage.cycles_for(windowless)

    assert cycles == 0
