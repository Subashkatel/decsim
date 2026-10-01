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

import dataclasses

import stim

import decsim.config as config
import decsim.decoders.detection_events as detection_events
import decsim.detector_error_model.detection_event_formation as formation
import decsim.detector_error_model.detector_formation as detector_formation
import decsim.detector_error_model.settings as event_settings
import decsim.records.decoding as decoding_records
import decsim.records.rounds as round_records

# a distance-3 memory: 8 raw bits a round before the last; 4 events on
# the first round and 8 on every bulk round
CIRCUIT = stim.Circuit.generated(
    "surface_code:rotated_memory_z", rounds=10, distance=3
)
TABLE = detector_formation.build_formation_table(CIRCUIT, 10)
FORMER_CLOCK = config.Clock(4000)
BOTH_TIERS = ("weak_decoder", "strong_decoder")


class CircuitSource:
    """A source whose recipes are one circuit's, for every operation."""

    def formation_table(self, operation_id):
        """The one table."""
        del operation_id
        return TABLE


class CountingDetector:
    """A burst detector that records every round it is shown."""

    def __init__(self):
        self.observed = []

    def observe_round(self, operation_id, round_index, events):
        """One round's events."""
        del operation_id, events
        self.observed.append(round_index)


def placement(source=None, detector=None):
    """Both tiers' decoders seated, at Yang's latency and one round a clock."""
    if source is None:
        source = CircuitSource()
    settings = event_settings.DetectionEventSettings(
        formed_at=BOTH_TIERS,
        clock=FORMER_CLOCK,
        latency_cycles=5,
        cycles_per_round=1,
    )
    return formation.SeatedFormation(source, settings, "weak_decoder", detector)


def fragment(round_index, bits=(0,) * 8):
    return round_records.RetainedSyndromeFragment(
        operation_id=1,
        patch_ids=(0,),
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


def weak_tier(detector=None, source=None):
    seated = placement(source, detector)
    return detection_events.TierFormation(seated, "weak_decoder")


def stage(tier_formation):
    return detection_events.DetectionEventFormationStage(
        "detection_event_formation", formation=tier_formation
    )


def weak_stage():
    """The weak tier's formation stage, on a placement of its own."""
    tier = weak_tier()
    return stage(tier)


def test_a_formed_round_carries_its_events_at_the_size_it_landed():
    """The input memory was written eight raw bits a round."""
    tier = weak_tier()
    reading_two_rounds = job([1, 2])

    formed = tier.form(reading_two_rounds.payloads)

    assert [len(carried.bits) for carried in formed] == [4, 8]
    assert [carried.size_bits for carried in formed] == [8, 8]


def test_a_round_two_windows_read_is_formed_once():
    detector = CountingDetector()
    tier = weak_tier(detector)
    first = job([1, 2, 3])
    second = job([2, 3, 4])

    formed_first = tier.form(first.payloads)
    formed_second = tier.form(second.payloads)

    assert detector.observed == [1, 2, 3, 4]
    assert formed_first[1].bits == formed_second[0].bits


def test_a_window_pays_the_stages_latency_then_one_round_a_clock():
    formation_stage = weak_stage()
    reading_six_rounds = job([1, 2, 3, 4, 5, 6])

    cycles = formation_stage.cycles_for(reading_six_rounds)

    assert cycles == 10


def test_one_round_costs_the_stages_latency_and_nothing_more():
    """A pipelined stage's fixed latency is one round's way through it."""
    formation_stage = weak_stage()
    reading_one_round = job([1])

    cycles = formation_stage.cycles_for(reading_one_round)

    assert cycles == 5


def test_the_stage_counts_on_the_formers_clock():
    formation_stage = weak_stage()
    unit_clock = config.Clock(2000)

    clock = formation_stage.priced_on(unit_clock)

    assert clock == FORMER_CLOCK


def test_a_tier_with_no_recipes_keeps_the_rounds_as_they_landed():
    settings = event_settings.DetectionEventSettings(formed_at=BOTH_TIERS)
    seated = formation.SeatedFormation(None, settings)
    tier = detection_events.TierFormation(seated, "weak_decoder")
    reading_two_rounds = job([1, 2])

    formed = tier.form(reading_two_rounds.payloads)

    assert formed == reading_two_rounds.payloads


def test_an_overlapping_window_pays_only_for_the_rounds_it_brings():
    """5 + (r - b - 1): the b rounds it shares are already formed."""
    formation_stage = weak_stage()
    first = job([1, 2, 3, 4, 5, 6])
    second = job([4, 5, 6, 7, 8, 9])

    first_cycles = formation_stage.cycles_for(first)
    second_cycles = formation_stage.cycles_for(second)

    assert first_cycles == 10
    assert second_cycles == 7


def test_the_second_tier_charges_its_own_rounds():
    """Two tiers read two stores, so each forms what it reads."""
    seated = placement()
    weak = detection_events.TierFormation(seated, "weak_decoder")
    strong = detection_events.TierFormation(seated, "strong_decoder")
    weak_job = job([1, 2, 3])
    strong_job = job([1, 2, 3])

    weak_stage_of_tier = stage(weak)
    strong_stage_of_tier = stage(strong)

    weak_cycles = weak_stage_of_tier.cycles_for(weak_job)
    strong_cycles = strong_stage_of_tier.cycles_for(strong_job)

    assert weak_cycles == 7
    assert strong_cycles == 7


def test_a_tier_keeps_no_claim_once_its_round_retires():
    """No job reads a retired round, so its claim goes."""
    seated = placement()
    tier = detection_events.TierFormation(seated, "weak_decoder")
    tier.rounds_to_form(job([1, 2]))

    seated.retire_round((1, 1))
    seated.retire_round((1, 2))

    history = seated.history_by_seat["weak_decoder"]
    assert history.claimed_keys == set()


def test_the_rounds_a_job_pays_for_are_frozen_at_the_first_ask():
    """The dispatcher's prediction and the stage's walk read one number."""
    formation_stage = weak_stage()
    reading_three_rounds = job([1, 2, 3])

    at_dispatch = formation_stage.cycles_for(reading_three_rounds)
    at_the_walk = formation_stage.cycles_for(reading_three_rounds)

    assert at_dispatch == 7
    assert at_the_walk == 7


def test_the_stage_reports_the_rounds_it_formed():
    formation_stage = weak_stage()
    reading_two_rounds = job([1, 2])

    round_keys = formation_stage.formed_round_keys(reading_two_rounds)

    assert round_keys == ((1, 1), (1, 2))


def test_a_job_that_carries_no_rounds_forms_and_pays_nothing():
    formation_stage = weak_stage()
    windowless = decoding_records.DecodeJob(
        operation_id=1, window_id=0, round_count=0
    )

    cycles = formation_stage.cycles_for(windowless)

    assert cycles == 0


def test_a_joint_round_forms_once_after_all_fragments_arrive() -> None:
    detector = CountingDetector()
    tier = weak_tier(detector)
    first = fragment(1, bits=(0, 0, 0))
    second = dataclasses.replace(first, patch_ids=("other",), fragment_index=1)
    last = fragment(1, bits=(0, 0))
    last = dataclasses.replace(last, fragment_index=2)

    (formed,) = tier.form([last, first, second])

    assert detector.observed == [1]
    assert formed.patch_ids == (0, "other")
    assert len(formed.bits) == 4
    # eight raw bits landed in the unit's memory: the size it holds
    assert formed.size_bits == 8
