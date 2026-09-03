"""The number cards carry their sources' numbers.

Sources: Khalid et al. Table II for the reference latencies; Caune et al.
2410.05202 (a 32-bit bus word) and Fruitwala et al. 2404.15260 (a 128-bit
instruction word) for the default payloads; the provisioning rule of the
bandwidth card (each path carries its nominal traffic in one commit
region, ns-3's per-device DataRate); gem5-Aladdin's setup cost (Shao et
al., MICRO 2016) for with_transfer_overhead.
"""

import pytest

import decsim.config as config
import decsim.links.link_profiles as link_profiles
import decsim.run_spec as run_spec

# One distance-5 patch: 24 syndrome bits per 1.0 us round, commit and
# buffer regions of 5 rounds.
DISTANCE_5_GEOMETRY = dict(
    syndrome_bits_per_round=24, round_us=1.0, commit_rounds=5, buffer_rounds=5
)


def latencies_of(profile):
    latencies = {}
    for path in profile.wired_paths():
        path_settings = profile.path_settings(path)
        channel = path_settings.channel
        latencies[path.value] = channel.propagation_latency_ticks
    return latencies


def capacities_of(profile):
    capacities = {}
    for path in profile.wired_paths():
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        capacities[path.value] = capacity.aggregate_bits_per_microsecond
    return capacities


def channel_names_of(profile):
    names = []
    for path in profile.wired_paths():
        path_settings = profile.path_settings(path)
        names.append(path_settings.channel.name)
    return names


def test_the_reference_card_carries_khalids_latencies():
    profile = link_profiles.logical_reference_profile()
    assert latencies_of(profile) == {
        "qc": config.microseconds_to_ticks(0.15),
        "wbd": config.microseconds_to_ticks(2.0),
        "wsd": config.microseconds_to_ticks(0.5),
        "sbd": config.microseconds_to_ticks(2.0),
        "wdo": config.microseconds_to_ticks(1.0),
        "dd": config.microseconds_to_ticks(0.5),
        "do": config.microseconds_to_ticks(1.0),
        "oc": config.microseconds_to_ticks(4.0),
        "cq": config.microseconds_to_ticks(0.15),
    }


def test_the_reference_card_is_unbounded_on_every_path():
    profile = link_profiles.logical_reference_profile()
    capacities = []
    for path in profile.wired_paths():
        path_settings = profile.path_settings(path)
        capacities.append(path_settings.channel.capacity)
    assert capacities == [None] * 9
    assert profile.profile_name == "logical_reference"
    assert profile.is_controller_processing_outside_qpu_to_controller is False


def test_the_reference_card_prices_a_bus_word_and_an_instruction_word():
    profile = link_profiles.logical_reference_profile()
    assert profile.dd.default_payload.aggregate_bits == 100
    assert profile.oc.default_payload.aggregate_bits == 32
    assert profile.cq.default_payload.aggregate_bits == 128


def test_the_reference_card_names_the_runtime_quantity_of_each_actual_path():
    profile = link_profiles.logical_reference_profile()
    assert profile.qc.actual_payload_source == "SyndromePayload.size_bits"
    assert profile.wbd.actual_payload_source == (
        "SyndromeRoundPacket.fragment_size_sum"
    )
    assert (
        profile.wsd.actual_payload_source == "switching decision payload_bits"
    )
    assert profile.sbd.actual_payload_source == (
        "DecodeJob.retained_payload_size_bits"
    )
    assert profile.wdo.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )
    assert profile.do.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )


def test_each_channel_of_the_reference_card_carries_its_paths_name():
    profile = link_profiles.logical_reference_profile()
    names = channel_names_of(profile)
    assert names == ["qc", "wbd", "wsd", "sbd", "wdo", "dd", "do", "oc", "cq"]


def test_the_bandwidth_card_provisions_each_path_for_one_commit_region():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert capacities_of(profile) == {
        "qc": 24.0,
        "wbd": 48.0,
        "wsd": 0.2,
        "sbd": 72.0,
        "wdo": 0.2,
        "dd": 20.0,
        "do": 0.2,
        "oc": 6.4,
        "cq": 25.6,
    }
    assert profile.profile_name == "bandwidth_limited"


def test_the_bandwidth_card_keeps_the_reference_latencies():
    reference = link_profiles.logical_reference_profile()
    bandwidth = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert latencies_of(bandwidth) == latencies_of(reference)


def test_the_bandwidth_cards_default_payloads_are_one_regions_traffic():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert profile.qc.default_payload.aggregate_bits == 24
    assert profile.wbd.default_payload.aggregate_bits == 240
    assert profile.sbd.default_payload.aggregate_bits == 360
    assert profile.wsd.default_payload.aggregate_bits == 1


def test_the_capacity_scale_multiplies_every_rate():
    halved = link_profiles.bandwidth_limited_profile(
        **DISTANCE_5_GEOMETRY, capacity_scale=0.5
    )
    capacities = capacities_of(halved)
    assert capacities["qc"] == 12.0
    assert capacities["cq"] == 12.8


def test_a_capacity_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        link_profiles.bandwidth_limited_profile(
            **DISTANCE_5_GEOMETRY, capacity_scale=0.0
        )


def test_the_setup_cost_lands_on_the_two_decoder_input_paths():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_transfer_overhead(reference, overhead_us=0.4)
    assert profile.wbd.setup_ticks == config.microseconds_to_ticks(0.4)
    assert profile.sbd.setup_ticks == config.microseconds_to_ticks(0.4)
    assert profile.qc.setup_ticks == 0
    assert profile.profile_name == "logical_reference+transfer_overhead"


def test_a_setup_cost_on_an_unwired_path_is_refused():
    reference = link_profiles.logical_reference_profile()
    with pytest.raises(ValueError, match="csb is not wired on this card"):
        link_profiles.with_transfer_overhead(
            reference, overhead_us=0.4, paths=("csb",)
        )


def test_the_priced_strong_store_hop_is_added_with_its_numbers():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_csb_edge(
        reference,
        latency_us=0.5,
        aggregate_bits_per_us=100.0,
        source="test card",
    )
    assert profile.csb.channel.name == "csb"
    assert profile.csb.channel.propagation_latency_ticks == 500_000
    assert profile.csb.channel.capacity.aggregate_bits_per_microsecond == 100.0
    assert profile.csb.channel.configuration_source == "test card"
    assert profile.profile_name == "logical_reference+priced_csb"


def test_the_priced_weak_store_hop_is_unbounded_when_no_rate_is_given():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_controller_to_buffer_edge(
        reference,
        latency_us=0.25,
        aggregate_bits_per_us=None,
        source="test card",
    )
    assert profile.cwb.channel.capacity is None
    assert profile.cwb.actual_payload_source == (
        "SyndromeRoundPacket.fragment_size_sum"
    )
    assert profile.profile_name == "logical_reference+priced_cwb"


def test_a_run_without_a_card_uses_the_reference_card():
    default_run = run_spec.RunSpec(ops=[])
    default_completed = default_run.build()
    reference = link_profiles.logical_reference_profile()
    explicit_run = run_spec.RunSpec(ops=[], links=reference)
    explicit_completed = explicit_run.build()
    assert default_completed.result == explicit_completed.result
