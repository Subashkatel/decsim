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
import decsim.machine as machine

# One distance-5 patch: 24 syndrome bits per 1.0 us round, commit and
# buffer regions of 5 rounds.
DISTANCE_5_GEOMETRY = dict(
    syndrome_bits_per_round=24,
    round_microseconds=1.0,
    commit_rounds=5,
    buffer_rounds=5,
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
        "qpu_to_controller": config.microseconds_to_ticks(0.15),
        "weak_buffer_to_weak_decoder": config.microseconds_to_ticks(2.0),
        "weak_decoder_to_strong_decoder": config.microseconds_to_ticks(0.5),
        "strong_buffer_to_strong_decoder": config.microseconds_to_ticks(2.0),
        "weak_decoder_to_frame": config.microseconds_to_ticks(1.0),
        "decoder_to_decoder": config.microseconds_to_ticks(0.5),
        "strong_decoder_to_frame": config.microseconds_to_ticks(1.0),
        "frame_to_controller": config.microseconds_to_ticks(4.0),
        "controller_to_qpu": config.microseconds_to_ticks(0.15),
    }


def test_the_reference_card_is_unbounded_on_every_path():
    profile = link_profiles.logical_reference_profile()
    assert profile.qpu_to_controller.channel.capacity is None
    assert profile.weak_buffer_to_weak_decoder.channel.capacity is None
    assert profile.weak_decoder_to_strong_decoder.channel.capacity is None
    assert profile.strong_buffer_to_strong_decoder.channel.capacity is None
    assert profile.weak_decoder_to_frame.channel.capacity is None
    assert profile.decoder_to_decoder.channel.capacity is None
    assert profile.strong_decoder_to_frame.channel.capacity is None
    assert profile.frame_to_controller.channel.capacity is None
    assert profile.controller_to_qpu.channel.capacity is None
    assert profile.profile_name == "logical_reference"
    assert profile.is_controller_processing_outside_qpu_to_controller is False


def test_the_reference_card_prices_a_bus_word_and_an_instruction_word():
    profile = link_profiles.logical_reference_profile()
    assert profile.decoder_to_decoder.default_payload.aggregate_bits == 100
    assert profile.frame_to_controller.default_payload.aggregate_bits == 32
    assert profile.controller_to_qpu.default_payload.aggregate_bits == 128


def test_the_reference_card_names_the_runtime_quantity_of_each_actual_path():
    profile = link_profiles.logical_reference_profile()
    assert (
        profile.qpu_to_controller.actual_payload_source
        == "SyndromePayload.size_bits"
    )
    assert profile.weak_buffer_to_weak_decoder.actual_payload_source == (
        "SyndromeRoundPacket.fragment_size_sum"
    )
    assert (
        profile.weak_decoder_to_strong_decoder.actual_payload_source
        == "switching decision payload_bits"
    )
    assert profile.strong_buffer_to_strong_decoder.actual_payload_source == (
        "DecodeJob.retained_payload_size_bits"
    )
    assert profile.weak_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )
    assert profile.strong_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )


def test_each_channel_of_the_reference_card_carries_its_paths_name():
    profile = link_profiles.logical_reference_profile()
    names = channel_names_of(profile)
    assert names == [
        "qpu_to_controller",
        "weak_buffer_to_weak_decoder",
        "weak_decoder_to_strong_decoder",
        "strong_buffer_to_strong_decoder",
        "weak_decoder_to_frame",
        "decoder_to_decoder",
        "strong_decoder_to_frame",
        "frame_to_controller",
        "controller_to_qpu",
    ]


def test_the_bandwidth_card_provisions_each_path_for_one_commit_region():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert capacities_of(profile) == {
        "qpu_to_controller": 24.0,
        "weak_buffer_to_weak_decoder": 48.0,
        "weak_decoder_to_strong_decoder": 0.2,
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": 0.2,
        "decoder_to_decoder": 20.0,
        "strong_decoder_to_frame": 0.2,
        "frame_to_controller": 6.4,
        "controller_to_qpu": 25.6,
    }
    assert profile.profile_name == "bandwidth_limited"


def test_the_bandwidth_card_keeps_the_reference_latencies():
    reference = link_profiles.logical_reference_profile()
    bandwidth = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert latencies_of(bandwidth) == latencies_of(reference)


def test_the_bandwidth_cards_default_payloads_are_one_regions_traffic():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert profile.qpu_to_controller.default_payload.aggregate_bits == 24
    assert (
        profile.weak_buffer_to_weak_decoder.default_payload.aggregate_bits
        == 240
    )
    assert (
        profile.strong_buffer_to_strong_decoder.default_payload.aggregate_bits
        == 360
    )
    assert (
        profile.weak_decoder_to_strong_decoder.default_payload.aggregate_bits
        == 1
    )


def test_the_capacity_scale_multiplies_every_rate():
    halved = link_profiles.bandwidth_limited_profile(
        **DISTANCE_5_GEOMETRY, capacity_scale=0.5
    )
    capacities = capacities_of(halved)
    assert capacities["qpu_to_controller"] == 12.0
    assert capacities["controller_to_qpu"] == 12.8


def test_a_capacity_scale_of_zero_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        link_profiles.bandwidth_limited_profile(
            **DISTANCE_5_GEOMETRY, capacity_scale=0.0
        )


def test_the_setup_cost_lands_on_the_two_decoder_input_paths():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_transfer_overhead(
        reference, overhead_microseconds=0.4
    )
    assert (
        profile.weak_buffer_to_weak_decoder.setup_ticks
        == config.microseconds_to_ticks(0.4)
    )
    assert (
        profile.strong_buffer_to_strong_decoder.setup_ticks
        == config.microseconds_to_ticks(0.4)
    )
    assert profile.qpu_to_controller.setup_ticks == 0
    assert profile.profile_name == "logical_reference+transfer_overhead"


def test_a_setup_cost_on_an_unwired_path_is_refused():
    reference = link_profiles.logical_reference_profile()
    with pytest.raises(
        ValueError,
        match="controller_to_strong_buffer is not wired on this card",
    ):
        link_profiles.with_transfer_overhead(
            reference,
            overhead_microseconds=0.4,
            paths=("controller_to_strong_buffer",),
        )


def test_the_priced_strong_store_hop_is_added_with_its_numbers():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_controller_to_strong_buffer_path(
        reference,
        latency_microseconds=0.5,
        aggregate_bits_per_microsecond=100.0,
        source="test card",
    )
    assert (
        profile.controller_to_strong_buffer.channel.name
        == "controller_to_strong_buffer"
    )
    assert (
        profile.controller_to_strong_buffer.channel.propagation_latency_ticks
        == 500_000
    )
    assert (
        profile.controller_to_strong_buffer.channel.capacity.aggregate_bits_per_microsecond
        == 100.0
    )
    assert (
        profile.controller_to_strong_buffer.channel.configuration_source
        == "test card"
    )
    assert (
        profile.profile_name
        == "logical_reference+priced_controller_to_strong_buffer"
    )


def test_the_priced_weak_store_hop_is_unbounded_when_no_rate_is_given():
    reference = link_profiles.logical_reference_profile()
    profile = link_profiles.with_controller_to_weak_buffer_path(
        reference,
        latency_microseconds=0.25,
        aggregate_bits_per_microsecond=None,
        source="test card",
    )
    assert profile.controller_to_weak_buffer.channel.capacity is None
    assert profile.controller_to_weak_buffer.actual_payload_source == (
        "SyndromeRoundPacket.fragment_size_sum"
    )
    assert (
        profile.profile_name
        == "logical_reference+priced_controller_to_weak_buffer"
    )


def test_a_run_without_a_card_uses_the_reference_card():
    default_settings = machine.MachineSettings()
    default_machine = machine.Machine.build(default_settings)
    default_result = default_machine.run()
    reference = link_profiles.logical_reference_profile()
    explicit_settings = machine.MachineSettings(links=reference)
    explicit_machine = machine.Machine.build(explicit_settings)
    explicit_result = explicit_machine.run()
    assert default_result == explicit_result
