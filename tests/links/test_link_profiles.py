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
import decsim.front.experiment as experiment
import decsim.links.fabric as fabric_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine
import decsim.records.transfers as transfer_records
import decsim.settings as machine_settings
import tests.front.yaml_configs as yaml_configs

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
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        channel = path_settings.channel
        latencies[path.value] = channel.propagation_latency_ticks
    return latencies


def capacities_of(profile):
    capacities = {}
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        capacity = path_settings.channel.capacity
        capacities[path.value] = capacity.aggregate_bits_per_microsecond
    return capacities


def channel_names_of(profile):
    names = []
    for path in transfer_records.LinkPath:
        path_settings = profile.path_settings(path)
        names.append(path_settings.channel.name)
    return names


def path_names():
    names = []
    for path in transfer_records.LinkPath:
        names.append(path.value)
    return names


def test_the_reference_card_carries_its_sources_latencies():
    profile = link_profiles.logical_reference_profile()
    assert latencies_of(profile) == {
        "qpu_to_controller": config.microseconds_to_ticks(0.15),
        "controller_to_weak_buffer": config.microseconds_to_ticks(0.04),
        "controller_to_strong_buffer": config.microseconds_to_ticks(0.26),
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
    assert profile.controller_to_weak_buffer.channel.capacity is None
    assert profile.controller_to_strong_buffer.channel.capacity is None
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
    assert profile.frame_to_controller.default_payload.aggregate_bits == 32
    assert profile.controller_to_qpu.default_payload.aggregate_bits == 128


def test_the_reference_card_prices_a_boundary_on_the_seam_it_updates():
    profile = link_profiles.logical_reference_profile()
    assert profile.decoder_to_decoder.default_payload is None
    assert profile.decoder_to_decoder.actual_payload_source == (
        "DependencyResidual seam-layer detectors"
    )


def test_the_reference_card_names_the_runtime_quantity_of_each_actual_path():
    profile = link_profiles.logical_reference_profile()
    assert (
        profile.qpu_to_controller.actual_payload_source
        == "SyndromePayload.size_bits"
    )
    assert profile.controller_to_weak_buffer.actual_payload_source == (
        "PackedRound.wire_bits"
    )
    assert profile.controller_to_strong_buffer.actual_payload_source == (
        "PackedRound.wire_bits"
    )
    assert profile.weak_buffer_to_weak_decoder.actual_payload_source == (
        "DecodeJob.payload_bits()"
    )
    assert (
        profile.weak_decoder_to_strong_decoder.actual_payload_source
        == "switching decision payload_bits"
    )
    assert profile.strong_buffer_to_strong_decoder.actual_payload_source == (
        "DecodeJob.payload_bits()"
    )
    assert profile.weak_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )
    assert profile.strong_decoder_to_frame.actual_payload_source == (
        "DecodeResult.logical_observables bits"
    )


def test_every_hop_of_the_reaction_path_has_a_reference_card():
    """A null card is the reference numbers, so no hop is ever free.

    The traffic report lists one row per hop of the reaction path, and
    each row is priced: walking LinkPath finds a card on every one, and
    each card's channel carries the hop's own name.
    """
    profile = link_profiles.logical_reference_profile()
    names = channel_names_of(profile)
    assert names == path_names()


def test_the_bandwidth_card_provisions_each_path_for_one_commit_region():
    profile = link_profiles.bandwidth_limited_profile(**DISTANCE_5_GEOMETRY)
    assert capacities_of(profile) == {
        "qpu_to_controller": 24.0,
        "controller_to_weak_buffer": 24.0,
        "controller_to_strong_buffer": 24.0,
        "weak_buffer_to_weak_decoder": 48.0,
        "weak_decoder_to_strong_decoder": 0.2,
        "strong_buffer_to_strong_decoder": 72.0,
        "weak_decoder_to_frame": 0.2,
        "decoder_to_decoder": 4.8,
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
    assert profile.decoder_to_decoder.default_payload.aggregate_bits == 24


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


def test_the_reference_card_prices_the_two_controller_to_buffer_hops():
    """Caune's per-hop stages: the intra-unit write, and the crossing.

    Syndrome buffer 0 sits with the controller, so its write is Fig. 1a
    stage D, the 40 ns the control system takes to handle a result
    message and prepare it for broadcast; syndrome buffer 1 sits at room
    temperature, so its write leaves the chassis on stage F, the 240 to
    260 ns inter-node broadcast, taken at the caption's stated worst
    case (arXiv:2410.05202, Fig. 1a).
    """
    profile = link_profiles.logical_reference_profile()
    weak_store = profile.controller_to_weak_buffer
    strong_store = profile.controller_to_strong_buffer
    weak_ticks = config.microseconds_to_ticks(0.04)
    strong_ticks = config.microseconds_to_ticks(0.26)
    weak_source = weak_store.channel.configuration_source
    strong_source = strong_store.channel.configuration_source
    assert weak_store.channel.propagation_latency_ticks == weak_ticks
    assert strong_store.channel.propagation_latency_ticks == strong_ticks
    assert "Caune 2410.05202 Fig. 1a D" in weak_source
    assert "Caune 2410.05202 Fig. 1a F" in strong_source


def test_a_run_without_a_card_uses_the_reference_card():
    default_settings = machine_settings.MachineSettings()
    default_machine = machine.Machine.build(default_settings)
    default_result = default_machine.run()
    reference = link_profiles.logical_reference_profile()
    explicit_settings = machine_settings.MachineSettings(links=reference)
    explicit_machine = machine.Machine.build(explicit_settings)
    explicit_result = explicit_machine.run()
    assert default_result == explicit_result


class CountingFabric:
    """A fabric row written outside decsim: it counts what it carried.

    Its base card is the reference row's, so a run on it prices every hop
    exactly as the shipped row does and the count is the only difference.
    """

    sent = []

    @staticmethod
    def base_card():
        """The numbers the section's per-path cards override."""
        return link_profiles.logical_reference_profile()

    @staticmethod
    def build(card, engine):
        """One LinkFabric, with every send counted on the way through."""
        return _CountingLinkFabric(card, engine)


class _CountingLinkFabric(fabric_module.LinkFabric):
    """The reference fabric, counting the transfers it carried."""

    def send(self, path, payload_bits, now_ticks, attribution, on_delivered):
        """Count the send, then carry it."""
        CountingFabric.sent.append(path)
        reference = super()
        reference.send(path, payload_bits, now_ticks, attribution, on_delivered)


def test_a_fabric_row_written_outside_decsim_runs_from_a_yaml(
    monkeypatch, tmp_path
):
    """One LINK_FABRICS row and one links.kind is the whole edit."""
    monkeypatch.setitem(link_profiles.LINK_FABRICS, "counting", CountingFabric)
    monkeypatch.setattr(CountingFabric, "sent", [])
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "counting"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    experiment_config = experiment.load_experiment(config_path)
    settings = experiment_config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    built = machine.Machine.build(settings, 0)
    result = built.run()

    assert settings.links.kind == "counting"
    assert isinstance(built.links, _CountingLinkFabric)
    assert result.terminal_status == "complete"
    assert CountingFabric.sent != []


def test_a_links_kind_off_the_table_is_refused_naming_the_rows(tmp_path):
    """The table's own refusal, at the yaml boundary."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "not_a_row"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    with pytest.raises(ValueError, match="links.kind 'not_a_row' is not"):
        experiment.load_experiment(config_path)


def test_the_bandwidth_row_says_what_it_needs_instead_of_a_yaml(tmp_path):
    """Its channels come from the sweep point's geometry, not the section."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["kind"] = "bandwidth_limited"
    config_path = yaml_configs.write_config(tmp_path, {"links": links})
    with pytest.raises(ValueError, match="provisions every channel"):
        experiment.load_experiment(config_path)
