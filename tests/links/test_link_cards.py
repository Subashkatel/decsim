"""A link card in the yaml reaches the path settings the run is built from.

Source: configs/reference.yaml, the links section (one card per path in
cycles of a named clock), and decsim/links/link_profiles.from_yaml.
"""

import pytest

import decsim.config as config_module
import decsim.experiments.experiment as experiment
import decsim.links.credit_channel as credit_channel
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
import tests.experiments.yaml_configs as yaml_configs

CARD_YAML = (
    "qpu: {kind: stim_device}\n"
    "escalation: {kind: weak_baseline}\n"
    "workload: {kind: producer, function: decsim.producers:memory_circuit,\n"
    "           arguments: {code_task: surface_code:rotated_memory_z,\n"
    "                       rounds_per_shot: 15,\n"
    "                       distance: '${qpu.distance}'}}\n"
    "windows: {kind: sliding, commit_rounds: null, buffer_rounds: null}\n"
    "sweep: [{axes: {workload.arguments.physical_error_probability: [0.001],\n"
    "                qpu.distance: [3],\n"
    "                qpu.round_period_microseconds: [1.0]},\n"
    "         collection: {max_shots: 1}}]\n"
    "controller: {clock: fridge, "
    "readout_to_bits_cycles: 0, "
    "packing_cycles_per_round: 0, "
    "decision_to_pulse_cycles: 0, packing_rounds_in_flight: null}\n"
    "clocks: {fridge: 250.0, room: 250.0}\n"
    "links:\n"
    "  qpu_to_controller: {latency_cycles: 250, clock: fridge, "
    "bits_per_cycle: null}\n"
    "  controller_to_weak_buffer: {latency_cycles: 125, clock: fridge, "
    "bits_per_cycle: 400.0}\n"
    "  weak_buffer_to_weak_decoder: {latency_cycles: 250, clock: fridge, "
    "bits_per_cycle: null,\n"
    "        setup_cycles_per_transfer: 100}\n"
    "  decoder_to_decoder: {latency_cycles: 125, clock: fridge, "
    "bits_per_cycle: null, header_bits_per_transfer: 448}\n"
    "  weak_decoder_to_frame: {latency_cycles: 250, clock: fridge, "
    "bits_per_cycle: null}\n"
    "weak_syndrome_buffer: {bits: null}\n"
    "strong_syndrome_buffer: {bits: null}\n"
    "weak_decoder:\n"
    "  kind: 0.028\n"
    "  units: 1\n"
    "  unit_memory: {bits: null}\n"
    "  engine: {clock: fridge, fetch_cycles_per_round: 1, "
    "fetch_cycles_per_job: 0, release_cycles_per_job: 1, "
    "release_cycles_per_round: 0}\n"
    "pauli_frame: {clock: fridge, write_cycles: 1}\n"
)

CLOCKS = config_module.ClockSettings.from_yaml({"fridge": 250.0})
GOOD_CARD = {"latency_cycles": 1, "clock": "fridge", "bits_per_cycle": 1.0}


def _load_readout_card(card):
    """The links section with one card on the readout hop, read."""
    section = {"qpu_to_controller": card}
    return link_profiles.from_yaml(section, CLOCKS, "probe")


def test_the_setup_cost_key_reaches_the_path(tmp_path):
    card_path = tmp_path / "overhead_card.yaml"
    card_path.write_text(CARD_YAML)
    config = experiment.load_experiment(card_path)
    first_point = config.first_point_task()
    card = first_point.settings.links
    assert (
        card.weak_buffer_to_weak_decoder.setup_ticks
        == config_module.microseconds_to_ticks(0.4)
    )
    assert card.decoder_to_decoder.setup_ticks == 0


def test_the_header_key_reaches_the_path(tmp_path):
    card_path = tmp_path / "header_card.yaml"
    card_path.write_text(CARD_YAML)
    config = experiment.load_experiment(card_path)
    first_point = config.first_point_task()
    card = first_point.settings.links
    assert card.decoder_to_decoder.header_bits_per_transfer == 448
    assert card.weak_decoder_to_frame.header_bits_per_transfer == 0


def test_the_latency_and_rate_keys_reach_the_channel(tmp_path):
    card_path = tmp_path / "card.yaml"
    card_path.write_text(CARD_YAML)
    config = experiment.load_experiment(card_path)
    first_point = config.first_point_task()
    card = first_point.settings.links
    assert (
        card.weak_buffer_to_weak_decoder.channel.name
        == "weak_buffer_to_weak_decoder"
    )
    assert (
        card.weak_buffer_to_weak_decoder.channel.propagation_latency_ticks
        == config_module.microseconds_to_ticks(1.0)
    )
    store_capacity = card.controller_to_weak_buffer.channel.capacity
    store_rate = store_capacity.exact_aggregate_bits_per_microsecond()
    assert store_rate == 100_000
    assert card.qpu_to_controller.excludes_receiver_processing is True


def _card_yaml_without_the_readout_hop() -> str:
    """The same card with qpu_to_controller left out of the links section."""
    lines = []
    for line in CARD_YAML.splitlines():
        if line.startswith("  qpu_to_controller:"):
            continue
        if line.startswith("  bits_per_cycle: null}"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def test_a_readout_hop_the_yaml_never_wrote_keeps_its_own_cost_inside(
    tmp_path,
):
    """The claim belongs to the card it is about.

    The reference qpu_to_controller latency is the whole transfer, the
    controller's own turning of the readout into bits included. A yaml
    that leaves the path null keeps that number, so it must not also
    claim that the cost sits outside it.
    """
    text = _card_yaml_without_the_readout_hop()
    card_path = tmp_path / "no_readout_hop.yaml"
    card_path.write_text(text)
    config = experiment.load_experiment(card_path)
    first_point = config.first_point_task()
    card = first_point.settings.links
    assert card.qpu_to_controller.excludes_receiver_processing is False
    assert card.controller_to_weak_buffer.excludes_receiver_processing


def test_a_separate_readout_cost_is_refused_on_an_uncarded_readout_hop(
    tmp_path,
):
    """The refusal reads the one card, not a flag for the whole fabric."""
    text = _card_yaml_without_the_readout_hop()
    text = text.replace(
        "readout_to_bits_cycles: 0", "readout_to_bits_cycles: 25"
    )
    card_path = tmp_path / "double_charged.yaml"
    card_path.write_text(text)
    config = experiment.load_experiment(card_path)
    point = config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.001,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings, 0)
    assert "qpu_to_controller card" in str(refusal.value)


def test_a_card_that_is_not_a_mapping_is_refused_naming_its_path():
    with pytest.raises(ValueError) as refusal:
        _load_readout_card([1, 2])
    message = str(refusal.value)
    assert message.startswith(
        "links.qpu_to_controller holds [1, 2]; a path's card is a mapping"
    )


@pytest.mark.parametrize("key", ["latency_cycles", "clock", "bits_per_cycle"])
def test_a_card_missing_a_key_every_card_writes_is_refused_naming_it(key):
    card = dict(GOOD_CARD)
    del card[key]
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message.startswith(
        f"links.qpu_to_controller needs ['{key}']; a card writes"
    )


def test_a_card_key_no_card_reads_is_refused_naming_the_card_keys():
    card = dict(GOOD_CARD, latency_cycle=2)
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message.startswith(
        "links.qpu_to_controller does not know ['latency_cycle']; its keys "
        "are ['latency_cycles', 'clock', 'bits_per_cycle', 'channels', "
    )


def test_unknown_card_keys_of_mixed_types_are_refused_naming_them():
    card = dict(GOOD_CARD)
    card[2] = 1
    card["bad"] = 2
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message.startswith(
        "links.qpu_to_controller does not know [2, 'bad']; its keys are "
    )


@pytest.mark.parametrize(
    "bits_per_cycle", [True, "fast", 0, -1.0, float("nan"), float("inf")]
)
def test_a_lane_rate_that_is_not_a_positive_number_is_refused(bits_per_cycle):
    card = dict(GOOD_CARD, bits_per_cycle=bits_per_cycle)
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message == (
        f"links.qpu_to_controller.bits_per_cycle is {bits_per_cycle!r}; it "
        f"is the positive number of bits each lane moves per cycle, or null "
        f"for an unbounded wire"
    )


@pytest.mark.parametrize(
    ("bits_per_cycle", "bits_per_microsecond"), [(1, 250), (2.5, 625)]
)
def test_a_positive_lane_rate_is_read_at_its_clock(
    bits_per_cycle, bits_per_microsecond
):
    card = dict(GOOD_CARD, bits_per_cycle=bits_per_cycle)
    fabric = _load_readout_card(card)
    capacity = fabric.qpu_to_controller.channel.capacity
    rate = capacity.exact_aggregate_bits_per_microsecond()
    assert rate == bits_per_microsecond


def test_a_null_lane_rate_is_an_unbounded_wire():
    card = dict(GOOD_CARD, bits_per_cycle=None)
    fabric = _load_readout_card(card)
    assert fabric.qpu_to_controller.channel.capacity is None


@pytest.mark.parametrize("lane_count", [True, "four", 0, -2, 2.5])
def test_a_lane_count_that_is_not_a_positive_whole_number_is_refused(
    lane_count,
):
    card = dict(GOOD_CARD, channels=lane_count)
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message == (
        f"links.qpu_to_controller.channels is {lane_count!r}; it is the "
        f"positive whole number of parallel lanes the path's wire has"
    )


def test_a_cards_lanes_multiply_its_lane_rate():
    card = dict(GOOD_CARD, channels=8)
    fabric = _load_readout_card(card)
    capacity = fabric.qpu_to_controller.channel.capacity
    assert capacity.exact_aggregate_bits_per_microsecond() == 8 * 250


@pytest.mark.parametrize("header_bits", [True, -8, 448.5, "448"])
def test_a_header_that_is_not_whole_bits_is_refused_naming_the_card(
    header_bits,
):
    card = dict(GOOD_CARD, header_bits_per_transfer=header_bits)
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message == (
        f"links.qpu_to_controller.header_bits_per_transfer is "
        f"{header_bits!r}; it is the whole number of framing bits every "
        f"transfer of the path carries, zero or more"
    )


@pytest.mark.parametrize(
    ("key", "cycles"),
    [
        ("latency_cycles", True),
        ("latency_cycles", 1.5),
        ("setup_cycles_per_transfer", True),
    ],
)
def test_a_cycle_count_that_is_not_whole_is_refused_naming_the_card(
    key, cycles
):
    card = dict(GOOD_CARD)
    card[key] = cycles
    with pytest.raises(ValueError) as refusal:
        _load_readout_card(card)
    message = str(refusal.value)
    assert message == (
        f"links.qpu_to_controller.{key} must be a nonnegative integer"
    )


# A credit card: 8-bit flits one a cycle, one flit of receive buffer and a
# one-cycle credit return, so a round of several flits waits for credits.
CREDIT_PROTOCOL = {
    "kind": "credit",
    "framing": {"kind": "flits", "flit_bits": 8},
    "receive_buffer_frames": 1,
    "credit_latency_cycles": 1,
}
CREDIT_CARD = {
    "latency_cycles": 1,
    "clock": "fridge",
    "bits_per_cycle": 8.0,
    "protocol": CREDIT_PROTOCOL,
}


def test_a_card_without_a_protocol_is_the_ideal_row():
    profile = _load_readout_card(GOOD_CARD)

    protocol = profile.qpu_to_controller.channel.protocol
    assert protocol is None


def test_a_protocol_card_reaches_the_channel_counted_on_its_clock():
    profile = _load_readout_card(CREDIT_CARD)

    protocol = profile.qpu_to_controller.channel.protocol
    assert type(protocol) is credit_channel.CreditChannel.Settings
    assert protocol.receive_buffer_frames == 1
    assert protocol.framing.flit_bits == 8
    assert protocol.clock.period_ticks == config_module.microseconds_to_ticks(
        0.004
    )


def test_a_protocol_off_the_table_is_refused_naming_the_rows():
    card = dict(CREDIT_CARD, protocol={"kind": "carrier_pigeon"})

    with pytest.raises(ValueError, match="is not a row of its table"):
        _load_readout_card(card)


def test_a_packet_protocol_on_an_unbounded_wire_is_refused():
    card = dict(CREDIT_CARD, bits_per_cycle=None)

    with pytest.raises(ValueError, match="give it a bounded wire"):
        _load_readout_card(card)


def test_a_protocol_key_its_row_does_not_declare_is_refused():
    protocol = dict(CREDIT_PROTOCOL, window_packets=128)
    card = dict(CREDIT_CARD, protocol=protocol)

    with pytest.raises(ValueError, match="does not know"):
        _load_readout_card(card)


def test_a_credit_protocol_without_its_buffer_is_refused():
    protocol = dict(CREDIT_PROTOCOL)
    del protocol["receive_buffer_frames"]
    card = dict(CREDIT_CARD, protocol=protocol)

    with pytest.raises(ValueError, match="needs receive_buffer_frames"):
        _load_readout_card(card)


def _built_from(tmp_path, overrides: dict):
    config_path = yaml_configs.write_config(tmp_path, overrides)
    experiment_config = experiment.load_experiment(config_path)
    point = experiment_config.point_task(
        {
            "workload.arguments.physical_error_probability": 0.008,
            "qpu.distance": 3,
            "qpu.round_period_microseconds": 1.0,
        },
    )
    settings = point.settings
    return machine_module.Machine.build(settings, 0)


def test_a_binding_credit_card_on_the_weak_store_hop_runs_a_shot(tmp_path):
    """A gate point: a round's flits wait for the one-flit buffer's credit."""
    links = dict(yaml_configs.MINIMAL_CONFIG["links"])
    links["controller_to_weak_buffer"] = CREDIT_CARD
    built = _built_from(tmp_path, {"links": links})
    frames = []
    built.links.trace.frame_landed.connect(frames.append)

    result = built.run()

    credit_waits = [frame.timing.credit_wait_ticks for frame in frames]
    assert result.terminal_status == "complete"
    assert max(credit_waits) > 0


def _off_board_card(protocol: dict) -> dict:
    """The strong node one chassis hop away at 100 Gb/s (the reference card)."""
    return {
        "latency_cycles": 65,
        "clock": "room",
        "bits_per_cycle": 400.0,
        "protocol": protocol,
    }


def test_a_reliable_card_at_no_errors_runs_as_its_credit_card(tmp_path):
    """A gate point: RoCE v2 frames on the escalation hop, nothing lost."""
    credit = {
        "kind": "credit",
        "framing": {"kind": "roce_v2", "path_mtu_bytes": 1024},
        "receive_buffer_frames": 8,
        "credit_latency_cycles": 65,
    }
    reliable = dict(
        credit,
        kind="reliable",
        window_packets=128,
        ack_every_packets=66,
        retransmit_timeout_cycles=100_000,
        retry_count=7,
        bit_error_rate=0.0,
    )
    base_path = yaml_configs.CONFIGS_DIR / "examples/two_tiers.yaml"
    base = str(base_path)
    links = {"weak_decoder_to_strong_decoder": _off_board_card(credit)}
    by_credit = _built_from(tmp_path, {"extends": base, "links": links})
    links = {"weak_decoder_to_strong_decoder": _off_board_card(reliable)}
    by_reliable = _built_from(tmp_path, {"extends": base, "links": links})
    frames = []
    by_reliable.links.trace.frame_landed.connect(frames.append)

    credit_result = by_credit.run()
    reliable_result = by_reliable.run()

    assert reliable_result.terminal_status == "complete"
    assert len(frames) > 0
    assert reliable_result.operation_results == credit_result.operation_results
    assert reliable_result.link_traffic == credit_result.link_traffic
