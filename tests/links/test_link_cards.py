"""A link card in the yaml reaches the path settings the run is built from.

Source: configs/reference.yaml, the links section (one card per path in
cycles of a named clock), and decsim/links/link_profiles.from_yaml.
"""

import pytest

import decsim.config as config_module
import decsim.links.link_profiles as link_profiles
import decsim.machine as machine_module
from decsim.config import microseconds_to_ticks
from decsim.experiments.experiment import load_experiment

CARD_YAML = (
    "qpu: {kind: stim_device}\n"
    "escalation: {kind: weak_baseline}\n"
    "workload: {kind: memory_circuit, "
    "code_task: surface_code:rotated_memory_z,\n"
    "           rounds_per_shot: 15}\n"
    "windows: {kind: sliding, commit_rounds: null, buffer_rounds: null}\n"
    "sweep: [{physical_error_probability: [0.001], distance: [3],\n"
    "         round_period_microseconds: [1.0], shots: 1}]\n"
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
    config = load_experiment(card_path)
    card = config.settings.links
    assert (
        card.weak_buffer_to_weak_decoder.setup_ticks
        == microseconds_to_ticks(0.4)
    )
    assert card.decoder_to_decoder.setup_ticks == 0


def test_the_header_key_reaches_the_path(tmp_path):
    card_path = tmp_path / "header_card.yaml"
    card_path.write_text(CARD_YAML)
    config = load_experiment(card_path)
    card = config.settings.links
    assert card.decoder_to_decoder.header_bits == 448
    assert card.weak_decoder_to_frame.header_bits == 0


def test_the_latency_and_rate_keys_reach_the_channel(tmp_path):
    card_path = tmp_path / "card.yaml"
    card_path.write_text(CARD_YAML)
    config = load_experiment(card_path)
    card = config.settings.links
    assert (
        card.weak_buffer_to_weak_decoder.channel.name
        == "weak_buffer_to_weak_decoder"
    )
    assert (
        card.weak_buffer_to_weak_decoder.channel.propagation_latency_ticks
        == microseconds_to_ticks(1.0)
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
    config = load_experiment(card_path)
    card = config.settings.links
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
    config = load_experiment(card_path)
    settings = config.point_settings(
        physical_error_probability=0.001,
        distance=3,
        round_period_microseconds=1.0,
    )
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
