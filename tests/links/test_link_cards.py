"""A link card in the yaml reaches the path settings the run is built from.

Source: configs/reference.yaml, the links section (one card per path in
cycles of a named clock), and decsim/links/link_profiles.from_yaml.
"""

import pytest

import decsim.machine as machine_module
from decsim.config import microseconds_to_ticks
from decsim.front.experiment import load_experiment

CARD_YAML = (
    "qpu: {kind: stim_device}\n"
    "escalation: {kind: weak_baseline}\n"
    "workload: {kind: memory_circuit, "
    "code_task: surface_code:rotated_memory_z,\n"
    "           rounds_per_shot: 15}\n"
    "windows: {kind: sliding, commit_rounds: null, buffer_rounds: null}\n"
    "sweep: [{physical_error_probability: [0.001], distance: [3],\n"
    "         round_period_us: [1.0], shots: 1}]\n"
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
    "bits_per_cycle: null}\n"
    "  weak_decoder_to_frame: {latency_cycles: 250, clock: fridge, "
    "bits_per_cycle: null}\n"
    "round_store: {rounds: null}\n"
    "strong_round_store: {rounds: null}\n"
    "weak_decoder:\n"
    "  kind: 0.028\n"
    "  units: 1\n"
    "  unit_memory_rounds: null\n"
    "  engine: {clock: fridge, fetch_cycles_per_round: 1, "
    "release_cycles_per_job: 1}\n"
    "pauli_frame: {clock: fridge, write_cycles: 1}\n"
)


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
    assert (
        card.controller_to_weak_buffer.channel.capacity.aggregate_bits_per_microsecond
        == 100_000.0
    )
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
    """C7 item 4: the claim belongs to the card it is about.

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
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    with pytest.raises(ValueError) as refusal:
        machine_module.Machine.build(settings, 0)
    assert "qpu_to_controller card" in str(refusal.value)
