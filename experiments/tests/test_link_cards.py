"""A link card in the yaml reaches the path settings the run is built from.

Source: experiments/configs/reference.yaml, the links section (one card
per path in cycles of a named clock), and experiments/build_run.link_model.
"""

from decsim.config import microseconds_to_ticks
from experiments.build_run import link_model
from experiments.experiment_config import load_experiment

CARD_YAML = (
    "mode: weak_baseline\n"
    "code_task: surface_code:rotated_memory_z\n"
    "rounds_per_shot: 15\n"
    "windowing: {scheme: sliding, commit_rounds: null, buffer_rounds: null}\n"
    "sweep: [{physical_error_probability: [0.001], distance: [3],\n"
    "         round_period_us: [1.0], shots: 1}]\n"
    "controller: {clock: fridge, "
    "measurement_signal_to_classical_bits_cycles: 0, "
    "t_pack_cycles: 0, "
    "instruction_or_decision_to_analog_control_pulse_cycles: 0}\n"
    "clocks: {fridge: 250.0, room: 250.0}\n"
    "links:\n"
    "  qc:  {latency_cycles: 250, clock: fridge, bits_per_cycle: null}\n"
    "  cwb: {latency_cycles: 125, clock: fridge, bits_per_cycle: 400.0}\n"
    "  wbd: {latency_cycles: 250, clock: fridge, bits_per_cycle: null,\n"
    "        transfer_overhead_cycles: 100}\n"
    "  dd:  {latency_cycles: 125, clock: fridge, bits_per_cycle: null}\n"
    "  wdo: {latency_cycles: 250, clock: fridge, bits_per_cycle: null}\n"
    "buffers: {buffer_0_size: null, buffer_1_size: null,\n"
    "          packing_workspace_size: null}\n"
    "decoder:\n"
    "  weak:\n"
    "    algorithm: 0.028\n"
    "    units: 1\n"
    "    unit_buffer_size: null\n"
    "    engine: {clock: fridge, fetch_cycles_per_round: 1, "
    "release_cycles_per_job: 1}\n"
    "pauli_frame: {clock: fridge, commit_cycles: 1}\n"
)


def test_the_setup_cost_key_reaches_the_path(tmp_path):
    (tmp_path / "overhead_card.yaml").write_text(CARD_YAML)
    card = link_model(load_experiment(tmp_path / "overhead_card.yaml"))
    assert card.wbd.setup_ticks == microseconds_to_ticks(0.4)
    assert card.dd.setup_ticks == 0


def test_the_latency_and_rate_keys_reach_the_channel(tmp_path):
    (tmp_path / "card.yaml").write_text(CARD_YAML)
    card = link_model(load_experiment(tmp_path / "card.yaml"))
    assert card.wbd.channel.name == "wbd"
    assert card.wbd.channel.propagation_latency_ticks == microseconds_to_ticks(
        1.0
    )
    assert card.cwb.channel.capacity.aggregate_bits_per_microsecond == 100_000.0
    assert card.is_controller_processing_outside_qpu_to_controller is True
