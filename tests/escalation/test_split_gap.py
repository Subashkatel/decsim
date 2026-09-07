"""The split gap pair: two forced-class solves on two decoder units.

split_pair exists for the unit that cannot hold two matching cores:
the second forced-class solve becomes a sibling job in its own "gap"
unit pool, with its own syndrome transfer over WBD, and the window's
keep-or-escalate decision waits at the manager's join until both
halves report. The contract: identical decisions and commits to the
serial engine on the same seeds, a visibly doubled input-transfer
count, and a join that always resolves (the settled check would stop
the run otherwise).
"""

import dataclasses

import pytest
import yaml

from decsim.front.experiment import load_experiment
from decsim.machine import Machine, build_decoder_unit
from tests.escalation.test_parallel_gap import surface_code_metric
from tests.escalation.test_switching_mode import measured_shot, switching_config
from tests.front.yaml_configs import write_config


def split_config(tmp_path, gap_threshold_db: float, gap_units: int = 1):
    path = switching_config(tmp_path, gap_threshold_db)
    config_text = path.read_text()
    raw = yaml.safe_load(config_text)
    raw["escalation"]["gap_computation"] = "split_pair"
    raw["escalation"]["gap_units"] = gap_units
    return write_config(tmp_path, raw)


def _data_movement_of(config, seed: int) -> dict:
    """One split-pair shot with the data-path counters asked for."""
    settings = config.point_settings(
        physical_error_probability=0.008, distance=3, round_period_us=1.0
    )
    observation = dataclasses.replace(settings.observation, data_movement=True)
    settings = dataclasses.replace(settings, observation=observation)
    machine = Machine.build(settings, seed)
    machine.run()
    return machine.observation.data_movement.json_value()


def test_the_gap_sibling_copies_the_primarys_landed_rounds(tmp_path):
    """data_path.md section 8: the sibling carries its own copy.

    The sibling reads the masked rounds the primary decodes, and it
    takes a copy where the pair is formed rather than holding Buffer 0
    for a second read. So the copy is one event per window that spawns a
    sibling, and it carries exactly the rounds and bits the primary's
    own memory was loaded with.
    """
    path = split_config(tmp_path, 20.0, gap_units=2)
    config = load_experiment(path)

    counts = _data_movement_of(config, 0)

    by_path = counts["copies_by_path"]
    sibling = by_path["unit default#0 memory -> gap sibling input"]
    primary = by_path["Buffer 0 -> unit default#0 memory"]
    assert sibling == primary
    assert sibling["events"] == 10
    assert sibling["rounds"] == 57


def test_gap_units_requires_split_pair(tmp_path):
    path = switching_config(tmp_path, 20.0)
    config_text = path.read_text()
    raw = yaml.safe_load(config_text)
    raw["escalation"]["gap_units"] = 2
    edited_path = write_config(tmp_path, raw)
    with pytest.raises(ValueError, match="gap_units"):
        load_experiment(edited_path)


def test_split_pair_refuses_double_window(tmp_path):
    path = switching_config(tmp_path, 20.0)
    config_text = path.read_text()
    raw = yaml.safe_load(config_text)
    raw["escalation"]["gap_computation"] = "split_pair"
    raw["escalation"]["double_window"] = True
    edited_path = write_config(tmp_path, raw)
    with pytest.raises(ValueError, match="serial switching only"):
        load_experiment(edited_path)


def test_split_pair_refuses_a_priced_card_weak_tier(tmp_path):
    path = split_config(tmp_path, 20.0)
    config_text = path.read_text()
    raw = yaml.safe_load(config_text)
    raw["weak_decoder"]["kind"] = 0.028
    edited_path = write_config(tmp_path, raw)
    config = load_experiment(edited_path)
    with pytest.raises(ValueError, match="wall-clock"):
        build_decoder_unit(config.settings, "weak")


def test_the_two_forced_solves_reassemble_the_serial_gap():
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events[:50]:
        serial = metric.evaluate(shot_events)
        weight_class_0, _ = metric.forced_class_solve(shot_events, 0)
        weight_class_1, _ = metric.forced_class_solve(shot_events, 1)
        weight_difference = weight_class_0 - weight_class_1
        joined_gap = abs(weight_difference)
        assert joined_gap == pytest.approx(serial.gap, abs=1e-9)
        assert min(weight_class_0, weight_class_1) == pytest.approx(
            serial.decoded_class_weight, abs=1e-9
        )


def test_split_pair_matches_serial_decisions_and_doubles_wbd(tmp_path):
    serial_path = switching_config(tmp_path, 20.0)
    serial = load_experiment(serial_path)
    split_path = split_config(tmp_path, 20.0, gap_units=2)
    split = load_experiment(split_path)
    for seed in range(4):
        serial_shot = measured_shot(serial, seed)
        split_shot = measured_shot(split, seed)
        assert split_shot.logical_failure == serial_shot.logical_failure
        serial_links = serial_shot.link_totals
        split_links = split_shot.link_totals
        assert (
            split_links["weak_decoder_to_strong_decoder"]["transfers"]
            == serial_links["weak_decoder_to_strong_decoder"]["transfers"]
        )
        # every window's syndrome crosses WBD twice: once to the weak
        # unit, once to its gap sibling's unit
        assert (
            split_links["weak_buffer_to_weak_decoder"]["transfers"]
            == 2 * serial_links["weak_buffer_to_weak_decoder"]["transfers"]
        )


def test_split_pair_threshold_edges_still_pin_the_plumbing(tmp_path):
    never_path = split_config(tmp_path, 0.0)
    never = load_experiment(never_path)
    always_path = split_config(tmp_path, 10000.0)
    always = load_experiment(always_path)
    never_shot = measured_shot(never, 0)
    always_shot = measured_shot(always, 0)
    assert (
        never_shot.link_totals["weak_decoder_to_strong_decoder"]["transfers"]
        == 0
    )
    assert (
        always_shot.link_totals["weak_decoder_to_strong_decoder"]["transfers"]
        == always_shot.windows
    )
