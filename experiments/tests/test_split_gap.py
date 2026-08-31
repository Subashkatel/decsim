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

import numpy
import pytest
import yaml

from experiments.build_run import decoder_engine
from experiments.experiment_config import load_experiment

from test_decoder_units import write_config
from test_parallel_gap import surface_code_metric
from test_switching_mode import measured_shot, switching_config


def split_config(tmp_path, gap_threshold_db: float, gap_units: int = 1):
    path = switching_config(tmp_path, gap_threshold_db)
    raw = yaml.safe_load(path.read_text())
    raw["switching"]["gap_computation"] = "split_pair"
    raw["switching"]["gap_units"] = gap_units
    return write_config(tmp_path, raw)


def test_gap_units_requires_split_pair(tmp_path):
    path = switching_config(tmp_path, 20.0)
    raw = yaml.safe_load(path.read_text())
    raw["switching"]["gap_units"] = 2
    with pytest.raises(ValueError, match="gap_units"):
        load_experiment(write_config(tmp_path, raw))


def test_split_pair_refuses_double_window(tmp_path):
    path = switching_config(tmp_path, 20.0)
    raw = yaml.safe_load(path.read_text())
    raw["switching"]["gap_computation"] = "split_pair"
    raw["switching"]["double_window"] = True
    with pytest.raises(ValueError, match="serial switching only"):
        load_experiment(write_config(tmp_path, raw))


def test_split_pair_refuses_a_priced_card_weak_tier(tmp_path):
    path = split_config(tmp_path, 20.0)
    raw = yaml.safe_load(path.read_text())
    raw["decoder"]["weak"]["algorithm"] = 0.028
    config = load_experiment(write_config(tmp_path, raw))
    with pytest.raises(ValueError, match="wall-clock"):
        decoder_engine(config)


def test_the_two_forced_solves_reassemble_the_serial_gap():
    metric, detection_events = surface_code_metric()
    for shot_events in detection_events[:50]:
        serial = metric.evaluate(shot_events)
        weight_class_0, _ = metric.forced_class_solve(shot_events, 0)
        weight_class_1, _ = metric.forced_class_solve(shot_events, 1)
        joined_gap = abs(weight_class_0 - weight_class_1)
        assert joined_gap == pytest.approx(serial.gap, abs=1e-9)
        assert min(weight_class_0, weight_class_1) == pytest.approx(
            serial.w_min, abs=1e-9)


def test_split_pair_matches_serial_decisions_and_doubles_wbd(tmp_path):
    serial = load_experiment(switching_config(tmp_path, 20.0))
    split = load_experiment(split_config(tmp_path, 20.0, gap_units=2))
    for seed in range(4):
        serial_shot = measured_shot(serial, seed)
        split_shot = measured_shot(split, seed)
        assert (split_shot.logical_failure
                == serial_shot.logical_failure)
        serial_links = serial_shot.link_totals
        split_links = split_shot.link_totals
        assert (split_links["wsd"]["transfers"]
                == serial_links["wsd"]["transfers"])
        # every window's syndrome crosses WBD twice: once to the weak
        # unit, once to its gap sibling's unit
        assert (split_links["wbd"]["transfers"]
                == 2 * serial_links["wbd"]["transfers"])


def test_split_pair_threshold_edges_still_pin_the_plumbing(tmp_path):
    never = load_experiment(split_config(tmp_path, 0.0))
    always = load_experiment(split_config(tmp_path, 10000.0))
    never_shot = measured_shot(never, 0)
    always_shot = measured_shot(always, 0)
    assert never_shot.link_totals["wsd"]["transfers"] == 0
    assert (always_shot.link_totals["wsd"]["transfers"]
            == always_shot.windows)
