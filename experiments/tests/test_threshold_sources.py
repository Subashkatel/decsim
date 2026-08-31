"""The switching threshold's three sources: fixed, table, online.

fixed uses the card's gap_threshold_db as given (the paper's constant
gth). table computes nothing at run time: it looks the sweep point up
in an offline calibration csv (calibrate_threshold.py's shape) and
refuses a point the table does not certify. online starts at
gap_threshold_db and adapts it across the point's shots with the
two-loop controller (rate tracker + audit lane); one calibrator per
point, shared by every shot, so the controller learns over the point's
whole window stream.
"""

import math

from experiments.build_run import (online_threshold_calibrator,
                                   resolve_gap_threshold_nats)
from experiments.experiment_config import load_experiment
from experiments.run import run_sweep

import pytest

from test_decoder_units import strong_unit, write_config
from test_switching_mode import NEAR_THRESHOLD_P, switching_config

NATS_TO_DB = 10.0 / math.log(10.0)


def source_config(tmp_path, switching_card: dict, shots: int = 1):
    config_path = switching_config(tmp_path, 20.0)
    import yaml
    raw = yaml.safe_load(config_path.read_text())
    raw["switching"] = switching_card
    raw["sweep"][0]["shots"] = shots
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def calibration_table(tmp_path) -> str:
    table_path = tmp_path / "calibration_table.csv"
    table_path.write_text(
        "distance,p,gth_brute_force,gth_eq4,gth_eq4_wilson\n"
        f"3,{NEAR_THRESHOLD_P},2.5,18.0,19.5\n"
        "5,0.005,10.0,20.1,21.4\n"
        f"7,{NEAR_THRESHOLD_P},,20.0,\n")
    return table_path.name


def test_fixed_is_the_default_source(tmp_path):
    config = load_experiment(switching_config(tmp_path, 20.0))
    assert config.switching.threshold_source == "fixed"
    resolved = resolve_gap_threshold_nats(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3)
    assert resolved == config.switching.gap_threshold_nats
    assert online_threshold_calibrator(
        config, physical_error_probability=NEAR_THRESHOLD_P,
        distance=3) is None


def test_table_source_resolves_the_sweep_point_and_refuses_others(tmp_path):
    table = calibration_table(tmp_path)
    config = load_experiment(source_config(tmp_path, {
        "threshold_source": "table", "threshold_table": table}))
    assert config.switching.gap_threshold_db is None
    assert config.switching.threshold_column == "gth_eq4_wilson"

    resolved = resolve_gap_threshold_nats(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3)
    assert math.isclose(resolved * NATS_TO_DB, 19.5)

    with pytest.raises(ValueError, match="no row for d=3 p=0.002"):
        resolve_gap_threshold_nats(
            config, physical_error_probability=0.002, distance=3)
    with pytest.raises(ValueError, match="entry is empty"):
        resolve_gap_threshold_nats(
            config, physical_error_probability=NEAR_THRESHOLD_P, distance=7)

    brute = load_experiment(source_config(tmp_path, {
        "threshold_source": "table", "threshold_table": table,
        "threshold_column": "gth_brute_force"}))
    resolved = resolve_gap_threshold_nats(
        brute, physical_error_probability=NEAR_THRESHOLD_P, distance=3)
    assert math.isclose(resolved * NATS_TO_DB, 2.5)


def test_table_source_key_guards(tmp_path):
    table = calibration_table(tmp_path)
    with pytest.raises(ValueError, match="drop gap_threshold_db"):
        load_experiment(source_config(tmp_path, {
            "threshold_source": "table", "threshold_table": table,
            "gap_threshold_db": 20.0}))
    with pytest.raises(ValueError, match="needs threshold_table"):
        load_experiment(source_config(tmp_path, {
            "threshold_source": "table"}))
    with pytest.raises(ValueError, match="belong to"):
        load_experiment(source_config(tmp_path, {
            "gap_threshold_db": 20.0, "threshold_table": table}))
    with pytest.raises(ValueError, match="does not exist"):
        resolve_gap_threshold_nats(
            load_experiment(source_config(tmp_path, {
                "threshold_source": "table",
                "threshold_table": "missing.csv"})),
            physical_error_probability=NEAR_THRESHOLD_P, distance=3)


def test_online_card_guards(tmp_path):
    with pytest.raises(ValueError, match="serial-only"):
        load_experiment(source_config(tmp_path, {
            "threshold_source": "online", "gap_threshold_db": 20.0,
            "double_window": True}))
    with pytest.raises(ValueError, match="online card belongs"):
        load_experiment(source_config(tmp_path, {
            "gap_threshold_db": 20.0, "online": {"audit_rate": 0.1}}))
    with pytest.raises(ValueError, match="does not know"):
        load_experiment(source_config(tmp_path, {
            "threshold_source": "online", "gap_threshold_db": 20.0,
            "online": {"audit_probability": 0.1}}))
    with pytest.raises(ValueError, match="audit_rate"):
        load_experiment(source_config(tmp_path, {
            "threshold_source": "online", "gap_threshold_db": 20.0,
            "online": {"audit_rate": 1.5}}))


def test_online_source_learns_across_a_point_and_records_the_path(tmp_path):
    """One calibrator serves every shot of the point (the controller's
    window count spans all shots), every audit resolves, and the
    trajectory csv lands in the run dir."""
    config = load_experiment(source_config(tmp_path, {
        "threshold_source": "online", "gap_threshold_db": 15.0,
        "online": {"audit_rate": 0.3, "target_escalation_rate": 0.2,
                   "max_escalation_rate": 0.5}}, shots=2))
    run_dir = tmp_path / "results"
    run_dir.mkdir()

    measurements = run_sweep(config, run_dir)

    assert len(measurements) == 2
    windows_per_shot = measurements[0].windows
    calibrator = online_threshold_calibrator(
        config, physical_error_probability=NEAR_THRESHOLD_P, distance=3)
    assert calibrator.summary()["windows"] == 0     # a fresh one is fresh
    trajectory_files = list(run_dir.glob("online_threshold_*.csv"))
    assert len(trajectory_files) == 1
    header, first_row = trajectory_files[0].read_text().splitlines()[:2]
    assert header == "window_count,threshold_db,event"
    assert first_row == "0,15.0,start"
    last_window_count = int(
        trajectory_files[0].read_text().splitlines()[-1].split(",")[0])
    assert last_window_count > windows_per_shot     # learned across shots


def test_online_source_reproduces_its_decisions(tmp_path):
    """The calibrator's random stream is seeded by the point identity:
    rerunning the point reruns the same audits."""
    config = load_experiment(source_config(tmp_path, {
        "threshold_source": "online", "gap_threshold_db": 15.0,
        "online": {"audit_rate": 0.3, "target_escalation_rate": 0.2,
                   "max_escalation_rate": 0.5}}, shots=2))

    first = run_sweep(config, None)
    second = run_sweep(config, None)

    first_links = [measurement.link_totals["wsd"]["transfers"]
                   for measurement in first]
    second_links = [measurement.link_totals["wsd"]["transfers"]
                    for measurement in second]
    assert first_links == second_links
    assert ([m.logical_failure for m in first]
            == [m.logical_failure for m in second])
