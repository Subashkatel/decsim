"""The yaml surface of the decoder card, the sweep and the run folder.

These tests keep configs/reference.yaml and the loader from drifting
apart: every shipped config loads, an unknown key or a stale one is
refused with a sentence, the swept distance reaches the rounds policy,
and a run writes its manifest and its per-shot records.
"""

import pytest

import decsim.front.collect_command as collect_command
import decsim.front.experiment as experiment
import decsim.front.report as report
from tests.front.yaml_configs import (
    CONFIGS_DIR,
    MINIMAL_CONFIG,
    SHIPPED_CONFIGS,
    measure_point_shot,
    write_config,
)


def test_reference_config_defines_both_tiers_and_the_mode_picks_weak():
    reference_path = CONFIGS_DIR / "reference.yaml"
    config = experiment.load_experiment(reference_path)
    settings = config.settings
    assert settings.weak_decoder.kind == "pymatching"
    assert settings.strong_decoder.kind == "belief_matching"
    assert config.active_decoder is settings.weak_decoder
    # engine cycles price on a named domain, resolved once like the links
    assert settings.weak_decoder.engine_megahertz == settings.clocks.megahertz(
        "fridge"
    )
    assert (
        settings.strong_decoder.engine_megahertz
        == settings.clocks.megahertz("room")
    )


def test_every_shipped_config_loads():
    for name in SHIPPED_CONFIGS:
        config_path = CONFIGS_DIR / name
        config = experiment.load_experiment(config_path)
        assert config.active_decoder is not None, name


def test_every_config_the_tests_name_is_shipped():
    for name in SHIPPED_CONFIGS:
        config_path = CONFIGS_DIR / name
        assert config_path.is_file(), name


def test_controller_cycle_card_reaches_both_runtime_paths(tmp_path):
    from decsim.config import microseconds_to_ticks
    from decsim.machine import Machine

    config_path = write_config(
        tmp_path,
        {
            "clocks": {"fridge": 500.0, "room": 250.0},
            "controller": {
                "clock": "fridge",
                "readout_to_bits_cycles": 27,
                "packing_cycles_per_round": 0,
                "decision_to_pulse_cycles": 8,
            },
        },
    )
    config = experiment.load_experiment(config_path)
    assert config.settings.controller.readout_to_bits_microseconds == 0.054
    assert config.settings.controller.decision_to_pulse_microseconds == 0.016

    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    completed = Machine.build(settings, 0)
    assert (
        completed.controller.settings.readout_to_bits_ticks()
        == microseconds_to_ticks(0.054)
    )
    assert completed.instruction_output.pulse_ticks == microseconds_to_ticks(
        0.016
    )


def test_a_mode_without_its_tier_is_refused(tmp_path):
    from decsim.machine import Machine

    config_path = write_config(
        tmp_path, {"escalation": {"kind": "strong_only"}}
    )  # only weak_decoder is defined
    config = experiment.load_experiment(config_path)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    with pytest.raises(ValueError, match="strong tier, which names no decoder"):
        Machine.build(settings)


def test_unknown_algorithms_and_stale_keys_fail_loudly(tmp_path):
    from decsim.machine import Machine

    unknown_algorithm = write_config(
        tmp_path,
        {
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
                "kind": "lookup_table",
            }
        },
    )
    config = experiment.load_experiment(unknown_algorithm)
    settings = config.point_settings(
        physical_error_probability=0.001, distance=3, round_period_us=1.0
    )
    with pytest.raises(
        ValueError, match="weak_decoder.kind 'lookup_table' is not a row"
    ):
        Machine.build(settings)

    old_flat_decoder = write_config(
        tmp_path,
        {
            "decoder": {
                "units": 1,
                "unit_memory_rounds": None,
                "engine": {
                    "clock": "fridge",
                    "fetch_cycles_per_round": 1,
                    "release_cycles_per_job": 1,
                },
            }
        },
    )
    with pytest.raises(ValueError, match=r"no section \['decoder'\]"):
        experiment.load_experiment(old_flat_decoder)

    old_sweep_axis = write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3],
                    "round_period_us": [1.0],
                    "algorithm_latency_us": [0.028],
                    "shots": 1,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="decoder card"):
        experiment.load_experiment(old_sweep_axis)

    fixed_distance_key = write_config(
        tmp_path,
        {
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "round_period_us": [1.0],
                    "shots": 1,
                }
            ]
        },
    )
    with pytest.raises(KeyError):
        experiment.load_experiment(fixed_distance_key)


def test_engine_clock_must_name_a_clock_domain(tmp_path):
    config_path = write_config(
        tmp_path,
        {
            "weak_decoder": {
                **MINIMAL_CONFIG["weak_decoder"],
                "engine": {
                    "clock": "sfq",
                    "fetch_cycles_per_round": 1,
                    "release_cycles_per_job": 1,
                },
            }
        },
    )
    with pytest.raises(ValueError, match="clock 'sfq' is not a clocks entry"):
        experiment.load_experiment(config_path)


def test_report_rows_carry_the_algorithm_column(tmp_path):
    config_path = write_config(tmp_path, {})
    config = experiment.load_experiment(config_path)
    measurements = [
        measure_point_shot(
            config,
            physical_error_probability=0.001,
            distance=3,
            round_period_us=1.0,
            seed=seed,
        )
        for seed in range(2)
    ]
    record = report.record_of(measurements)
    rows = report.summarize(record.shots, record.window_samples)
    assert len(rows) == 1
    assert rows[0]["algorithm"] == 0.028
    per_link = report.link_rows(record.shot_links)
    assert per_link and all(row["algorithm"] == 0.028 for row in per_link)


def test_rounds_per_shot_scales_with_the_swept_distance(tmp_path):
    # "10d" is Toshio 2510.25222's memory-experiment convention: the shot
    # length follows the swept code distance.
    config_path = write_config(
        tmp_path,
        {
            "workload": {
                **MINIMAL_CONFIG["workload"],
                "rounds_per_shot": "10d",
            },
            "sweep": [
                {
                    "physical_error_probability": [0.001],
                    "distance": [3, 5],
                    "round_period_us": [1.0],
                    "shots": 1,
                }
            ],
        },
    )
    config = experiment.load_experiment(config_path)
    rounds_per_shot = config.settings.workload.rounds_per_shot
    assert rounds_per_shot.rounds_for(3) == 30
    assert rounds_per_shot.rounds_for(5) == 50
    assert str(rounds_per_shot) == "10d"
    measurement = measure_point_shot(
        config,
        physical_error_probability=0.001,
        distance=5,
        round_period_us=1.0,
        seed=0,
    )
    assert measurement.distance == 5
    assert measurement.windows > 5


def test_a_run_writes_its_manifest_and_per_shot_records(tmp_path, monkeypatch):
    # One tiny run end to end: a timestamped run dir with manifest.json,
    # the config copy, shots.csv, sweep.csv and links.csv.
    import csv
    import json

    config_path = write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    run_dir, rows = collect_command.run_experiment(config_path)

    manifest_path = run_dir / "manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    assert manifest["versions"]["stim"]
    assert (
        manifest["resolved_config"]["settings"]["escalation"]["kind"]
        == "weak_baseline"
    )
    assert manifest["started_utc"] and manifest["finished_utc"]

    shots_csv_path = run_dir / "shots.csv"
    with open(shots_csv_path) as handle:
        reader = csv.DictReader(handle)
        shots = list(reader)
    assert len(shots) == 1 and shots[0]["seed"] == "0"

    assert run_dir.name.endswith("-unit_test_config")
    assert (run_dir / "config" / "unit_test_config.yaml").exists()
    assert (run_dir / "sweep.csv").exists() and (run_dir / "links.csv").exists()
