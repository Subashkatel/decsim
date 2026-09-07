"""The `decsim` command set: the verbs, and what each one is equal to.

The shape is sinter's (sinter/_command/_main.py: one command, a verb per
first word, each verb's module imported lazily) as
docs/rewrite/notes/slice_11_front.md section 3 sets it out. Each test
here pins a verb against what the same work done in Python returns, so a
command line is never the only record of a number.
"""

import csv

import pytest

import decsim.front.command as command
import decsim.front.experiment as experiment
import decsim.front.run_command as run_command
import decsim.machine as machine_module
import tests.front.yaml_configs as yaml_configs
import tests.observe.gate_point as gate_point

CONFIGS_DIR = yaml_configs.CONFIGS_DIR
FOUR_POINT_SWEEP = {
    "sweep": [
        {
            "physical_error_probability": [0.001, 0.003],
            "distance": [3, 5],
            "round_period_us": [1.0],
            "shots": 2,
        }
    ]
}


def _rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _rows_without_wall_clock(run_dir, name):
    path = run_dir / name
    rows = _rows(path)
    stripped = []
    for row in rows:
        row.pop("sim_wall_seconds", None)
        row.pop("sim_wall_seconds_per_shot", None)
        stripped.append(row)
    return stripped


def test_an_unknown_verb_prints_the_verbs_and_fails():
    with pytest.raises(SystemExit):
        command.main(["decode-everything"])


def test_help_prints_the_verbs_without_failing():
    command.main(["--help"])


def test_show_lists_every_sections_kind_of_every_shipped_config():
    found = CONFIGS_DIR.glob("*.yaml")
    for config_path in sorted(found):
        config = experiment.load_experiment(config_path)
        lines = experiment.resolved_description(config)
        text = "\n".join(lines)
        assert f"config: {config_path}" in lines[0]
        assert "qpu: kind stim_device" in text
        assert "sweep block 1:" in text
        assert "log: " in text
        assert "trace: " in text


@gate_point.needs_the_frozen_suite
def test_run_prints_the_result_fields_the_gate_hashes():
    config_path = gate_point.SUITE / "weak_decoder_baseline.yaml"
    lines = run_command.run_one_shot(config_path, seed=0)
    config = experiment.load_experiment(config_path)
    block = config.sweep[0]
    settings = config.point_settings(
        physical_error_probability=block.physical_error_probabilities[0],
        distance=block.distances[0],
        round_period_us=block.round_periods_microseconds[0],
    )
    machine = machine_module.Machine.build(settings, 0)
    result = machine.run()
    row = result.operation_results[0]
    text = "\n".join(lines)
    assert f"terminal status: {result.terminal_status}" in text
    assert f"execution done: {result.execution_done_ticks} ticks" in text
    assert f"fully done: {result.fully_done_ticks} ticks" in text
    assert f"observables {row.logical_observables}" in text
    assert f"truth {row.observable_truth}" in text


@gate_point.needs_the_frozen_suite
def test_run_with_trace_writes_the_shots_trace_file(tmp_path):
    config_path = gate_point.SUITE / "weak_decoder_baseline.yaml"
    run_command.run_one_shot(config_path, seed=0, out_dir=tmp_path, trace=True)
    trace_dir = tmp_path / "trace"
    entries = trace_dir.iterdir()
    written = sorted(entries)
    assert len(written) == 1


def test_run_without_a_log_or_a_trace_writes_no_folder(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "out"
    lines = run_command.run_one_shot(config_path, seed=0, out_dir=out_dir)
    assert not out_dir.exists()
    assert "terminal status: complete" in lines[2]


def test_a_pooled_collect_writes_the_serial_collects_rows(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(pooled_dir),
            "--processes",
            "4",
        ]
    )
    for name in ("sweep.csv", "shots.csv", "links.csv"):
        serial = _rows_without_wall_clock(serial_dir, name)
        pooled = _rows_without_wall_clock(pooled_dir, name)
        assert serial == pooled


def test_two_shards_combined_are_the_unsharded_collects_rows(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    whole_dir = tmp_path / "whole"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
    command.main(
        ["collect", str(config_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    command.main(
        [
            "combine",
            str(first_dir),
            str(second_dir),
            "--out",
            str(combined_dir),
        ]
    )
    for name in ("sweep.csv", "shots.csv", "links.csv"):
        whole = _rows_without_wall_clock(whole_dir, name)
        combined = _rows_without_wall_clock(combined_dir, name)
        assert whole == combined


def test_every_shard_of_a_sweep_runs_a_share_of_its_points(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    command.main(
        ["collect", str(config_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    first_path = first_dir / "sweep.csv"
    second_path = second_dir / "sweep.csv"
    first_rows = _rows(first_path)
    second_rows = _rows(second_path)
    assert len(first_rows) == 2
    assert len(second_rows) == 2


def test_a_shard_outside_its_count_is_refused(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="is not a shard"):
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(out_dir),
                "--shard",
                "2/2",
            ]
        )


def test_combining_two_folders_that_hold_the_same_point_is_refused(tmp_path):
    config_path = yaml_configs.write_config(tmp_path, {})
    run_dir = tmp_path / "one"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(run_dir)])
    with pytest.raises(ValueError, match="more than one of"):
        command.main(
            [
                "combine",
                str(run_dir),
                str(run_dir),
                "--out",
                str(combined_dir),
            ]
        )
