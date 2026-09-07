"""The `decsim` command set: the verbs, and what each one is equal to.

The shape is sinter's (sinter/_command/_main.py: one command, a verb per
first word, each verb's module imported lazily) as
docs/rewrite/notes/slice_11_front.md section 3 sets it out. Each test
here pins a verb against what the same work done in Python returns, so a
command line is never the only record of a number.
"""

import csv
import json

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


EVERY_FILE = (
    "sweep.csv",
    "links.csv",
    "shots.csv",
    "shot_links.csv",
    "window_samples.csv",
)


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


def _point_and_seed_of_every_row(run_dir, name):
    path = run_dir / name
    rows = _rows(path)
    order = []
    for row in rows:
        point_and_seed = (
            row["distance"],
            row["physical_error_probability"],
            row["round_period_us"],
            row["seed"],
        )
        order.append(point_and_seed)
    return order


def _collect_one_shard(config_path, out_dir, shard, shots_per_unit):
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(out_dir),
            "--shard",
            shard,
            "--shots-per-unit",
            str(shots_per_unit),
        ]
    )


def _combined(first_dir, second_dir, out_dir):
    command.main(
        ["combine", str(first_dir), str(second_dir), "--out", str(out_dir)]
    )


def _manifest_of(run_dir):
    path = run_dir / "manifest.json"
    text = path.read_text()
    return json.loads(text)


def _seeds_of_every_shot(run_dir):
    path = run_dir / "shots.csv"
    rows = _rows(path)
    return [row["seed"] for row in rows]


def test_an_unknown_verb_prints_the_verbs_and_fails():
    with pytest.raises(SystemExit):
        command.main(["decode-everything"])


def test_help_prints_the_verbs_without_failing():
    command.main(["--help"])


def test_show_lists_every_sections_kind_of_every_shipped_config():
    for name in yaml_configs.SHIPPED_CONFIGS:
        config_path = CONFIGS_DIR / name
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
    for name in EVERY_FILE:
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
    for name in EVERY_FILE:
        whole = _rows_without_wall_clock(whole_dir, name)
        combined = _rows_without_wall_clock(combined_dir, name)
        assert whole == combined


def test_combine_writes_the_same_rows_whichever_order_the_shards_come_in(
    tmp_path,
):
    """The shards split every point's seeds, so every file is folded."""
    wall_clock_unit = {
        **yaml_configs.MINIMAL_CONFIG["weak_decoder"],
        "kind": "pymatching",
    }
    config_path = yaml_configs.write_config(
        tmp_path, {**FOUR_POINT_SWEEP, "weak_decoder": wall_clock_unit}
    )
    serial_dir = tmp_path / "serial"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    forwards_dir = tmp_path / "forwards"
    backwards_dir = tmp_path / "backwards"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(first_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "0/2",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shots-per-unit",
            "1",
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
            str(forwards_dir),
        ]
    )
    command.main(
        [
            "combine",
            str(second_dir),
            str(first_dir),
            "--out",
            str(backwards_dir),
        ]
    )

    every_file = EVERY_FILE + ("latency_samples.csv",)
    for name in every_file:
        forwards_path = forwards_dir / name
        backwards_path = backwards_dir / name
        assert forwards_path.read_bytes() == backwards_path.read_bytes()
    for name in ("shots.csv", "latency_samples.csv"):
        serial = _point_and_seed_of_every_row(serial_dir, name)
        combined = _point_and_seed_of_every_row(forwards_dir, name)
        assert serial == combined


def test_combining_folders_of_two_different_sweeps_is_refused(tmp_path, capsys):
    first_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    second_path = tmp_path / "other.yaml"
    other_text = first_path.read_text()
    more_shots = other_text.replace("shots: 2", "shots: 3")
    second_path.write_text(more_shots)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    combined_dir = tmp_path / "combined"
    command.main(
        ["collect", str(first_path), "--out", str(first_dir), "--shard", "0/2"]
    )
    command.main(
        [
            "collect",
            str(second_path),
            "--out",
            str(second_dir),
            "--shard",
            "1/2",
        ]
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "combine",
                str(first_dir),
                str(second_dir),
                "--out",
                str(combined_dir),
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "ran a different experiment from" in printed.err


def test_a_unit_size_that_splits_a_point_writes_the_serial_runs_rows(
    tmp_path,
):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    split_dir = tmp_path / "split"
    pooled_dir = tmp_path / "pooled"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(split_dir),
            "--shots-per-unit",
            "1",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(pooled_dir),
            "--shots-per-unit",
            "1",
            "--processes",
            "4",
        ]
    )

    for name in EVERY_FILE:
        serial = _rows_without_wall_clock(serial_dir, name)
        assert _rows_without_wall_clock(split_dir, name) == serial
        assert _rows_without_wall_clock(pooled_dir, name) == serial


def test_shards_that_split_every_points_seeds_fold_to_the_serial_rows(
    tmp_path,
):
    """The additive record's whole point: a folded point is the point.

    Every point of the sweep is split, seed 0 to one shard and seed 1
    to the other, so no shard holds a whole point and every summary
    column has to come out of the additive files.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    forwards_dir = tmp_path / "forwards"
    backwards_dir = tmp_path / "backwards"
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(first_dir),
            "--shots-per-unit",
            "1",
            "--shard",
            "0/2",
        ]
    )
    command.main(
        [
            "collect",
            str(config_path),
            "--out",
            str(second_dir),
            "--shots-per-unit",
            "1",
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
            str(forwards_dir),
        ]
    )
    command.main(
        [
            "combine",
            str(second_dir),
            str(first_dir),
            "--out",
            str(backwards_dir),
        ]
    )

    for name in EVERY_FILE:
        serial = _rows_without_wall_clock(serial_dir, name)
        assert _rows_without_wall_clock(forwards_dir, name) == serial
    for name in EVERY_FILE:
        forwards_path = forwards_dir / name
        backwards_path = backwards_dir / name
        assert forwards_path.read_bytes() == backwards_path.read_bytes()


def test_a_combined_folder_folds_again_with_a_later_shard(tmp_path):
    """A Slurm array finishing in waves folds each wave as it lands.

    So a combined folder is a run folder: the additive files plus a
    manifest recording the sweep, and folding it with the last shard
    gives the rows of the run that never sharded at all.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    serial_dir = tmp_path / "serial"
    shard_dirs = []
    command.main(["collect", str(config_path), "--out", str(serial_dir)])
    for index in (0, 1, 2):
        shard_dir = tmp_path / f"shard{index}"
        shard_dirs.append(shard_dir)
        _collect_one_shard(config_path, shard_dir, f"{index}/3", 1)
    first_two_dir = tmp_path / "first_two"
    combined_dir = tmp_path / "combined"
    _combined(shard_dirs[0], shard_dirs[1], first_two_dir)
    _combined(first_two_dir, shard_dirs[2], combined_dir)
    for name in EVERY_FILE:
        serial = _rows_without_wall_clock(serial_dir, name)
        assert _rows_without_wall_clock(combined_dir, name) == serial


def test_a_manifest_records_the_shard_and_the_unit_size_it_ran(tmp_path):
    """How a folder ran, beside what it ran: the array's own bookkeeping.

    Nothing reads these to fold the rows, which come back in the order
    the recorded sweep gives; they say what one of a hundred folders an
    array left behind holds.
    """
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    whole_dir = tmp_path / "whole"
    shard_dir = tmp_path / "shard"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(whole_dir)])
    _collect_one_shard(config_path, shard_dir, "0/2", 1)
    command.main(["combine", str(whole_dir), "--out", str(combined_dir)])
    whole = _manifest_of(whole_dir)
    sharded = _manifest_of(shard_dir)
    combined = _manifest_of(combined_dir)
    assert whole["shard"] is None
    assert whole["shots_per_unit"] is None
    assert sharded["shard"] == "0/2"
    assert sharded["shots_per_unit"] == 1
    assert combined["folded"] == [str(whole_dir)]
    assert combined["resolved_config"] == whole["resolved_config"]


def test_a_shard_with_no_work_unit_says_so_and_writes_no_rows(tmp_path, capsys):
    """A Slurm array wider than the sweep's units is a shape, not a fault."""
    config_path = yaml_configs.write_config(tmp_path, {})
    out_dir = tmp_path / "empty"
    capsys.readouterr()
    command.main(
        ["collect", str(config_path), "--out", str(out_dir), "--shard", "1/2"]
    )

    printed = capsys.readouterr()
    manifest_path = out_dir / "manifest.json"
    sweep_path = out_dir / "sweep.csv"
    assert manifest_path.is_file()
    assert not sweep_path.exists()
    assert "wrote no rows beyond its manifest" in printed.err


def test_combine_skips_a_folder_that_ran_no_shot(tmp_path, capsys):
    config_path = yaml_configs.write_config(tmp_path, {})
    whole_dir = tmp_path / "whole"
    empty_dir = tmp_path / "empty"
    combined_dir = tmp_path / "combined"
    command.main(
        ["collect", str(config_path), "--out", str(whole_dir), "--shard", "0/2"]
    )
    command.main(
        ["collect", str(config_path), "--out", str(empty_dir), "--shard", "1/2"]
    )
    capsys.readouterr()
    command.main(
        [
            "combine",
            str(whole_dir),
            str(empty_dir),
            "--out",
            str(combined_dir),
        ]
    )

    printed = capsys.readouterr()
    assert "has no shots.csv, so combine skips it" in printed.err
    whole = _rows_without_wall_clock(whole_dir, "sweep.csv")
    combined = _rows_without_wall_clock(combined_dir, "sweep.csv")
    assert combined == whole


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


def test_one_points_seeds_divide_across_two_shards(tmp_path):
    """The unit, not the point, is what a shard selects.

    This is the whole reason --shots-per-unit exists: a sweep of one
    point can still fill a Slurm array. Its law is that shard i of n
    runs the units whose position modulo n is i, so at one shot to a
    unit the even seeds go to shard 0 of 2 and the odd seeds to shard 1.
    """
    one_point = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_us": [1.0],
                "shots": 4,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, one_point)
    first_dir = tmp_path / "shard0"
    second_dir = tmp_path / "shard1"
    for index, out_dir in ((0, first_dir), (1, second_dir)):
        _collect_one_shard(config_path, out_dir, f"{index}/2", 1)
    first_seeds = _seeds_of_every_shot(first_dir)
    second_seeds = _seeds_of_every_shot(second_dir)
    assert first_seeds == ["0", "2"]
    assert second_seeds == ["1", "3"]


def test_a_shard_outside_its_count_is_refused(tmp_path, capsys):
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(out_dir),
                "--shard",
                "5/2",
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: --shard 5/2 is not a shard")
    assert not out_dir.exists()


@pytest.mark.parametrize("given", ["0", "-1"])
def test_a_unit_size_below_one_is_refused(tmp_path, capsys, given):
    """Zero steps `range` by nothing; -1 would run no unit at all."""
    config_path = yaml_configs.write_config(tmp_path, FOUR_POINT_SWEEP)
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "collect",
                str(config_path),
                "--out",
                str(out_dir),
                "--shots-per-unit",
                given,
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(
        f"decsim: --shots-per-unit {given} is not a unit size"
    )
    assert not out_dir.exists()


def test_combining_two_folders_that_hold_the_same_shot_is_refused(
    tmp_path, capsys
):
    config_path = yaml_configs.write_config(tmp_path, {})
    run_dir = tmp_path / "one"
    combined_dir = tmp_path / "combined"
    command.main(["collect", str(config_path), "--out", str(run_dir)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        command.main(
            [
                "combine",
                str(run_dir),
                str(run_dir),
                "--out",
                str(combined_dir),
            ]
        )

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "is in more than one of" in printed.err


def test_run_refuses_a_yaml_that_is_not_there(tmp_path, capsys):
    missing = tmp_path / "not_a_config.yaml"
    with pytest.raises(SystemExit) as stopped:
        command.main(["run", str(missing)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith(f"decsim: {missing} is not a file")
    assert "reference" in printed.err


def test_show_refuses_a_sweep_axis_the_yaml_layer_does_not_have(
    tmp_path, capsys
):
    unknown_axis = {
        "sweep": [
            {
                "physical_error_probability": [0.001],
                "distance": [3],
                "round_period_us": [1.0],
                "algorithm": ["pymatching"],
                "shots": 1,
            }
        ]
    }
    config_path = yaml_configs.write_config(tmp_path, unknown_axis)
    with pytest.raises(SystemExit) as stopped:
        command.main(["show", str(config_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "sweep block 1 does not know ['algorithm']" in printed.err


def test_plot_refuses_a_figure_it_does_not_draw(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        command.main(["plot", str(tmp_path), "--figure", "everything"])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: no figure named everything")


def test_trace_refuses_an_action_it_does_not_have(tmp_path, capsys):
    trace_path = tmp_path / "shot.json"
    with pytest.raises(SystemExit) as stopped:
        command.main(["trace", "summarise", str(trace_path)])

    printed = capsys.readouterr()
    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert printed.err.startswith("decsim: decsim trace has no action")


def test_a_refused_combine_leaves_no_folder(tmp_path):
    """The out folder is made once the fold is accepted, not before."""
    missing = tmp_path / "never_ran"
    out_dir = tmp_path / "combined"
    with pytest.raises(SystemExit) as exit_info:
        command.main(["combine", str(missing), "--out", str(out_dir)])
    assert exit_info.value.code == 1
    assert not out_dir.exists()
