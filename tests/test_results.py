"""decsim.results and `decsim diff`: run folders read back and compared.

Two runs of one yaml whose decoder is priced by a card are the same
run, number for number; a run at another shot length differs in one
setting and in the circuit it ran. The folders come from `decsim
collect` itself, so the reader is pinned against what the writer wrote.
"""

import csv
import json
import pathlib
import shutil
import subprocess
import sys

import pytest
from matplotlib.figure import Figure

import decsim.experiments.command as command
import decsim.results as results
import tests.experiments.yaml_configs as yaml_configs

TWO_POINT_SWEEP = [
    {
        "physical_error_probability": [0.003, 0.01],
        "distance": [3],
        "round_period_microseconds": [1.0],
        "shots": 20,
    }
]

ROUNDS_COLUMN = "settings.workload.row_settings.arguments.rounds_per_shot"


def _collected(folder, overrides: dict):
    """One `decsim collect` of the minimal config into its own folder."""
    folder.mkdir()
    sweep = {"sweep": TWO_POINT_SWEEP}
    sweep.update(overrides)
    config_path = yaml_configs.write_config(folder, sweep)
    run_dir = folder / "run"
    command.main(["collect", str(config_path), "--out", str(run_dir)])
    return run_dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs of one yaml, and one at another shot length."""
    root = tmp_path_factory.mktemp("runs")
    first_folder = root / "first"
    second_folder = root / "second"
    shorter_folder = root / "shorter"
    shorter_workload = yaml_configs.memory_workload(12)
    shorter_overrides = {"workload": shorter_workload}
    first = _collected(first_folder, {})
    second = _collected(second_folder, {})
    shorter = _collected(shorter_folder, shorter_overrides)
    return {"first": first, "second": second, "shorter": shorter}


def _diff_printed(capsys, first, second) -> list:
    command.main(["diff", str(first), str(second)])
    printed = capsys.readouterr()
    return printed.out.splitlines()


def _csv_rows(path: pathlib.Path) -> list:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_csv_rows(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rewritten(run_dir, tmp_path, name: str, values: dict):
    """A copy of a run folder whose first sweep.csv row holds values."""
    destination = tmp_path / name
    shutil.copytree(run_dir, destination)
    sweep_path = destination / "sweep.csv"
    rows = _csv_rows(sweep_path)
    rows[0].update(values)
    _write_csv_rows(sweep_path, rows)
    return destination


def test_two_runs_of_one_yaml_are_the_same(runs, capsys):
    lines = _diff_printed(capsys, runs["first"], runs["second"])

    assert lines == [
        "settings:",
        "  the same",
        "inputs:",
        "  the same",
        "results:",
        "  the same",
    ]


def test_diff_names_the_setting_and_the_input_that_changed(runs, capsys):
    lines = _diff_printed(capsys, runs["first"], runs["shorter"])

    point = "p 0.003 d 3 round period 1.0 us:"
    assert (
        f"  {point} settings.workload.row_settings.arguments."
        "rounds_per_shot: 15 -> 12"
    ) in lines
    assert f"  {point} operation_1.stim: sha256 differs" in lines
    assert f"  {point} operations.json: sha256 differs" in lines


def test_diff_judges_a_logical_error_rate_by_its_wilson_interval(
    runs, tmp_path, capsys
):
    first_rows = results.load(runs["first"])
    rate = first_rows[0]["logical_error_rate"]
    nudged_rate = rate + 1e-9
    inside = _rewritten(
        runs["first"], tmp_path, "inside", {"logical_error_rate": nudged_rate}
    )
    apart = _rewritten(
        runs["first"],
        tmp_path,
        "apart",
        {
            "logical_error_rate": 0.9,
            "ler_wilson_low": 0.85,
            "ler_wilson_high": 0.95,
        },
    )

    inside_lines = _diff_printed(capsys, runs["first"], inside)
    apart_lines = _diff_printed(capsys, runs["first"], apart)

    assert inside_lines[-1].endswith(", within error bars")
    assert apart_lines[-1].endswith(", beyond error bars")
    assert len(apart_lines) == 6


def _spread_shots(run_dir, column: str, half_width: float) -> None:
    """The first point's shots of a column pushed apart, their mean kept.

    Alternate shots move up and down by half_width, so the column's
    standard error over the point's 20 shots is half_width times
    sqrt(20 / 19) / sqrt(20), about 0.23 half_width.
    """
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    first_point = rows[0]["physical_error_probability"]
    sign = 1
    for row in rows:
        if row["physical_error_probability"] != first_point:
            continue
        value = float(row[column])
        row[column] = value + sign * half_width
        sign = -sign
    _write_csv_rows(shots_path, rows)


@pytest.mark.parametrize(
    "column, shot_column",
    [("load", "load"), ("windows_per_shot", "windows")],
)
def test_diff_judges_a_mean_by_the_standard_error_of_its_shots(
    runs, tmp_path, capsys, column, shot_column
):
    """A mean moved by 1 standard error agrees; one moved by 10 does not.

    windows_per_shot is the mean of shots.csv's windows column.
    """
    first_rows = results.load(runs["first"])
    mean = first_rows[0][column]
    standard_error = 0.01 * (20 / 19) ** 0.5 / 20**0.5
    near_mean = mean + standard_error
    far_mean = mean + 10 * standard_error
    near = _rewritten(runs["first"], tmp_path, "near", {column: near_mean})
    far = _rewritten(runs["first"], tmp_path, "far", {column: far_mean})
    _spread_shots(near, shot_column, 0.01)
    _spread_shots(far, shot_column, 0.01)

    near_lines = _diff_printed(capsys, runs["first"], near)
    far_lines = _diff_printed(capsys, runs["first"], far)

    assert near_lines[-1].endswith(", within error bars")
    assert far_lines[-1].endswith(", beyond error bars")
    assert f"{column}:" in far_lines[-1]


def test_diff_compares_a_column_with_no_error_bar_exactly(
    runs, tmp_path, capsys
):
    """The largest over shots has no standard error, though shots hold it."""
    changed = _rewritten(
        runs["first"], tmp_path, "changed", {"max_queued_windows": 999}
    )

    lines = _diff_printed(capsys, runs["first"], changed)

    assert lines[-1].endswith(", no error bar: compared exactly")


def _kept_shots(run_dir, keep) -> None:
    """The run's shots.csv rows that keep(row) holds, the rest dropped."""
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    kept = []
    for row in rows:
        if keep(row):
            kept.append(row)
    _write_csv_rows(shots_path, kept)


def _first_shot_only(runs, tmp_path, name: str, values: dict):
    """A copy whose first point kept one shot and whose row holds values."""
    changed = _rewritten(runs["first"], tmp_path, name, values)
    first_rows = results.load(runs["first"])
    first_point = str(first_rows[0]["physical_error_probability"])
    seen = []

    def keep(row) -> bool:
        if row["physical_error_probability"] != first_point:
            return True
        seen.append(row)
        return len(seen) == 1

    _kept_shots(changed, keep)
    return changed


def test_diff_gives_a_mean_over_one_shot_no_error_bar(runs, tmp_path, capsys):
    """A standard error needs two shots; one shot's mean is compared exactly."""
    first_rows = results.load(runs["first"])
    moved = first_rows[0]["load"] + 1.0
    changed = _first_shot_only(runs, tmp_path, "one_shot", {"load": moved})

    lines = _diff_printed(capsys, runs["first"], changed)

    assert lines[-1].endswith(", no error bar: compared exactly")
    assert "load:" in lines[-1]


def _blank_shot_column(run_dir, column: str) -> None:
    """The run's shots.csv with one column's values left empty."""
    shots_path = run_dir / "shots.csv"
    rows = _csv_rows(shots_path)
    for row in rows:
        row[column] = ""
    _write_csv_rows(shots_path, rows)


def test_diff_gives_a_mean_its_shots_do_not_hold_no_error_bar(
    runs, tmp_path, capsys
):
    """A run that kept no per-shot values of a mean has no error bar for it."""
    first_rows = results.load(runs["first"])
    moved = first_rows[0]["load"] + 1.0
    changed = _rewritten(runs["first"], tmp_path, "unkept", {"load": moved})
    _blank_shot_column(changed, "load")

    lines = _diff_printed(capsys, runs["first"], changed)

    assert lines[-1].endswith(", no error bar: compared exactly")


def test_diff_names_a_column_only_the_second_folder_holds(
    runs, tmp_path, capsys
):
    changed = _rewritten(runs["first"], tmp_path, "wider", {"extra": 5})

    lines = _diff_printed(capsys, runs["first"], changed)

    point = "p 0.003 d 3 round period 1.0 us:"
    assert (
        f"  {point} extra: None -> 5, no error bar: compared exactly" in lines
    )


def test_diff_names_a_point_only_one_folder_holds(runs, tmp_path, capsys):
    fewer = tmp_path / "fewer"
    shutil.copytree(runs["first"], fewer)
    sweep_path = fewer / "sweep.csv"
    sweep_text = sweep_path.read_text()
    sweep_lines = sweep_text.splitlines()
    kept_text = "\n".join(sweep_lines[:2]) + "\n"
    sweep_path.write_text(kept_text)

    first_lines = _diff_printed(capsys, runs["first"], fewer)
    second_lines = _diff_printed(capsys, fewer, runs["first"])

    point = "p 0.01 d 3 round period 1.0 us:"
    assert f"  {point} only in the first folder" in first_lines
    assert f"  {point} only in the second folder" in second_lines


def test_diff_refuses_a_folder_without_its_resolved_settings(
    runs, tmp_path, capsys
):
    """Without resolved/ the settings would read as never differing."""
    bare = tmp_path / "bare"
    shutil.copytree(runs["first"], bare)
    resolved_dir = bare / "resolved"
    shutil.rmtree(resolved_dir)

    with pytest.raises(SystemExit) as stopped:
        command.main(["diff", str(runs["shorter"]), str(bare)])
    printed = capsys.readouterr()

    assert stopped.value.code == 1
    assert printed.err.count("\n") == 1
    assert "holds no resolved/ folder" in printed.err


def test_load_gives_a_row_per_point_with_its_results_and_settings(runs):
    rows = results.load(runs["first"], runs["shorter"])

    rounds = [row[ROUNDS_COLUMN] for row in rows]
    assert len(rows) == 4
    assert rounds == [15, 15, 12, 12]
    assert rows[0]["run_dir"] == str(runs["first"])
    assert rows[0]["shots"] == 20


def _point_count(line) -> int:
    positions = line.get_xdata()
    return len(positions)


def test_the_plot_draws_a_labelled_curve_per_group_of_the_kept_rows(runs):
    """One probability is kept; each shot length is its own curve."""
    rows = results.load(runs["first"], runs["shorter"])
    figure = Figure()
    ax = figure.subplots()

    results.plot_error_rate(
        ax=ax,
        rows=rows,
        x="physical_error_probability",
        group=ROUNDS_COLUMN,
        where={"physical_error_probability": 0.003},
    )

    labels = [line.get_label() for line in ax.lines]
    point_counts = [_point_count(line) for line in ax.lines]
    legend = ax.get_legend()
    legend_title = legend.get_title()
    assert labels == [f"{ROUNDS_COLUMN}=15", f"{ROUNDS_COLUMN}=12"]
    assert point_counts == [1, 1]
    assert legend_title.get_text() == ROUNDS_COLUMN


def test_a_saved_figure_keeps_its_script_numbers_and_folders(runs, tmp_path):
    folders = [runs["first"], runs["shorter"]]
    rows = results.load(*folders)
    figure = Figure()
    ax = figure.subplots()

    results.plot_error_rate(
        ax=ax,
        rows=rows,
        x="physical_error_probability",
        group=ROUNDS_COLUMN,
    )
    picture_path = tmp_path / "ler.png"
    results.save_figure(figure, picture_path, rows, folders)

    script_copy = tmp_path / "ler.py"
    numbers_path = tmp_path / "ler.csv"
    with open(numbers_path, newline="") as handle:
        reader = csv.DictReader(handle)
        drawn_rows = list(reader)
    record_path = tmp_path / "ler.json"
    record_text = record_path.read_text()
    record = json.loads(record_text)
    this_file = pathlib.Path(__file__)
    this_text = this_file.read_text()
    assert len(ax.lines) == 2
    assert picture_path.is_file()
    assert script_copy.read_text() == this_text
    assert len(drawn_rows) == 4
    assert record["input_folders"] == [str(folder) for folder in folders]


FIGURE_SCRIPT = """
from matplotlib.figure import Figure

import decsim.results as results

figure = Figure()
figure.subplots()
results.save_figure(figure, "plot.png", [{"distance": 3}], ["run"])
"""


def test_a_figure_named_after_its_script_keeps_the_script(tmp_path):
    """plot.py drawing plot.png is already the copy of its script."""
    script_path = tmp_path / "plot.py"
    script_path.write_text(FIGURE_SCRIPT)
    command_line = [sys.executable, str(script_path)]

    subprocess.run(command_line, cwd=tmp_path, check=True)

    numbers_path = tmp_path / "plot.csv"
    assert script_path.read_text() == FIGURE_SCRIPT
    assert numbers_path.read_text() == "distance\n3\n"
