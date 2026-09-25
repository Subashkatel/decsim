"""Run folders read back as one table, drawn and saved.

A row per sweep point beside its settings is sinter's shape: its
read_stats_from_csv_files gives one TaskStats per task with the task's
json metadata beside its counts (Stim
glue/sample/src/sinter/_data/_existing_data.py:135-142).
"""

import inspect
import pathlib
from collections.abc import Mapping
from typing import Optional, Union

import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder


def load(*run_dirs: Union[str, pathlib.Path]) -> list:
    """Every sweep point of the folders, one row each, results and settings.

    A row holds its folder (run_dir), its sweep.csv columns read back as
    numbers, and every setting its point ran with, from the point's
    resolved/ file, as settings.<path> columns.
    """
    rows = []
    for run_dir_name in run_dirs:
        run_dir = pathlib.Path(run_dir_name)
        resolved = run_folder.resolved_by_point(run_dir)
        results = report.rows_by_point(run_dir)
        for point, result in results.items():
            record = resolved.get(point, {})
            settings = run_folder.resolved_values(record, ("settings",))
            row = {"run_dir": str(run_dir)}
            row.update(result)
            row.update(settings)
            rows.append(row)
    return rows


def plot_error_rate(
    *,
    ax,
    rows: list,
    x: str,
    group: Optional[str] = None,
    where: Optional[Mapping] = None,
) -> None:
    """The logical error rate against column x, a curve per value of group.

    where keeps the rows whose columns hold the values it names. Each
    curve carries its Wilson 95% band, as sinter shades a likelihood
    region around each curve (sinter 1.16.0 _plotting.py:317-330).
    """
    kept = _kept_rows(rows, where)
    for label, curve in _curves(kept, group):
        curve.sort(key=lambda row: row[x])
        positions = [row[x] for row in curve]
        rates = [row["logical_error_rate"] for row in curve]
        lows = [row["ler_wilson_low"] for row in curve]
        highs = [row["ler_wilson_high"] for row in curve]
        lines = ax.plot(positions, rates, marker="o", label=label)
        color = lines[0].get_color()
        ax.fill_between(positions, lows, highs, color=color, alpha=0.2)
    ax.set_xlabel(x)
    ax.set_ylabel("logical_error_rate")
    if group is not None:
        ax.legend(title=group)


def save_figure(
    figure, path: Union[str, pathlib.Path], rows: list, input_folders: list
) -> None:
    """The picture, and beside it the script, rows and folders that made it.

    So a figure can be drawn again from its own folder.
    """
    path = pathlib.Path(path)
    figure.savefig(path, dpi=150)
    frames = inspect.stack(context=0)
    script = pathlib.Path(frames[1].filename)
    if script.is_file():
        script_text = script.read_text()
        script_copy = path.with_suffix(".py")
        script_copy.write_text(script_text)
    numbers_path = path.with_suffix(".csv")
    report.write_csv(rows, numbers_path)
    folders = [str(folder) for folder in input_folders]
    record = {"script": str(script), "input_folders": folders}
    record_path = path.with_suffix(".json")
    run_folder.write_json(record_path, record)


def _kept_rows(rows: list, where: Optional[Mapping]) -> list:
    if where is None:
        return list(rows)
    kept = []
    for row in rows:
        if _holds(row, where):
            kept.append(row)
    return kept


def _holds(row: Mapping, where: Mapping) -> bool:
    """Whether a row's columns hold every value where names."""
    for column, value in where.items():
        if row.get(column) != value:
            return False
    return True


def _curves(rows: list, group: Optional[str]) -> list:
    """(label, rows) per value of the group column, in first-seen order."""
    curves = {}
    for row in rows:
        group_value = row.get(group)
        curve = curves.setdefault(group_value, [])
        curve.append(row)
    labelled = []
    for group_value, curve in curves.items():
        label = None
        if group is not None:
            label = f"{group}={group_value}"
        labelled.append((label, curve))
    return labelled
