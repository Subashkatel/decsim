"""Run folders read back as one table, and a figure saved with its source.

A row per sweep point beside its settings is sinter's shape: its
read_stats_from_csv_files gives one TaskStats per task with the task's
json metadata beside its counts (Stim
glue/sample/src/sinter/_data/_existing_data.py:135-142). The figure is
the reader's own; save_figure keeps the script, the rows and the
folders beside it.
"""

import inspect
import pathlib
from typing import Union

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
