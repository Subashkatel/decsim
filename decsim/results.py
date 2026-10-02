"""Results folders read back as one table.

A row per sweep point beside its settings is sinter's shape: its
read_stats_from_csv_files gives one TaskStats per task with the task's
json metadata beside its counts (Stim
glue/sample/src/sinter/_data/_existing_data.py:135-142). The figure is
the reader's own, drawn with decsim.plots.
"""

import pathlib
from typing import Union

import decsim.experiments.report as report
import decsim.experiments.run_folder as run_folder


def load(*run_dirs: Union[str, pathlib.Path]) -> list:
    """Every sweep point of the folders, one row each, results and settings.

    A row holds its folder (run_dir), its sweep.csv columns read back as
    numbers, and every setting its point ran with, from the point's
    machine.json, as settings.<path> columns.
    """
    rows = []
    for run_dir_name in run_dirs:
        run_dir = pathlib.Path(run_dir_name)
        records = run_folder.point_records(run_dir)
        results = report.rows_by_point(run_dir)
        for point, result in results.items():
            record = records.get(point, {})
            settings = run_folder.resolved_values(record, ("settings",))
            row = {"run_dir": str(run_dir)}
            row.update(result)
            row.update(settings)
            rows.append(row)
    return rows
