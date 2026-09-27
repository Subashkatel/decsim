"""A piece: seeds [first, first + count) of one sweep point, kept as one folder.

pieces/<point id>/<first>-<last>/ holds the piece's additive files and
piece.json. They are written into a hidden staging folder beside it,
piece.json last, and the folder is renamed into place last, so a piece
folder exists only whole: a rename within one directory is atomic under
POSIX, and GPFS keeps POSIX semantics under its distributed locking
(Schmuck and Haskin, FAST 2002). A run skips a piece whose folder
exists, which is how a killed collect resumes, as sinter resumes from
the counts its save file already holds
(sinter/_collection/_collection.py:387-397). A staging folder a killed
writer left is no piece and is passed over; no writer can tell from
another host whether its owner is still writing, so none removes it.
"""

import json
import os
import pathlib
import shutil
import uuid

import decsim.experiments.report as report
import decsim.experiments.residence as residence
import decsim.experiments.run_folder as run_folder

PIECES_FOLDER = "pieces"
PIECE_FILE = "piece.json"


def piece_dir(
    experiment_dir: pathlib.Path, point_id: str, first_seed: int, count: int
) -> pathlib.Path:
    """Where the piece's folder is: its point, then its first and last seed."""
    last_seed = first_seed + count - 1
    point_dir = experiment_dir / PIECES_FOLDER / point_id
    return point_dir / f"{first_seed}-{last_seed}"


def write(
    experiment_dir: pathlib.Path,
    point_id: str,
    first_seed: int,
    measurements: list,
    facts: dict,
) -> pathlib.Path:
    """One piece's files, whole or not at all, and where they went.

    facts are the piece's own lines of piece.json, beside the counts
    read off its shots.
    """
    count = len(measurements)
    folder = piece_dir(experiment_dir, point_id, first_seed, count)
    staging = _staging_dir(folder)
    staging.mkdir(parents=True)
    record = report.record_of(measurements)
    report.write_record(record, staging, None)
    residence_rows = residence.rows_of(measurements)
    if residence_rows:
        residence_path = staging / residence.PIECE_FILE
        report.write_csv(residence_rows, residence_path)
    counts = _counts_of(record.shots)
    identity = run_folder.piece_identity()
    piece = {
        "point_id": point_id,
        "first_seed": first_seed,
        "count": count,
        **counts,
        **facts,
        **identity,
    }
    piece_path = staging / PIECE_FILE
    run_folder.write_json(piece_path, piece)
    _publish(staging, folder)
    return folder


def folders_of(experiment_dir: pathlib.Path, point_ids: list) -> list:
    """Every whole piece of these points, point by point, in seed order.

    A staging folder is no piece and is passed over.
    """
    folders = []
    for point_id in point_ids:
        point_dir = experiment_dir / PIECES_FOLDER / point_id
        if not point_dir.is_dir():
            continue
        written = _whole_pieces(point_dir)
        folders.extend(written)
    return folders


def saved_counts(experiment_dir: pathlib.Path, point_id: str) -> dict:
    """One point's whole pieces, each first seed mapped to its count."""
    counts = {}
    for folder in folders_of(experiment_dir, [point_id]):
        first_seed, count = _range_of(folder)
        counts[first_seed] = count
    return counts


def seed_ranges_of(folders: list) -> dict:
    """Each point's seed ranges, [first, count], from its pieces' names."""
    ranges = {}
    for folder in folders:
        point_id = folder.parent.name
        first_seed, count = _range_of(folder)
        point_ranges = ranges.setdefault(point_id, [])
        point_ranges.append((first_seed, count))
    joined = {}
    for point_id, point_ranges in ranges.items():
        joined[point_id] = run_folder.seed_ranges(point_ranges)
    return joined


def read_piece(folder: pathlib.Path) -> dict:
    """One piece's piece.json."""
    piece_path = folder / PIECE_FILE
    text = piece_path.read_text()
    return json.loads(text)


def _staging_dir(folder: pathlib.Path) -> pathlib.Path:
    """A hidden folder beside the piece that only this writer uses.

    Two tasks handed the same piece may write it at once, so each
    stages its own copy under a random name and neither touches the
    other's files.
    """
    identifier = uuid.uuid4()
    token = identifier.hex
    return folder.with_name(f".{folder.name}.{token}.partial")


def _publish(staging: pathlib.Path, folder: pathlib.Path) -> None:
    """The staged piece renamed into place, or dropped when one is there.

    A rename onto a folder that holds files fails (rename(2), ENOTEMPTY),
    so of two writers of one piece the first to rename wins. The other's
    copy holds the same seeds of the same point, the same piece, and is
    dropped.
    """
    try:
        os.replace(staging, folder)
    except OSError:
        if not folder.is_dir():
            raise
        shutil.rmtree(staging)


def _whole_pieces(point_dir: pathlib.Path) -> list:
    """One point's piece folders in first-seed order, staging ones left out."""
    pieces = []
    for folder in point_dir.iterdir():
        if folder.name.startswith("."):
            continue
        pieces.append(folder)
    return sorted(pieces, key=_range_of)


def _range_of(folder: pathlib.Path) -> tuple:
    """(first seed, count) from a piece folder's <first>-<last> name."""
    first_text, last_text = folder.name.split("-")
    first_seed = int(first_text)
    last_seed = int(last_text)
    count = last_seed - first_seed + 1
    return (first_seed, count)


def _counts_of(shots: list) -> dict:
    """The piece's shot counts, the lines a planner reads without its rows.

    core_seconds is the shots' own simulated wall time, the cost a
    planner deals pieces by. The window status counts are the summary's
    own sums (report.STATUS_SUMS), so a status reads without the rows
    why a piece's shots went unscored, as sinter keeps its discards
    beside its shots (sinter/_data/_task_stats.py:71).
    """
    scored_shots = _sum_of(shots, "is_scored")
    counts = {
        "scored_shots": scored_shots,
        "failures": _sum_of(shots, "logical_failure"),
        "unscored_shots": len(shots) - scored_shots,
        "core_seconds": _sum_of(shots, "sim_wall_seconds"),
    }
    for name in report.STATUS_SUMS:
        counts[name] = _sum_of(shots, name)
    return counts


def _sum_of(shots: list, column: str):
    """One column summed over the piece's shot rows."""
    values = []
    for row in shots:
        values.append(row[column])
    return sum(values)
